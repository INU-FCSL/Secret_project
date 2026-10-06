"""저장 관측에서 latent distribution·평형 편향·대칭성을 계측한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel
from .config import ppo_config
from .normalization import configure_model, checkpoint_mode
from .distributions import checkpoint_distribution, inverse_squash
from .action_mapping import bounded_action
from .compare_standing import stats


def load_actor(checkpoint, device='cuda'):
    state=torch.load(checkpoint,weights_only=False,map_location=device)
    cfg=ppo_config(checkpoint_distribution(state.get('infos')))['actor']
    dimension=state['actor_state_dict']['mlp.4.weight'].shape[0]
    obs=TensorDict({'actor':torch.zeros((1,42),device=device)},batch_size=[1])
    actor=MLPModel(obs,{'actor':['actor']},'actor',dimension,hidden_dims=cfg['hidden_dims'],
        activation=cfg['activation'],obs_normalization=True,distribution_cfg=cfg['distribution_cfg']).to(device)
    actor.load_state_dict(state['actor_state_dict'])
    configure_model(actor,checkpoint_mode(state.get('infos')))
    actor.eval()
    return actor,state


@torch.inference_mode()
def diagnose(checkpoint, trace, equilibrium, output):
    actor,state=load_actor(checkpoint)
    mapping=state['infos'].get('action_mapping','clip')
    env=json.loads((Path(checkpoint).parent/'environment.json').read_text())
    reference=np.array(env['reference_orientation'][:2])
    data=np.load(trace);physical=np.load(equilibrium)
    raw=torch.tensor(data['raw'].reshape(-1,42),device='cuda')
    mean=actor.mlp(actor.obs_normalizer(raw))
    actor.distribution.update(mean)
    sigma=actor.output_std
    torch.manual_seed(1729)
    sample=actor.distribution.sample()
    latent_sample=inverse_squash(sample) if checkpoint_distribution(state['infos'])=='squashed' else sample
    sample_bounded=bounded_action(sample,mapping)
    command=bounded_action(actor.distribution.deterministic_output(mean),mapping)
    shape=data['raw'].shape[:2];m=mean.cpu().numpy().reshape(*shape,-1)
    s=sigma.cpu().numpy().reshape(*shape,-1)
    time=np.arange(shape[0])[:,None]*.02;start=data['push_start']
    phases=dict(before=time<start,during=(time>=start)&(time<start+.5),
        immediate_after=(time>=start+.5)&(time<start+1.5),late=time>=start+1.5)
    gravity=data['raw'][...,:3]
    rpy_observed=np.stack((np.arctan2(-gravity[...,1],-gravity[...,2]),
        np.arcsin(np.clip(gravity[...,0],-1,1))),axis=-1)
    observation_error=np.rad2deg(rpy_observed-reference)
    equilibrium_mask=(data['valid']&(np.abs(observation_error).max(-1)<.05)&
        (np.linalg.norm(data['raw'][...,3:6],axis=-1)<.005))
    result=dict(checkpoint=str(checkpoint),trace=str(trace),distribution=checkpoint_distribution(state['infos']),
        sigma=sigma[0].cpu().tolist(),latent_mean=stats(m[data['valid']],axis=0),
        sample=stats(sample.cpu().numpy(),axis=0),bounded_sample=stats(sample_bounded.cpu().numpy(),axis=0),
        latent_sample=stats(latent_sample.cpu().numpy(),axis=0),
        deterministic_action=stats(command.cpu().numpy(),axis=0),
        deterministic_90_fraction=(command.abs()>=.9).float().mean(0).cpu().tolist(),
        sampled_99_fraction=(sample_bounded.abs()>=.99).float().mean(0).cpu().tolist(),
        numerical_guard_fraction=(sample.abs()>=1-torch.finfo(sample.dtype).eps/2).float().mean().item(),
        entropy=actor.output_entropy.mean().item(),
        phases={name:dict(samples=int(mask.sum()),mean=stats(m[mask],axis=0),sigma=stats(s[mask],axis=0))
                for name,mask in phases.items()},
        equilibrium_trajectory=dict(samples=int(equilibrium_mask.sum()),angular_speed_limit=.005,
            mean=stats(m[equilibrium_mask],axis=0) if equilibrium_mask.any() else None))
    obs=torch.tensor(physical['raw'][None],device='cuda',dtype=torch.float32)
    equilibrium_mean=actor.mlp(actor.obs_normalizer(obs))
    equilibrium_action=bounded_action(actor.distribution.deterministic_output(equilibrium_mean),mapping)
    rpy=physical['rpy']
    coefficients=np.rad2deg(2*(reference-rpy[:2])-.2*physical['raw'][3:5])@physical['inverse'].T
    scripted=np.clip(np.clip(coefficients,-1,1)@physical['scripted_basis'].T,-1,1)
    matrix=np.array(state['infos']['standing_basis'])
    scripted=np.clip(scripted@np.linalg.pinv(matrix).T,-1,1)
    result['equilibrium_probe']=dict(mean=equilibrium_mean[0].cpu().tolist(),
        mean_abs=equilibrium_mean[0].abs().cpu().tolist(),action=equilibrium_action[0].cpu().tolist(),
        tanh_mean_abs=equilibrium_mean[0].tanh().abs().cpu().tolist(),
        scripted_action=scripted.tolist(),physical_reference_error_degrees=np.rad2deg(rpy[:2]-reference).tolist())
    symmetry={}
    for axis,name in enumerate(('roll','pitch')):
        outputs=[]
        for sign in (-1,1):
            o=obs.clone();q=rpy.copy();q[axis]+=sign*np.deg2rad(.05)
            o[:,0]=float(np.sin(q[1]));o[:,1]=float(-np.sin(q[0])*np.cos(q[1]));o[:,2]=float(-np.cos(q[0])*np.cos(q[1]))
            outputs.append(bounded_action(actor(TensorDict({'actor':o},batch_size=[1])),mapping)[0])
        minus,plus=outputs;common=(plus+minus)/2;corrective=(plus-minus)/2
        latent_axis=1 if name=='roll' else 0
        symmetry[name]=dict(error_degrees=.05,minus=minus.cpu().tolist(),plus=plus.cpu().tolist(),
            common_offset=common.cpu().tolist(),corrective_component=corrective.cpu().tolist(),
            odd_symmetry_error=(plus+minus).abs().cpu().tolist(),gain_per_degree=(corrective/.05).cpu().tolist(),
            same_axis_common=float(common[latent_axis]),same_axis_corrective=float(corrective[latent_axis]),
            common_exceeds_corrective=bool(common[latent_axis].abs()>corrective[latent_axis].abs()))
    result['symmetry']=symmetry
    result['persistent_bias_evidence']=bool(max(result['equilibrium_probe']['tanh_mean_abs'])>.25 and
        all(row['common_exceeds_corrective'] for row in symmetry.values()))
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'trajectory.npz',mu=m,sigma=s,time=time,push_start=start,
        equilibrium_mask=equilibrium_mask,**phases)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('distribution·평형·대칭 진단 완료:',checkpoint,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description='Standing distribution 진단')
    for name in ('checkpoint','trace','equilibrium','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    diagnose(args.checkpoint,args.trace,args.equilibrium,args.output)


if __name__=='__main__':
    main()
