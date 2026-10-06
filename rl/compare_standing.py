"""같은 평가 경로로 V2와 V3-A의 자세 복구·행동 분포를 비교한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import StandingCfg
from .evaluate import evaluate
from .diagnose_ppo import stats, correlation
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict
from .config import ppo_config
from .action_mapping import bounded_action
from .normalization import configure_model, checkpoint_mode
from .action_diagnostics import mode_decomposition


class ActionTrace:
    def __init__(self):
        self.rows=[]

    def __call__(self,env,obs,action,reward,done,extras,alive,time):
        d=extras['diagnostics']
        error=torch.stack((d['roll'],d['pitch']),-1)-env.reference_standing_orientation[:2]
        row=dict(requested=d.get('raw_joint_actions',action),policy_raw=action,
            bounded=d['bounded_actions'],applied=d['applied_actions'],target=d['targets'],
            actual=extras['terminal_observation']['actor'][:,6:18]+env.default_joint_position,
            error=error.rad2deg(),omega=obs['actor'][:,3:5],raw=obs['actor'],
            reward=reward,valid=alive,after=alive&(time>=d['push_start']),
            push_start=d['push_start'],terms=torch.stack(list(d['reward_terms'].values()),-1)*env.cfg.step_dt)
        self.rows.append({k:v.detach().cpu().numpy().copy() for k,v in row.items()})

    def arrays(self):
        return {k:np.stack([row[k] for row in self.rows]) for k in self.rows[0]}


def action_summary(data,action_mapping='clip'):
    mask=data['valid']
    requested=data['requested'][mask]
    policy_raw=data.get('policy_raw',data['requested'])[mask]
    policy_bounded=np.tanh(policy_raw) if action_mapping=='tanh' else policy_raw.clip(-1,1)
    applied=data['applied'][mask]
    clipped=requested.clip(-1,1)
    bounded=data['bounded'][mask] if 'bounded' in data else (np.tanh(requested) if action_mapping=='tanh' else clipped)
    gain=1-np.tanh(policy_raw)**2 if action_mapping=='tanh' else (np.abs(policy_raw)<1).astype(float)
    feedback={}
    for key in ('requested','bounded','applied'):
        if key=='bounded' and key not in data:
            continue
        knee=data[key][...,2::3]
        differential=np.stack(((knee[...,0]-knee[...,1]+knee[...,2]-knee[...,3])/4,
                               (knee[...,0]+knee[...,1]-knee[...,2]-knee[...,3])/4),-1)
        for j,axis in enumerate(('roll','pitch')):
            after=data['after']
            feedback[f'{key}_{axis}']=dict(error_correlation=correlation(data['error'][...,j][after],differential[...,j][after]) if after.any() else None,
                velocity_correlation=correlation(data['omega'][...,j][after],differential[...,j][after]) if after.any() else None,
                differential=stats(differential[...,j][after]))
    return dict(requested=stats(requested,axis=0),bounded=stats(bounded,axis=0),applied=stats(applied,axis=0),
        policy_raw=stats(policy_raw,axis=0),
        policy_bounded=stats(policy_bounded,axis=0),policy_boundary_fraction=float((np.abs(policy_bounded)>=.99).mean()),
        policy_positive_99=(policy_bounded>=.99).mean(0).tolist(),policy_negative_99=(policy_bounded<=-.99).mean(0).tolist(),
        joint_requested_outside_fraction=float((np.abs(requested)>1).mean()),
        outside_range_fraction=float((np.abs(policy_raw)>1).mean()),
        actual_clipping_fraction=float((np.abs(policy_raw)>1).mean()) if action_mapping=='clip' else 0.,
        action_mapping=action_mapping,bounded_90_fraction=float((np.abs(bounded)>=.9).mean()),
        bounded_99_fraction=float((np.abs(bounded)>=.99).mean()),
        bounded_positive_99=(bounded>=.99).mean(0).tolist(),bounded_negative_99=(bounded<=-.99).mean(0).tolist(),
        applied_positive_99=(applied>=.99).mean(0).tolist(),applied_negative_99=(applied<=-.99).mean(0).tolist(),
        local_gain=stats(gain,axis=0),gain_below_half=float((gain<.5).mean()),gain_below_tenth=float((gain<.1).mean()),
        modes={key:mode_decomposition(value) for key,value in [('requested',requested),('bounded',bounded),('applied',applied)]},
        boundary_applied_fraction=float((np.abs(applied)>=.99).mean()),
        target_degrees=stats(np.rad2deg(data['target'][mask]),axis=0),
        actual_degrees=stats(np.rad2deg(data['actual'][mask]),axis=0),feedback=feedback)


def aggregate(rows):
    result={}
    mean_keys=('peak_orientation_error','integrated_orientation_error','recovery_seconds','success_rate','survival_rate',
        'mean_return','saturation_fraction','four_foot_fraction','reset_recovery_rate','push_recovery_rate')
    for key in mean_keys:
        values=[r[key] for r in rows if r[key] is not None]
        result[key]=float(np.mean(values)) if values else None
    for key in ('self_contact_env_steps','unexpected_contact_env_steps'):
        result[key]=sum(r[key] for r in rows)
    result['peak_torque']=max(r['peak_torque'] for r in rows)
    result['episodes']=sum(r['episodes'] for r in rows)
    return result


@torch.no_grad()
def policy_probe(checkpoint,data,action_mapping='clip'):
    """같은 관측에서 자세·각속도만 바꿔 국소 피드백 부호를 확인한다."""
    actor_cfg=ppo_config()['actor']
    state=torch.load(checkpoint,weights_only=False,map_location='cuda')
    output_dim=state['actor_state_dict']['mlp.4.weight'].shape[0]
    raw=torch.tensor(data['raw'][data['valid']],device='cuda')
    observation=TensorDict({'actor':raw[:1]},batch_size=[1])
    actor=MLPModel(observation,{'actor':['actor']},'actor',output_dim,hidden_dims=actor_cfg['hidden_dims'],
        activation=actor_cfg['activation'],obs_normalization=True,distribution_cfg=actor_cfg['distribution_cfg']).cuda()
    actor.load_state_dict(state['actor_state_dict'])
    configure_model(actor,checkpoint_mode(state.get('infos')))
    actor.eval()
    normalized=actor.obs_normalizer(raw)
    groups={'gravity':(0,3),'angular_velocity':(3,6),'joint_position':(6,18),
            'joint_velocity':(18,30),'previous_action':(30,42)}
    observations={name:dict(raw=stats(raw[:,a:b].cpu().numpy(),axis=0),
        normalized=stats(normalized[:,a:b].cpu().numpy(),axis=0)) for name,(a,b) in groups.items()}
    warmup_path=Path(checkpoint).parent/'warmup_observations.pt'
    if warmup_path.exists():
        warmup=torch.load(warmup_path,weights_only=True,map_location='cpu')[:,30:42].numpy()
        previous=raw[:,30:42].cpu().numpy()
        low,high=warmup.min(0),warmup.max(0)
        outside=(previous<low)|(previous>high)
        mean,std=warmup.mean(0),warmup.std(0)
        observations['previous_action'].update(warmup_min=low.tolist(),warmup_max=high.tolist(),
            outside_warmup_range=float(outside.mean()),outside_warmup_range_per_joint=outside.mean(0).tolist(),
            outside_warmup_three_std=float((np.abs(previous-mean)>3*std).mean()))
    selected=torch.tensor(data['raw'][data['after']][::16],device='cuda')
    if len(selected)==0:
        return dict(observations=observations,local_feedback=None)
    gravity=selected[:,:3]
    rpy=torch.stack((torch.atan2(-gravity[:,1],-gravity[:,2]),torch.asin(gravity[:,0].clamp(-1,1))),-1)
    def differential(obs):
        command=bounded_action(actor(TensorDict({'actor':obs},batch_size=[len(obs)])),action_mapping)
        matrix=state.get('infos',{}).get('standing_basis')
        if matrix is not None:
            command=command@torch.tensor(matrix,device=command.device,dtype=command.dtype).T
        knees=command[:,2::3]
        return torch.stack(((knees[:,0]-knees[:,1]+knees[:,2]-knees[:,3])/4,
                            (knees[:,0]+knees[:,1]-knees[:,2]-knees[:,3])/4),-1)
    feedback={}
    for j,axis in enumerate(('roll','pitch')):
        outputs=[]
        for sign in (-1,1):
            obs=selected.clone();angles=rpy.clone();angles[:,j]+=sign*np.deg2rad(.05)
            obs[:,0]=angles[:,1].sin();obs[:,1]=-angles[:,0].sin()*angles[:,1].cos();obs[:,2]=-angles[:,0].cos()*angles[:,1].cos()
            outputs.append(differential(obs)[:,j])
        pose=(outputs[1]-outputs[0])/.1
        outputs=[]
        for sign in (-1,1):
            obs=selected.clone();obs[:,3+j]+=sign*.01
            outputs.append(differential(obs)[:,j])
        feedback[axis]=dict(position_per_degree=stats(pose.cpu().numpy()),
            velocity_per_rad_s=stats(((outputs[1]-outputs[0])/.02).cpu().numpy()))
    return dict(observations=observations,local_feedback=feedback)


def gate_decision(results):
    """25회 결과의 물리 오차·피드백·경계 편향으로 연장 여부를 판정한다."""
    policy=results['v3b_25'];zero=results['zero'];previous=results['v3a_25']
    metrics=policy['aggregate'];baseline=zero['aggregate'];old=previous['aggregate']
    feedback=policy['policy_probe']['local_feedback']
    correct=feedback['roll']['position_per_degree']['mean']>0 and feedback['pitch']['position_per_degree']['mean']<0
    clipping=policy['actions']['outside_range_fraction']
    severe_peak=metrics['peak_orientation_error']>1.5*baseline['peak_orientation_error']
    severe_integrated=metrics['integrated_orientation_error']>2*baseline['integrated_orientation_error']
    early_failure=severe_peak and severe_integrated and (not correct or clipping>.25)
    safe=metrics['survival_rate']==1 and metrics['self_contact_env_steps']==0 and metrics['unexpected_contact_env_steps']==0 and metrics['saturation_fraction']<.01
    promising=(metrics['peak_orientation_error']<.9*old['peak_orientation_error'] or
               metrics['integrated_orientation_error']<.9*old['integrated_orientation_error'] or
               metrics['peak_orientation_error']<=1.1*baseline['peak_orientation_error'] or
               correct or clipping<.5*previous['actions']['outside_range_fraction'])
    return dict(extend_to_100=bool(safe and promising and not early_failure),early_failure=bool(early_failure),
                severe_peak=bool(severe_peak),severe_integrated=bool(severe_integrated),
                correct_local_feedback=bool(correct),clipping_fraction=clipping,safe=bool(safe),promising=bool(promising),
                thresholds=dict(severe_peak_ratio=1.5,severe_integrated_ratio=2.,high_clipping=.25,
                                meaningful_v3a_ratio=.9,near_zero_peak_ratio=1.1),
                reason='큰 물리 오차와 잘못된 피드백 또는 높은 clipping이면 연장하지 않습니다. 안전한 개선 징후가 있을 때만 연장합니다.')


def compare(v2,v3,output,*,v3b=None,checkpoints=(0,25,50,75,100),legacy_checkpoints=(0,25,50,75,100),orientation_reward_scale=.05):
    output.mkdir(parents=True,exist_ok=True)
    policies=[('zero',None),('random',None),('scripted',None)]
    for prefix,directory in [('v2',v2),('v3a',v3)]:
        policies.extend((f'{prefix}_{i}',directory/f'checkpoint_{i}.pt') for i in legacy_checkpoints)
    if v3b is not None:
        policies.extend((f'v3b_{i}',v3b/f'checkpoint_{i}.pt') for i in checkpoints)
    results={};base_initial=None;base_push=None
    for name,path in policies:
        rows=[];traces=[]
        for seed in (2026,2027,2028):
            cfg=StandingCfg(num_envs=16,seed=seed,stage=2,v2_stage='A',smoothing_seconds=.15,
                            episode_seconds=10.,balanced_push_directions=True,
                            orientation_reward_scale=orientation_reward_scale)
            recorder=ActionTrace()
            row=evaluate(cfg,checkpoint=path,random=name=='random',scripted=name=='scripted',step_callback=recorder)
            arrays=recorder.arrays()
            rows.append(row);traces.append(arrays)
        data={k:np.concatenate([trace[k] for trace in traces],axis=1) for k in traces[0]}
        np.savez_compressed(output/f'{name}.npz',**data)
        if base_initial is None:
            base_initial=data['raw'][0].copy();base_push=data['push_start'][0].copy()
        if not np.array_equal(base_initial,data['raw'][0]) or not np.array_equal(base_push,data['push_start'][0]):
            raise RuntimeError('제어기별 초기 관측 또는 외란 시작 시점이 다릅니다.')
        summary=dict(per_seed=rows,aggregate=aggregate(rows),actions=action_summary(data),matched_initial_and_push=True,
                     orientation_reward_scale=orientation_reward_scale)
        if path is not None:
            state=torch.load(path,weights_only=False,map_location='cpu')
            actor_state=state['actor_state_dict']
            std=actor_state['distribution.std_param'].numpy()
            summary['action_std']=std.tolist()
            summary['normalizer_count']=int(actor_state['obs_normalizer.count'])
            summary['checkpoint']=str(path)
            summary['policy_probe']=policy_probe(path,data)
        results[name]=summary
        (output/'comparison.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        print(f"평가 완료: {name}, 최대={summary['aggregate']['peak_orientation_error']:.6f}°, 적분={summary['aggregate']['integrated_orientation_error']:.6f}",flush=True)
    final=results[f'v3b_{max(checkpoints)}'] if v3b is not None else results['v3a_100']
    zero=results['zero']
    per_seed=[]
    for row,baseline in zip(final['per_seed'],zero['per_seed']):
        safe=row['survival_rate']==1 and row['self_contact_env_steps']==0 and row['unexpected_contact_env_steps']==0 and row['saturation_fraction']<.01
        per_seed.append(dict(seed=row['seed'],peak_improved=row['peak_orientation_error']<baseline['peak_orientation_error'],
            integrated_improved=row['integrated_orientation_error']<baseline['integrated_orientation_error'],safe=safe))
    verdict=dict(success=all(r['peak_improved'] and r['integrated_improved'] and r['safe'] for r in per_seed),per_seed=per_seed,
        boundary_bias=final['actions']['outside_range_fraction']>.5,
        training_seed_difference='버전별 학습 seed가 달라 변경 효과의 학습 재현성은 추가 학습 seed 검증이 필요합니다.')
    (output/'verdict.json').write_text(json.dumps(verdict,indent=2,allow_nan=False))
    if 'v3b_25' in results and 'v3a_25' in results:
        (output/'gate.json').write_text(json.dumps(gate_decision(results),indent=2,allow_nan=False))
    return results


def main():
    parser=argparse.ArgumentParser(description='V2·V3-A·V3-B의 동일 조건 평가')
    parser.add_argument('--v2',type=Path,required=True)
    parser.add_argument('--v3',type=Path,required=True)
    parser.add_argument('--v3b',type=Path)
    parser.add_argument('--checkpoints',type=int,nargs='+',default=[0,25,50,75,100])
    parser.add_argument('--legacy-checkpoints',type=int,nargs='+',default=[0,25,50,75,100])
    parser.add_argument('--orientation-reward-scale',type=float,default=.05)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compare(args.v2,args.v3,args.output,v3b=args.v3b,checkpoints=args.checkpoints,
        legacy_checkpoints=args.legacy_checkpoints,orientation_reward_scale=args.orientation_reward_scale)


if __name__=='__main__':
    main()
