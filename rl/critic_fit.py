"""정책과 보상을 고정하고 critic의 MC 예측과 warm-start를 분리한다."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel
from .config import ppo_config
from .env import StandingEnv
from .policy_credit import profile,runner_for,restore_batch,value_metrics
from .normalization import configure_model,checkpoint_mode
from .reward_alignment import save


def collect_target(source,states_path,output,replicates=32,horizon=1024):
    states=torch.load(states_path,weights_only=False);n=len(states)
    env=StandingEnv(replace(profile(source),num_envs=n*replicates,episode_seconds=40.))
    runner=runner_for(env,source);alg=runner.alg
    try:
        obs=restore_batch(env,states,np.repeat(np.arange(n),replicates))
        with torch.no_grad():critic_value=alg.critic(obs).reshape(n,replicates)[:,0].cpu()
        generators=[torch.Generator(device=env.device).manual_seed(105124+i) for i in range(replicates)]
        total=torch.zeros(n*replicates,device=env.device,dtype=torch.float64)
        alive=torch.ones(n*replicates,device=env.device,dtype=torch.bool);failures=0
        with torch.no_grad():
            for step in range(horizon):
                eps=torch.stack([torch.randn(n,2,device=env.device,generator=g) for g in generators],dim=1).reshape(-1,2)
                mu=alg.actor.mlp(alg.actor.get_latent(obs));alg.actor.distribution.update(mu)
                from .distributions import squash
                command=squash(mu+alg.actor.output_std*eps)
                obs,reward,done,extras=env.step(command);bad=extras['diagnostics']['terminated']
                total+=(.99**step)*reward.double()*alive
                failures+=int((alive&bad).sum());alive&=~bad
                if step%128==0:print('critic 독립 MC target 수집:',step,'/',horizon,flush=True)
        returns=total.reshape(n,replicates).cpu()
        raw=torch.stack([s['state']['observation'] for s in states])
        # 동일 환경의 시간 표본이 train/validation에 섞이지 않도록 환경 ID로 분리한다.
        valid=torch.tensor([s['source_env'] in (1,4,6,7) for s in states])
        packet=dict(observation=raw,return_replicates=returns,target=returns.mean(-1).float(),validation_mask=valid,
            critic_value=critic_value,
            phases=[s['phase'] for s in states],source_env=[s['source_env'] for s in states],
            replicates=replicates,horizon=horizon,failures=failures,pose_reward_weight=env.cfg.pose_reward_weight,
            source=str(source),continuation='고정 actor의 stochastic policy; 인위적 timeout 이후도 같은 physical state에서 계속 계산')
        output.mkdir(parents=True,exist_ok=True);torch.save(packet,output/'dataset.pt')
        save(output/'dataset.json',{k:v for k,v in packet.items() if not isinstance(v,torch.Tensor)})
        return packet
    finally:env.close()


def fit(source,data,output,hidden=(64,64),steps=3000,center_head=False):
    state=torch.load(source,weights_only=False,map_location='cpu');cfg=ppo_config('squashed')['critic']
    raw=data['observation'].cuda();td=TensorDict({'critic':raw},batch_size=[len(raw)])
    torch.manual_seed(42424)
    model=MLPModel(td,{'critic':['critic']},'critic',1,hidden_dims=hidden,
        activation=cfg['activation'],obs_normalization=True).cuda()
    if hidden==(64,64):model.load_state_dict(state['critic_state_dict'])
    else:
        for name,value in model.obs_normalizer.state_dict().items():value.copy_(state['critic_state_dict']['obs_normalizer.'+name].cuda())
    configure_model(model,checkpoint_mode(state['infos']));model.obs_normalizer.until=int(model.obs_normalizer.count)
    model.eval();target=data['target'].cuda();valid=data['validation_mask'].cuda();train=~valid
    optimizer=torch.optim.Adam(model.mlp.parameters(),lr=.001)
    norm=copy.deepcopy(model.obs_normalizer.state_dict());history=[];best=None;best_error=float('inf')
    with torch.no_grad():initial=model(td).squeeze(-1).cpu().numpy()
    if center_head:
        # 큰 상수 baseline과 작은 상태별 residual을 분리해 초기 offset의 영향을 확인한다.
        with torch.no_grad():
            model.mlp[-1].weight.zero_();model.mlp[-1].bias.fill_(float(target[train].mean()))
            pred=model(td).squeeze(-1)
            best_error=float((pred[valid]-target[valid]).square().mean().sqrt())
            best=copy.deepcopy(model.state_dict())
        history.append(dict(step=-1,train_rmse=float((pred[train]-target[train]).square().mean().sqrt()),validation_rmse=best_error))
    for i in range(steps):
        pred=model(td).squeeze(-1);loss=(pred[train]-target[train]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if i%50==0 or i==steps-1:
            with torch.no_grad():pred=model(td).squeeze(-1);error=float((pred[valid]-target[valid]).square().mean().sqrt())
            history.append(dict(step=i,train_rmse=float((pred[train]-target[train]).square().mean().sqrt()),validation_rmse=error))
            if error<best_error:best_error=error;best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    assert all(torch.equal(model.obs_normalizer.state_dict()[k],v) for k,v in norm.items())
    with torch.no_grad():pred=model(td).squeeze(-1).cpu().numpy()
    y=data['target'].numpy();mask=data['validation_mask'].numpy();se=data['return_replicates'].numpy().std(-1,ddof=1)/np.sqrt(data['replicates'])
    metrics=dict(hidden_dims=list(hidden),steps=steps,centered_head_initialization=center_head,
        train_states=int((~mask).sum()),validation_states=int(mask.sum()),
        before_validation=value_metrics(initial[mask],y[mask]),train=value_metrics(pred[~mask],y[~mask]),validation=value_metrics(pred[mask],y[mask]),
        mc_target_standard_error_mean=float(se[mask].mean()),history=history,normalization_unchanged=True,
        actor_unchanged=True,pose_reward_weight=data['pose_reward_weight'])
    name='x'.join(map(str,hidden))+('_centered' if center_head else '');save(output/f'fit_{name}.json',metrics)
    torch.save(model.state_dict(),output/f'critic_{name}.pt')
    print('critic offline fit 완료:',name,'검증 RMSE',metrics['validation']['rmse'],'EV',metrics['validation']['explained_variance'],flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description='고정 actor에서 critic capacity·warm-start 진단')
    parser.add_argument('--source',type=Path,default=Path('/tmp/microdog_v7/v7a/ppo/checkpoint_0.pt'))
    parser.add_argument('--states',type=Path,default=Path('/tmp/microdog_v7/credit/states.pt'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/microdog_v7/critic_fit'))
    args=parser.parse_args();data=collect_target(args.source,args.states,args.output)
    current=fit(args.source,data,args.output)
    # 큰 bias를 현재 구조로 줄이고 holdout EV가 양수면 capacity를 키울 근거가 없다.
    candidate=None
    if current['validation']['explained_variance'] is None or current['validation']['explained_variance']<0:
        candidate=fit(args.source,data,args.output,hidden=(128,128))
    centered=None;selected=current;critic_path=args.output/'critic_64x64.pt'
    if current['validation']['rmse']>=.05:
        centered=fit(args.source,data,args.output,center_head=True)
        selected=centered;critic_path=args.output/'critic_64x64_centered.pt'
    save(args.output/'decision.json',dict(current=current['validation'],candidate=None if candidate is None else candidate['validation'],
        centered=None if centered is None else centered['validation'],
        selected_change='critic warm-start' if selected['validation']['rmse']<.05 else None,
        architecture_preserved=True,source=str(args.source),critic_path=str(critic_path),
        reason='현재64×64가 큰 value bias를 fit할 수 있으면 capacity 변경 대신 초기 critic 예측만 교정한다.'))


if __name__=='__main__':main()
