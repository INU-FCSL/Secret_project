"""공통 물리 상태에서 같은 확률정책의 MC·GAE·실제 갱신을 비교한다."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg,checkpoint_ppo_config
from .env import StandingEnv
from .credit_audit import STATE_FIELDS,ENV_FIELDS,capture,cosine
from .evaluate import ScriptedController
from .normalization import restore,snapshot,assert_unchanged,pre_update_metrics
from .diagnose_ppo import instrument_update
from .distributions import squash
from .reward_alignment import save,correlation,TERMS

PHASES=('pre','during','early_recovery','late_recovery','steady')


def profile(checkpoint):
    return StandingCfg(**json.loads((checkpoint.parent/'environment.json').read_text())['config'])


def runner_for(env,checkpoint):
    state=torch.load(checkpoint,weights_only=False,map_location='cpu')
    cfg=checkpoint_ppo_config(state['infos']);cfg['num_steps_per_env']=32
    runner=OnPolicyRunner(env,cfg,device=env.device);infos=runner.load(str(checkpoint));restore(runner.alg,infos)
    runner.alg.actor.eval();runner.alg.critic.eval();return runner


def restore_batch(env,states,indices):
    selected=[states[int(i)]['state'] for i in indices]
    env._rng.set_state(selected[0]['rng'])
    with env._stream():
        for name in STATE_FIELDS:
            target=wp.to_torch(getattr(env.sim.wp_data,name))
            target.copy_(torch.stack([s['physics'][name] for s in selected]).to(target.device))
        for name in ENV_FIELDS:
            target=getattr(env,name)
            target.copy_(torch.stack([s['buffers'][name] for s in selected]).to(target.device))
        env.sim.forward()
    obs=env.get_observations()
    expected=torch.stack([s['observation'] for s in selected]).to(env.device)
    torch.testing.assert_close(obs['actor'],expected,atol=2e-6,rtol=1e-5)
    return obs


def collect_states(checkpoint,output):
    env=StandingEnv(replace(profile(checkpoint),num_envs=64,seed=77331,balanced_push_directions=True))
    runner=runner_for(env,checkpoint);states=[];collected=set()
    try:
        env._rng.manual_seed(77331);torch.manual_seed(77331);obs=env.reset()
        for step in range(400):
            time=env.episode_length_buf*.02
            for p in range(5):
                for i in range(10):
                    if (p,i) in collected:continue
                    fraction=i/9
                    offset=(.6+fraction,.05+.4*fraction,.55+.4*fraction,1.05+.9*fraction,2.2+2*fraction)[p]
                    target=offset if p==0 else float(env.push_start[i])+offset
                    if float(time[i])>=target:
                        states.append(dict(label=f'{PHASES[p]}_{i}',phase=PHASES[p],source_env=i,
                            source_step=step,state=capture(env,i)))
                        collected.add((p,i))
            if len(states)==50:break
            with torch.no_grad():obs,_,_,_=env.step(runner.alg.actor(obs,stochastic_output=True))
        if len(states)!=50:raise RuntimeError('공통 물리 상태 50개 수집을 완료하지 못했습니다.')
        states.sort(key=lambda s:(PHASES.index(s['phase']),s['source_env']))
        torch.save(states,output/'states.pt')
        print('공통 상태 수집 완료: 구간별 10개, 총 50개',flush=True)
    finally:env.close()


def diagnostic_update(checkpoint,states,output):
    env=StandingEnv(replace(profile(checkpoint),num_envs=64))
    runner=runner_for(env,checkpoint);alg=runner.alg
    try:
        obs=restore_batch(env,states,list(range(50))+list(range(14)))
        expected=snapshot(alg);torch.manual_seed(80922);physical=[]
        with torch.no_grad():
            initial_mu=alg.actor.mlp(alg.actor.get_latent(obs))[:50].clone()
            for _ in range(32):
                action=alg.act(obs);obs,reward,done,extras=env.step(action)
                physical.append(reward.cpu().numpy());alg.process_env_step(obs,reward,done,extras)
            alg.compute_returns(obs)
        st=alg.storage
        sample=st.actions[0,:50].cpu().numpy();value=st.values[0,:50,0].cpu().numpy()
        raw=(st.returns-st.values)[0,:50,0].cpu().numpy();normalized=st.advantages[0,:50,0].cpu().numpy()
        np.savez_compressed(output/'gae_rollout.npz',actions=st.actions.cpu().numpy(),values=st.values.cpu().numpy(),
            raw_advantage=(st.returns-st.values).cpu().numpy(),normalized_advantage=st.advantages.cpu().numpy(),
            physical_reward=np.stack(physical),stored_reward=st.rewards.cpu().numpy(),dones=st.dones.cpu().numpy())
        before=pre_update_metrics(alg);update=instrument_update(alg)
        with torch.no_grad():
            selected=TensorDict({'actor':torch.stack([s['state']['observation'] for s in states]).to(env.device)},batch_size=[50])
            after=alg.actor.mlp(alg.actor.get_latent(selected));delta=(after-initial_mu).cpu().numpy()
        assert_unchanged(alg,expected)
        save(output/'update.json',dict(checkpoint=str(checkpoint),pre_update=before,diagnostic=update,
            initialization='동일 snapshot 50개와 반복 상태14개에서 실제64×32 native PPO를 독립 복사본으로 1회 갱신',
            source_checkpoint_unchanged=True))
        np.savez_compressed(output/'state_gae.npz',sample=sample,value=value,raw_advantage=raw,
            normalized_advantage=normalized,delta_mu=delta,mu=initial_mu.cpu().numpy())
        return dict(sample=sample,value=value,raw_advantage=raw,normalized_advantage=normalized,
            delta_mu=delta,mu=initial_mu.cpu().numpy())
    finally:env.close()


def value_metrics(value,target):
    error=value-target
    return dict(rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),
        explained_variance=float(1-np.var(error)/np.var(target)) if np.var(target)>1e-12 else None,
        spearman=correlation(value,target,True))


def compare(states,gae,arrays,output):
    comparisons={}
    for horizon in ('32','64','remainder','extended'):
        q=arrays[horizon];v=q[:,8].mean(-1);adv=(q[:,7]-q[:,8]).mean(-1)
        sem=(q[:,7]-q[:,8]).std(-1,ddof=1)/np.sqrt(q.shape[-1]);resolved=np.abs(adv)>2*sem
        gradient=np.stack(((q[:,3]-q[:,4]).mean(-1)/arrays['fd_pitch'],
            (q[:,5]-q[:,6]).mean(-1)/arrays['fd_roll']),-1)
        local=gradient*(1-np.tanh(gae['mu'])**2) if 'mu' in gae else gradient
        alignment=[cosine(g,d) for g,d in zip(local,gae['delta_mu'])]
        advantages={}
        for name in ('raw_advantage','normalized_advantage'):
            a=gae[name]
            advantages[name]=dict(sign_accuracy=float(np.mean((a>0)==(adv>0))),
                resolved_sign_accuracy=float(np.mean((a[resolved]>0)==(adv[resolved]>0))) if resolved.any() else None,
                pearson=correlation(a,adv),spearman=correlation(a,adv,True))
        phases={p:value_metrics(gae['value'][np.array([s['phase']==p for s in states])],
            v[np.array([s['phase']==p for s in states])]) for p in PHASES}
        comparisons[horizon]=dict(critic=value_metrics(gae['value'],v),phases=phases,advantage=advantages,
            resolved_advantage_samples=int(resolved.sum()),gradient_alignment=float(np.mean([x for x in alignment if x is not None])),
            gradient_positive_fraction=float(np.mean([x>0 for x in alignment if x is not None])),
            gradient_sign_agreement=float(np.mean((local>0)==(gae['delta_mu']>0))),
            mean_q_gradient=gradient.mean(0).tolist(),mean_delta_mu=gae['delta_mu'].mean(0).tolist())
    save(output/'comparison.json',comparisons)
    return comparisons


def monte_carlo(checkpoint,states,gae,output,replicates=32,chunk_size=5,tail_steps=512):
    # 8개 고정 첫 명령 + 확률정책의 첫 명령. continuation noise는 후보 간 공유한다.
    branches=9;worlds=chunk_size*branches*replicates
    cfg=replace(profile(checkpoint),num_envs=worlds,episode_seconds=40.)
    env=StandingEnv(cfg);runner=runner_for(env,checkpoint);alg=runner.alg;teacher=ScriptedController(env)
    arrays={k:[] for k in ('32','64','remainder','extended','remainder_raw','actions','fd_pitch','fd_roll','failures')}
    try:
        for offset in range(0,len(states),chunk_size):
            count=min(chunk_size,len(states)-offset)
            if count!=chunk_size:raise ValueError('state 수는 chunk 크기로 나눌 수 있어야 합니다.')
            ids=np.repeat(np.arange(offset,offset+count),branches*replicates)
            obs=restore_batch(env,states,ids)
            remainder=torch.tensor([500-int(s['state']['buffers']['episode_length_buf']) for s in states[offset:offset+count]],device=env.device)
            per_world=remainder[:,None,None].expand(count,branches,replicates).reshape(-1)
            generators=[torch.Generator(device=env.device).manual_seed(91234+1000*offset+i) for i in range(replicates)]
            def noise():
                eps=torch.stack([torch.randn(count,2,device=env.device,generator=g) for g in generators],dim=1)
                return eps[:,None].expand(count,branches,replicates,2).reshape(-1,2)
            with torch.no_grad():
                mu=alg.actor.mlp(alg.actor.get_latent(obs));alg.actor.distribution.update(mu)
                mean=squash(mu).reshape(count,branches,replicates,2)[:,0,0]
                scripted=teacher(obs).reshape(count,branches,replicates,2)[:,0,0]
                delta=mean.new_tensor([[.1,0],[-.1,0],[0,.1],[0,-.1]])
                local=(mean[:,None]+delta[None]).clamp(-1,1)
                sample=torch.tensor(gae['sample'][offset:offset+count],device=env.device)
                fixed=torch.cat((mean[:,None],scripted[:,None],torch.zeros_like(mean[:,None]),local,sample[:,None],mean[:,None]),dim=1)
                actions=fixed[:,:,None].expand(count,branches,replicates,2).clone()
                random_first=squash(mu+alg.actor.output_std*noise()).reshape(count,branches,replicates,2)
                actions[:,8]=random_first[:,8]
                initial_actions=actions.clone();command=actions.reshape(-1,2)
                total=torch.zeros(worlds,device=env.device,dtype=torch.float64);alive=torch.ones(worlds,device=env.device,dtype=torch.bool)
                prefix={};episode=torch.zeros_like(total);episode_raw=torch.zeros_like(total);failure_count=torch.zeros(worlds,device=env.device,dtype=torch.int32)
                for step in range(int(remainder.max())+tail_steps):
                    if step:
                        mu=alg.actor.mlp(alg.actor.get_latent(obs));alg.actor.distribution.update(mu)
                        command=squash(mu+alg.actor.output_std*noise())
                    obs,reward,done,extras=env.step(command)
                    failure=extras['diagnostics']['terminated']
                    total+=(.99**step)*reward.double()*alive
                    failure_count+=failure.to(torch.int32)*alive
                    alive&=~failure
                    boundary=per_world==step+1
                    if boundary.any():
                        terminal_value=alg.critic(extras['terminal_observation']).squeeze(-1)
                        episode_raw[boundary]=total[boundary]
                        episode[boundary]=(total+.99**(step+1)*terminal_value*alive)[boundary]
                    if step+1 in (32,64):
                        terminal_value=alg.critic(extras['terminal_observation']).squeeze(-1)
                        prefix[str(step+1)]=(total+.99**(step+1)*terminal_value*alive).clone()
                    if step%128==0:print('동일 정책 MC 진행:',checkpoint.name,'상태',offset+1,'~',offset+count,'step',step,flush=True)
                for k in ('32','64'):arrays[k].append(prefix[k].reshape(count,branches,replicates).cpu().numpy())
                arrays['remainder'].append(episode.reshape(count,branches,replicates).cpu().numpy())
                arrays['remainder_raw'].append(episode_raw.reshape(count,branches,replicates).cpu().numpy())
                arrays['extended'].append(total.reshape(count,branches,replicates).cpu().numpy())
                arrays['actions'].append(initial_actions.cpu().numpy())
                arrays['fd_pitch'].append((local[:,0,0]-local[:,1,0]).cpu().numpy())
                arrays['fd_roll'].append((local[:,2,1]-local[:,3,1]).cpu().numpy())
                arrays['failures'].append(failure_count.reshape(count,branches,replicates).cpu().numpy())
            print('동일 정책 MC 묶음 완료:',offset+count,'/',len(states),flush=True)
        result={k:np.concatenate(v) for k,v in arrays.items()}
        np.savez_compressed(output/'mc.npz',**result)
        save(output/'protocol.json',dict(checkpoint=str(checkpoint),states=len(states),branches=branches,replicates=replicates,
            candidates=['mean','scripted','zero','pitch_plus','pitch_minus','roll_plus','roll_minus','actual_sample','policy_expectation'],
            horizons=[32,64,'episode remainder'],terminal_tail_steps=tail_steps,
            continuation='해당 checkpoint의 stochastic policy; 후보 간 common random numbers; replicate seed는 독립',
            timeout='32/64 및 원래10초 경계는 terminal critic bootstrap. extended는 시간제한만 늘려 terminal 이후 MC tail을 직접 계산',
            extended_return_note='추가512 step 뒤 남은 할인 tail은 0으로 절단한다. 원래 정책·외란·물리를 유지하며 새로운 episode로 reset하지 않는다.'))
        return compare(states,gae,result,output)
    finally:env.close()


def local_components(checkpoint,output):
    states=torch.load('/tmp/microdog_v6/after/representatives.pt',weights_only=False)
    env=StandingEnv(replace(profile(checkpoint),num_envs=30));runner=runner_for(env,checkpoint);results=[]
    try:
        obs=restore_batch(env,states,np.repeat(np.arange(6),5))
        with torch.no_grad():
            mean=runner.alg.actor(obs).reshape(6,5,2)[:,0]
            delta=mean.new_tensor([[0,0],[.1,0],[-.1,0],[0,.1],[0,-.1]])
            first=(mean[:,None]+delta[None]).clamp(-1,1)
            total=torch.zeros(30,6,device=env.device,dtype=torch.float64)
            for step in range(50):
                command=first.reshape(30,2) if step==0 else runner.alg.actor(obs)
                obs,_,_,extra=env.step(command)
                total+=(.99**step)*torch.stack([extra['diagnostics']['reward_terms'][k] for k in TERMS],-1).double()*.02
                if step+1 in (1,5,10,25,50):
                    value=total.reshape(6,5,6).cpu().numpy()
                    for i,state in enumerate(states):
                        g=np.stack(((value[i,1]-value[i,2])/float(first[i,1,0]-first[i,2,0]),
                            (value[i,3]-value[i,4])/float(first[i,3,1]-first[i,4,1])),axis=1)
                        results.append(dict(state=state['label'],horizon=step+1,
                            gradient={k:g[j].tolist() for j,k in enumerate(TERMS)},
                            orientation_pose_cosine=cosine(g[0],g[2]),continuation='imitation0 deterministic policy'))
        save(output/'local_component_gradient.json',results)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description='같은 정책 continuation의 critic·advantage 검증')
    parser.add_argument('--output',type=Path,default=Path('/tmp/microdog_v7/credit'))
    parser.add_argument('--part',choices=('states','local','audit','all'),default='all')
    parser.add_argument('--replicates',type=int,default=32);parser.add_argument('--chunk-size',type=int,default=5)
    parser.add_argument('--tail-steps',type=int,default=512);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    directory=Path('/tmp/microdog_v6/imitation/ppo')
    if args.part in ('states','all'):collect_states(directory/'checkpoint_0.pt',args.output)
    if args.part in ('local','all'):local_components(directory/'checkpoint_0.pt',args.output)
    if args.part in ('audit','all'):
        states=torch.load(args.output/'states.pt',weights_only=False)
        for i in (0,1,5):
            output=args.output/f'policy_{i}';output.mkdir(exist_ok=True)
            gae=diagnostic_update(directory/f'checkpoint_{i}.pt',states,output)
            monte_carlo(directory/f'checkpoint_{i}.pt',states,gae,output,args.replicates,args.chunk_size,args.tail_steps)


if __name__=='__main__':main()
