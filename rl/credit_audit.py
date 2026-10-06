"""실제 rollout과 복원된 GPU 상태의 counterfactual return을 비교한다."""
import argparse
import copy
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg,ppo_config
from .env import StandingEnv
from .normalization import restore,pre_update_metrics
from .distributions import inverse_squash
from .evaluate import ScriptedController
from .diagnose_ppo import instrument_update

STATE_FIELDS=('time','qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied',
    'xfrc_applied','eq_active','mocap_pos','mocap_quat','qacc')
ENV_FIELDS=('previous_actions','requested_actions','joint_targets','episode_length_buf','push_start','push_vector')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def statistics(x):
    a=np.asarray(x,dtype=float)
    return dict(count=int(a.size),mean=float(a.mean()),std=float(a.std()),
        min=float(a.min()),max=float(a.max())) if a.size else None


def cosine(a,b):
    a=np.asarray(a);b=np.asarray(b);den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else None


def capture(env,index):
    with env._stream():
        physics={name:wp.to_torch(getattr(env.sim.wp_data,name))[index].clone().cpu() for name in STATE_FIELDS}
        buffers={name:getattr(env,name)[index].clone().cpu() for name in ENV_FIELDS}
    return dict(physics=physics,buffers=buffers,rng=env._rng.get_state().cpu(),
        observation=env.get_observations()['actor'][index].clone().cpu())


def restore_state(env,state):
    env._rng.set_state(state['rng'])
    with env._stream():
        for name,value in state['physics'].items():
            target=wp.to_torch(getattr(env.sim.wp_data,name))
            target.copy_(value.to(target.device).expand_as(target))
        for name,value in state['buffers'].items():
            target=getattr(env,name);target.copy_(value.to(target.device).expand_as(target))
        env.sim.forward()
    actual=env.get_observations()['actor']
    torch.testing.assert_close(actual,state['observation'].to(env.device).expand_as(actual),atol=2e-6,rtol=1e-5)
    return env.get_observations()


def bootstrapped_mc(reward,done,last_value,gamma=.99):
    result=torch.zeros_like(reward);running=last_value
    for i in range(len(reward)-1,-1,-1):
        running=reward[i]+gamma*(1-done[i].float())*running
        result[i]=running
    return result


def collect(checkpoint,output,fixed=False):
    profile=json.loads((checkpoint.parent/'environment.json').read_text())['config']
    env=StandingEnv(StandingCfg(**{**profile,'seed':42424,'num_envs':64,'balanced_push_directions':True}))
    cfg=ppo_config('squashed',terminal_bootstrap=fixed);cfg['num_steps_per_env']=32
    runner=OnPolicyRunner(env,cfg,device=env.device);infos=runner.load(str(checkpoint));restore(runner.alg,infos)
    alg=runner.alg;env._rng.manual_seed(42424);torch.manual_seed(42424);obs=env.reset()
    records=[];snapshots=[];saved_storage=None
    try:
        for block in range(16):
            rows=[]
            with torch.no_grad():
                for step in range(32):
                    time=env.episode_length_buf*env.cfg.step_dt
                    phase=torch.where(time<env.push_start,0,torch.where(time<env.push_start+.5,1,
                        torch.where(time<env.push_start+1.5,2,3)))
                    # 각 구간의 실제 rollout 표본 6개. action 선택 전에 전체 상태를 저장한다.
                    for category in range(4):
                        existing=sum(s['phase']==category for s in snapshots)
                        if existing<6:
                            indices=(phase==category).nonzero().flatten()
                            if len(indices) and (block*32+step)%16==0:
                                index=int(indices[(existing*7)%len(indices)])
                                snapshots.append(dict(label=f'rollout_{category}_{existing}',phase=category,
                                    flat_index=(block*32+step)*64+index,state=capture(env,index)))
                    action=alg.act(obs);mean,std=alg.actor.output_distribution_params
                    before=obs['actor'].clone();value=alg.transition.values.clone().squeeze(-1)
                    log_prob=alg.transition.actions_log_prob.clone()
                    obs,reward,done,extras=env.step(action)
                    terminal_value=alg.critic(extras['terminal_observation']).squeeze(-1)
                    alg.process_env_step(obs,reward,done,extras)
                    rows.append(dict(raw=before,action=action.clone(),mu=mean.clone(),sigma=std.clone(),
                        value=value,physical_reward=reward.clone(),done=done.clone(),timeout=extras['time_outs'].clone(),
                        failure=extras['diagnostics']['terminated'].clone(),terminal_value=terminal_value,
                        phase=phase.clone(),log_prob=log_prob.clone()))
                alg.compute_returns(obs);st=alg.storage
                values=st.values.squeeze(-1);stored=st.rewards.squeeze(-1);done=st.dones.squeeze(-1)
                last=alg.critic(obs).squeeze(-1)
                raw_reward=torch.stack([r['physical_reward'] for r in rows])
                timeout=torch.stack([r['timeout'] for r in rows])
                terminal=torch.stack([r['terminal_value'] for r in rows])
                corrected=raw_reward+.99*timeout*terminal
                next_value=torch.cat((values[1:],last[None]))
                delta=stored+.99*(1-done.float())*next_value-values
                raw_adv=(st.returns-st.values).squeeze(-1)
                normalized=st.advantages.squeeze(-1).clone()
                native_returns=st.returns.squeeze(-1).clone()
                mc=bootstrapped_mc(corrected,done,last)
                mc_truncated=bootstrapped_mc(raw_reward,done,torch.zeros_like(last))
                corrected_adv=torch.zeros_like(values);running=torch.zeros_like(last)
                for i in range(31,-1,-1):
                    td=corrected[i]+.99*(1-done[i].float())*next_value[i]-values[i]
                    running=td+.99*.95*(1-done[i].float())*running;corrected_adv[i]=running
                batch={key:torch.stack([r[key] for r in rows]).cpu().numpy() for key in rows[0]}
                batch.update(stored_reward=stored.cpu().numpy(),td_residual=delta.cpu().numpy(),
                    raw_advantage=raw_adv.cpu().numpy(),normalized_advantage=normalized.cpu().numpy(),
                    gae_return=native_returns.cpu().numpy(),mc_bootstrapped=mc.cpu().numpy(),
                    mc_truncated=mc_truncated.cpu().numpy(),corrected_advantage=corrected_adv.cpu().numpy(),
                    corrected_gae_return=(corrected_adv+values).cpu().numpy())
                records.append(batch)
                if block==4:saved_storage=copy.deepcopy(st)
                st.clear()
        data={key:np.concatenate([row[key] for row in records],axis=0) for key in records[0]}
        np.savez_compressed(output/'rollout.npz',**data)
        torch.save(snapshots,output/'rollout_snapshots.pt')
        raw=torch.tensor(data['raw'].reshape(-1,42),device=env.device)
        actions=torch.tensor(data['action'].reshape(-1,2),device=env.device)
        raw_adv=torch.tensor(data['raw_advantage'].reshape(-1),device=env.device)
        norm_adv=torch.tensor(data['normalized_advantage'].reshape(-1),device=env.device)
        def gradient(adv):
            mean=alg.actor.mlp(alg.actor.get_latent(TensorDict({'actor':raw},batch_size=[len(raw)])))
            alg.actor.distribution.update(mean)
            loss=-(alg.actor.get_output_log_prob(actions)*adv).mean()
            return torch.cat([g.flatten() for g in torch.autograd.grad(loss,alg.actor.parameters())]).cpu().numpy()
        graw=gradient(raw_adv);gnorm=gradient(norm_adv)
        score=(np.arctanh(np.clip(data['action'],-1+2**-24,1-2**-24))-data['mu'])/data['sigma']**2
        contribution=-data['normalized_advantage'][...,None]*score
        phases={}
        for category,name in enumerate(('pre','during','post','steady')):
            mask=data['phase']==category;v=data['value'][mask];mc=data['mc_bootstrapped'][mask]
            ret=data['gae_return'][mask];adv=data['raw_advantage'][mask]
            phases[name]=dict(samples=int(mask.sum()),value_bias=float(np.mean(v-mc)),value_rmse=float(np.sqrt(np.mean((v-mc)**2))),
                explained_variance=float(1-np.var(ret-v)/np.var(ret)),advantage=statistics(adv),
                mc_bootstrapped_explained_variance=float(1-np.var(mc-v)/np.var(mc)),
                mc_truncated_value_bias=float(np.mean(v-data['mc_truncated'][mask])),
                mc_truncated_value_rmse=float(np.sqrt(np.mean((v-data['mc_truncated'][mask])**2))),
                mc_truncated_explained_variance=float(1-np.var(data['mc_truncated'][mask]-v)/np.var(data['mc_truncated'][mask])),
                normalized_advantage=statistics(data['normalized_advantage'][mask]),td_residual=statistics(data['td_residual'][mask]),
                mean_head_gradient_sum=contribution[mask].sum(0).tolist(),mean_head_gradient_mean=contribution[mask].mean(0).tolist(),
                sum_per_sample_gradient_norm=float(np.linalg.norm(contribution[mask],axis=-1).sum()),
                full_actor_gradient_norm=float(np.linalg.norm(gradient(norm_adv*torch.tensor(mask.flatten(),device=env.device)))))
        selected=torch.stack([s['state']['observation'] for s in snapshots]).to(env.device)
        td=TensorDict({'actor':selected},batch_size=[len(selected)])
        with torch.no_grad():
            before=alg.actor.mlp(alg.actor.get_latent(td)).clone()
            alg.actor.distribution.update(before)
            sampled=torch.stack([torch.tensor(data['action'][divmod(s['flat_index'],64)]) for s in snapshots]).to(env.device)
            stored_log_prob=torch.tensor([float(data['log_prob'][divmod(s['flat_index'],64)]) for s in snapshots],device=env.device)
            ratios=torch.exp(alg.actor.get_output_log_prob(sampled)-stored_log_prob)
        alg.storage=saved_storage
        preflight=pre_update_metrics(alg);torch.manual_seed(62624)
        update=instrument_update(alg)
        with torch.no_grad():after=alg.actor.mlp(alg.actor.get_latent(td))
        for i,s in enumerate(snapshots):
            index=s['flat_index'];t,n=divmod(index,64)
            s.update(action=torch.tensor(data['action'][t,n]),raw_advantage=float(data['raw_advantage'][t,n]),
                normalized_advantage=float(data['normalized_advantage'][t,n]),log_prob=float(data['log_prob'][t,n]),probability_ratio=float(ratios[i]),
                mu_before=before[i].cpu(),mu_after=after[i].cpu(),delta_mu=(after[i]-before[i]).cpu())
        torch.save(snapshots,output/'rollout_snapshots.pt')
        summary=dict(checkpoint=str(checkpoint),terminal_bootstrap=fixed,steps=512,transitions=32768,
            shapes={key:list(value.shape) for key,value in data.items()},phases=phases,
            timeout_count=int(data['timeout'].sum()),failure_count=int(data['failure'].sum()),
            timeout_reward_error=statistics((data['stored_reward']-(data['physical_reward']+.99*data['timeout']*data['terminal_value']))[data['timeout']]),
            bootstrap_correction_advantage_max=float(np.abs(data['corrected_advantage']-data['raw_advantage']).max()),
            normalization_gradient_cosine=cosine(graw,gnorm),raw_gradient_norm=float(np.linalg.norm(graw)),
            normalized_gradient_norm=float(np.linalg.norm(gnorm)),pre_update=preflight,diagnostic_update=update,
            formula='delta=r_boot+gamma*(1-done)*Vnext-V; A=delta+gamma*lambda*(1-done)*Anext',
            mc_note='32-step truncated MC에 terminal 또는 rollout cutoff value를 bootstrap. 독립적인 무한 horizon 정답은 아님')
        save(output/'audit.json',summary)
        print('실제 rollout advantage 계측 완료:',output,flush=True)
    finally:env.close()


def representatives(cfg,output):
    env=StandingEnv(replace(cfg,num_envs=4,balanced_push_directions=True))
    states=[]
    try:
        env.reset();env.push_vector.zero_()
        for _ in range(500):
            env.episode_length_buf.zero_();env.step(torch.zeros((4,2),device=env.device))
        base=capture(env,0);base['buffers']['episode_length_buf'].zero_();base['physics']['time'].zero_()
        states.append(dict(label='equilibrium',state=base))
        for axis in range(2):
            for sign in (-1,1):
                state=copy.deepcopy(base);state['buffers']['push_start'].zero_()
                state['buffers']['push_vector'].zero_();state['buffers']['push_vector'][axis]=sign*.5
                restore_state(env,state)
                for _ in range(15):env.step(torch.zeros((4,2),device=env.device))
                snap=capture(env,0)
                g=snap['observation'][:3];angles=torch.stack((torch.atan2(-g[1],-g[2]),torch.asin(g[0])))
                error=angles-env.reference_standing_orientation[:2].cpu()
                orient_axis=1-axis;name=('roll' if orient_axis==0 else 'pitch')+('positive' if error[orient_axis]>0 else 'negative')
                states.append(dict(label=name,state=snap))
                if axis==0 and sign==1:
                    for _ in range(20):env.step(torch.zeros((4,2),device=env.device))
                    recovery=capture(env,0)
        states.append(dict(label='recovery',state=recovery))
        torch.save(states,output/'representatives.pt')
    finally:env.close()


def landscape(checkpoint,output):
    profile=json.loads((checkpoint.parent/'environment.json').read_text())['config'];cfg=StandingCfg(**profile)
    representatives(cfg,output)
    states=torch.load(output/'representatives.pt',weights_only=False)+torch.load(output/'rollout_snapshots.pt',weights_only=False)
    # 81 grid points + mean/sample/scripted/wrong/zero + central finite differences.
    env=StandingEnv(replace(cfg,num_envs=91,balanced_push_directions=True))
    runner=OnPolicyRunner(env,ppo_config('squashed'),device=env.device);infos=runner.load(str(checkpoint));restore(runner.alg,infos)
    teacher=ScriptedController(env);results=[];arrays=[]
    try:
        for row in states:
            obs=restore_state(env,row['state'])
            with torch.no_grad():
                mean=runner.alg.actor(obs)[0]
                runner.alg.actor(obs,stochastic_output=True);sample=runner.alg.actor.distribution.sample()[0]
                script=teacher(obs)[0]
                grid=torch.cartesian_prod(torch.linspace(-1,1,9,device=env.device),torch.linspace(-1,1,9,device=env.device))
                sample=row.get('action',sample.cpu()).to(env.device)
                fd=.05;local=torch.stack((mean+torch.tensor([fd,0],device=env.device),mean-torch.tensor([fd,0],device=env.device),
                    mean+torch.tensor([0,fd],device=env.device),mean-torch.tensor([0,fd],device=env.device))).clamp(-1,1)
                actions=torch.cat((grid,torch.stack((mean,sample,script,-script,torch.zeros_like(mean))),local,mean[None]),dim=0)
                assert actions.shape==(91,2)
                reward_sum=torch.zeros(91,device=env.device);peak=torch.zeros_like(reward_sum);integrated=torch.zeros_like(reward_sum)
                alive=torch.ones(91,device=env.device,dtype=torch.bool);horizons={}
                for step in range(50):
                    obs,reward,done,extras=env.step(actions if step==0 else torch.zeros_like(actions))
                    d=extras['diagnostics'];err=torch.stack((d['roll'],d['pitch']),-1)-env.reference_standing_orientation[:2]
                    tilt=err.rad2deg().norm(dim=-1)
                    reward_sum+=(.99**step)*reward*alive
                    peak=torch.maximum(peak,torch.where(alive,tilt,0));integrated+=tilt*alive*.02
                    alive&=~done.bool()
                    if step+1 in (1,5,10,25,50):
                        q=reward_sum.cpu().numpy();labels={'ppo_mean':81,'ppo_sample':82,'scripted':83,'wrong':84,'zero':85}
                        gradient=np.array([(q[86]-q[87])/float(local[0,0]-local[1,0]),(q[88]-q[89])/float(local[2,1]-local[3,1])])
                        # Δμ와 비교할 때 da/dμ를 포함한 latent Q gradient도 기록한다.
                        latent_gradient=gradient*(1-mean.cpu().numpy()**2)
                        ranks={name:1+int(np.sum(q[:81]>q[idx]+1e-9)) for name,idx in labels.items()}
                        values={name:float(q[idx]) for name,idx in labels.items()}
                        best=int(np.argmax(q[:81]))
                        horizons[str(step+1)]=dict(best_action=actions[best].cpu().tolist(),best_grid_return=float(q[best]),
                            rank=ranks,return_value=values,q_gradient_action=gradient.tolist(),q_gradient_latent=latent_gradient.tolist(),
                            gradient_alignment=cosine(latent_gradient,row['delta_mu'].numpy()) if 'delta_mu' in row else None,
                            physical={name:dict(peak=float(peak[idx]),integrated=float(integrated[idx]),
                                mean_orientation_error=float(integrated[idx])/((step+1)*.02)) for name,idx in labels.items()})
                        arrays.append(dict(label=row['label'],horizon=step+1,actions=actions.cpu().numpy(),q=q,peak=peak.cpu().numpy(),integrated=integrated.cpu().numpy()))
                row_summary=dict(label=row['label'],phase=row.get('phase'),action_mean=mean.cpu().tolist(),action_sample=sample.cpu().tolist(),scripted=script.cpu().tolist(),
                    horizons=horizons,continuation='first action once; then zero',survived_fraction=float(alive.float().mean()))
                for key in ('raw_advantage','normalized_advantage','log_prob','probability_ratio'):row_summary[key]=row.get(key)
                if 'delta_mu' in row:row_summary['actor_delta_mu']=row['delta_mu'].tolist()
                results.append(row_summary);save(output/'landscape.json',results)
                print('동일 상태 action grid 완료:',row['label'],flush=True)
        torch.save(arrays,output/'landscape_arrays.pt')
        comparison={}
        actual=[r for r in results if r['phase'] is not None]
        for horizon in (1,5,10,25,50):
            key=str(horizon);adv=np.array([r['normalized_advantage'] for r in actual]);rawadv=np.array([r['raw_advantage'] for r in actual])
            diff=np.array([r['horizons'][key]['return_value']['ppo_sample']-r['horizons'][key]['return_value']['ppo_mean'] for r in actual])
            noise=np.array([max(abs(a['q'][81]-a['q'][90]),abs(a['q'][40]-a['q'][85]),1e-7)
                for a in arrays if a['horizon']==horizon and a['label'].startswith('rollout_')])
            meaningful=np.abs(diff)>5*noise
            align=[r['horizons'][key]['gradient_alignment'] for r in actual]
            comparison[key]=dict(samples=len(actual),meaningful_samples=int(meaningful.sum()),
                normalized_advantage_q_correlation=float(np.corrcoef(adv,diff)[0,1]),
                raw_advantage_q_correlation=float(np.corrcoef(rawadv,diff)[0,1]),
                sign_accuracy=float(((adv[meaningful]>0)==(diff[meaningful]>0)).mean()),
                raw_sign_accuracy=float(((rawadv[meaningful]>0)==(diff[meaningful]>0)).mean()),
                mean_gradient_alignment=float(np.mean([x for x in align if x is not None])),
                positive_alignment_fraction=float(np.mean([x>0 for x in align if x is not None])),
                noise_threshold='5 × max(replicate return difference,1e-7)',
                interpretation='sample Q−mean-action Q, zero continuation. 실제 stochastic continuation의 advantage 정답으로 동일시하지 않음')
        save(output/'q_comparison.json',comparison)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description='Standing advantage와 물리 action 가치 진단')
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--part',choices=('collect','landscape','all'),default='all')
    parser.add_argument('--terminal-bootstrap',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.part in ('collect','all'):collect(args.checkpoint,args.output,args.terminal_bootstrap)
    if args.part in ('landscape','all'):landscape(args.checkpoint,args.output)


if __name__=='__main__':main()
