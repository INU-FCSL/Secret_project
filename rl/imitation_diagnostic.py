"""scripted actor를 재현하고 PPO가 좋은 초기 정책을 유지하는지 진단한다."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg,ppo_config
from .env import StandingEnv
from .evaluate import ScriptedController
from .normalization import restore,assert_unchanged,snapshot,pre_update_metrics
from .diagnose_ppo import instrument_update
from .ablation import measure,save


def run(source,output,resume_pretraining=False):
    profile=json.loads((source.parent/'environment.json').read_text())['config']
    source_state=torch.load(source,weights_only=False,map_location='cpu')
    infos=copy.deepcopy(source_state['infos']);infos.update(terminal_bootstrap=True,mean_regularization=0.,completed_updates=0)
    output.mkdir(parents=True,exist_ok=True);cfg=StandingCfg(**profile)
    rows=[];targets=[];phases=[];reset_counts=[]
    if resume_pretraining:
        dataset=torch.load(output/'dataset.pt',weights_only=False)
        x,y,phase=dataset['observation'],dataset['action'],dataset['phase']
    else:
        env=StandingEnv(cfg);teacher=ScriptedController(env)
        try:
            for seed in (61026,61027,61028,61029):
                env._rng.manual_seed(seed);obs=env.reset()
                resets=0
                for step in range(500):
                    with torch.no_grad():
                        target=teacher(obs);time=env.episode_length_buf*env.cfg.step_dt
                        phase=torch.where(time<.5,0,torch.where(time<env.push_start,1,
                            torch.where(time<env.push_start+.5,2,torch.where(time<env.push_start+1.5,3,4))))
                        rows.append(obs['actor'].cpu());targets.append(target.cpu());phases.append(phase.cpu())
                        obs,_,done,_=env.step(target)
                        if step<499:resets+=int(done.sum())
                reset_counts.append(resets)
                print('scripted 학습 자료 수집:',seed,flush=True)
        finally:env.close()
        x=torch.stack(rows).reshape(4,500,64,42);y=torch.stack(targets).reshape(4,500,64,2)
        phase=torch.stack(phases).reshape(4,500,64)
        torch.save(dict(observation=x,action=y,phase=phase,seeds=[61026,61027,61028,61029],
            post_initial_reset_count=reset_counts),output/'dataset.pt')
    if resume_pretraining:
        reset_counts=dataset.get('post_initial_reset_count',
            (x[:,1:,:,30:]==0).all(-1).sum((1,2)).tolist())
    env=StandingEnv(cfg)
    try:
        torch.manual_seed(42424)
        config=ppo_config('squashed',terminal_bootstrap=True);config['num_steps_per_env']=32
        runner=OnPolicyRunner(env,config,device=env.device)
        # actor/critic은 독립적인 정상 초기 network이며 정규화 통계만 동일하게 복원한다.
        for side in ('actor','critic'):
            model=getattr(runner.alg,side)
            model.load_state_dict(source_state[side+'_state_dict'])
        restore(runner.alg,infos);expected=snapshot(runner.alg)
        actor=runner.alg.actor
        std_before=actor.distribution.std_param.detach().clone()
        if resume_pretraining:
            trained=torch.load(output/'ppo'/'checkpoint_0.pt',weights_only=False,map_location=env.device)
            actor.load_state_dict(trained['actor_state_dict'])
        train_x=x[:3].reshape(-1,42).cuda();train_y=y[:3].reshape(-1,2).cuda()
        valid_x=x[3].reshape(-1,42).cuda();valid_y=y[3].reshape(-1,2).cuda()
        train_phase=phase[:3].flatten().cuda()
        optimizer=torch.optim.Adam(actor.mlp.parameters(),lr=.001)
        torch.manual_seed(64124)
        groups=[(train_phase==i).nonzero().flatten() for i in range(5)]
        losses=[]
        for step in range(0 if resume_pretraining else 4000):
            indices=torch.cat([g[torch.randint(len(g),(128,),device='cuda')] for g in groups if len(g)])
            predicted=actor(TensorDict({'actor':train_x[indices]},batch_size=[len(indices)]))
            loss=(predicted-train_y[indices]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            if step%500==0:losses.append(dict(step=step,mse=float(loss.detach())))
        actor.eval()
        with torch.no_grad():
            pred=actor(TensorDict({'actor':valid_x},batch_size=[len(valid_x)]))
            mse=float((pred-valid_y).square().mean())
            per_phase={str(i):float((pred[phase[3].flatten().cuda()==i]-valid_y[phase[3].flatten().cuda()==i]).square().mean()) for i in range(5)}
        assert torch.equal(std_before,actor.distribution.std_param)
        assert_unchanged(runner.alg,expected)
        # PPO optimizer는 supervised optimizer와 별도로 초기 상태를 유지한다.
        assert len(runner.alg.optimizer.state)==0
        ppo_dir=output/'ppo';ppo_dir.mkdir(exist_ok=True)
        env_snapshot=json.loads((source.parent/'environment.json').read_text())
        save(ppo_dir/'environment.json',env_snapshot);save(ppo_dir/'config.json',config)
        save(ppo_dir/'normalization.json',infos['normalization'])
        # learn() 이전에는 logger writer가 생성되지 않으므로 명시적으로 초기화한다.
        runner.logger.writer=None
        native_save=runner.save
        def save_checkpoint(path,metadata=None):native_save(str(path),infos=dict(infos,**(metadata or {})))
        save_checkpoint(ppo_dir/'checkpoint_0.pt')
        save(output/'pretraining.json',dict(train_initial_trajectories=192,validation_initial_trajectories=64,
            train_episode_segments=192+sum(reset_counts[:3]),validation_episode_segments=64+reset_counts[3],
            post_initial_reset_count=reset_counts,
            training_observations=96000,validation_observations=32000,seeds=[61026,61027,61028,61029],
            validation_mse=mse,validation_mse_by_phase=per_phase,losses=losses,updates=4000,
            loss='mean((tanh(mu)-scripted_action)^2)',std_fixed=.3,critic_fresh=True,normalization_preserved=True))
        def evaluate_checkpoint(iteration):
            target=output/'evaluation'/f'checkpoint_{iteration}';target.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m','rl.imitation_diagnostic','--evaluate-checkpoint',str(ppo_dir/f'checkpoint_{iteration}.pt'),
                '--eval-output',str(target)]
            subprocess.run(command,check=True)
            return json.loads((target/'summary.json').read_text())
        start=evaluate_checkpoint(0)
        # zero의 물리 오차와 모든 seed를 비교한다. unsafe imitation으로는 fine-tuning하지 않는다.
        baseline=json.loads(Path('/tmp/microdog_v6/fix/evaluation/summary.json').read_text())['baselines']['zero']
        def success(summary):
            return all(row['peak_orientation_error']<zero['peak_orientation_error'] and
                row['integrated_orientation_error']<zero['integrated_orientation_error'] and
                row['self_contact_env_steps']==0 and row['unexpected_contact_env_steps']==0 and row['saturation_fraction']<.01
                for row,zero in zip(summary['per_seed'],baseline['per_seed']))
        if not success(start):
            save(output/'decision.json',dict(imitation_success=False,ppo_finetuning_executed=False,
                state_dependent_exploration_allowed=False,reason='imitation 초기 정책이 zero보다 좋지 않아 좋은 정책 유지 검사를 진행하지 않음'))
            print('imitation 초기 정책 성공 기준 미달',flush=True);return
        # 별도 초기 critic·PPO optimizer, 같은 정규화·환경·σ로 5번만 갱신한다.
        env._rng.manual_seed(42424);torch.manual_seed(42424);env.reset()
        runner.save=save_checkpoint
        original_update=runner.alg.update;metrics=[]
        def checked_update():
            before=pre_update_metrics(runner.alg)
            if before['exact_kl']>1e-7 or before['clip_fraction']>1e-7:raise RuntimeError('imitation PPO 갱신 전 분포 불일치')
            result=instrument_update(runner.alg,update_fn=original_update)
            assert_unchanged(runner.alg,expected)
            metrics.append(dict(update=len(metrics)+1,pre_update=before,diagnostic=result))
            save(output/'updates.json',metrics)
            return result['loss']
        runner.alg.update=checked_update
        evaluations=[start]
        for i in range(1,6):
            runner.current_learning_iteration=i-1
            runner.learn(1,init_at_random_ep_len=False)
            save_checkpoint(ppo_dir/f'checkpoint_{i}.pt',dict(completed_updates=i))
            evaluations.append(evaluate_checkpoint(i))
        maintained=success(evaluations[-1])
        save(output/'decision.json',dict(imitation_success=True,ppo_finetuning_executed=True,updates=5,
            final_zero_improved=maintained,state_dependent_exploration_allowed=maintained,
            per_update_zero_improved=[success(row) for row in evaluations],
            peak=[row['aggregate']['peak_orientation_error'] for row in evaluations],
            integrated=[row['aggregate']['integrated_orientation_error'] for row in evaluations],
            reason='좋은 정책 유지: exploration/discovery 후보' if maintained else '좋은 초기 정책의 PPO 성능 붕괴: credit/update 우선'))
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description='scripted imitation과 PPO 유지 진단')
    parser.add_argument('--source',type=Path,default=Path('/tmp/microdog_v5/v5a/ppo/checkpoint_0.pt'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/microdog_v6/imitation'))
    parser.add_argument('--evaluate-checkpoint',type=Path);parser.add_argument('--eval-output',type=Path)
    parser.add_argument('--resume-pretraining',action='store_true')
    args=parser.parse_args()
    if args.evaluate_checkpoint:
        profile=json.loads((args.evaluate_checkpoint.parent/'environment.json').read_text())['config']
        summary=measure(StandingCfg(**profile),args.eval_output,'imitation',args.evaluate_checkpoint)
        save(args.eval_output/'summary.json',summary)
    else:run(args.source,args.output,args.resume_pretraining)


if __name__=='__main__':main()
