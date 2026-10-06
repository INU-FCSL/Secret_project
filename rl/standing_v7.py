"""선정된 pose weight 하나로 좋은 초기 정책의 PPO drift를 검사한다."""
import argparse
import copy
import json
import shutil
from dataclasses import replace,asdict
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg,ppo_config
from .env import StandingEnv
from .normalization import restore,snapshot,assert_unchanged,pre_update_metrics
from .diagnose_ppo import instrument_update
from .ablation import measure,safe,correct_feedback
from .reward_alignment import save,TERMS


def components(summary):
    data=dict(np.load(summary['trace']));time=np.arange(len(data['reward']))[:,None]*.02;start=data['push_start']
    phases={'pre':time<start,'during':(time>=start)&(time<start+.5),
        'recovery':(time>=start+.5)&(time<start+1.5),'steady':time>=start+1.5,'episode':np.ones_like(data['valid'])}
    return {name:dict(zip(TERMS,(data['terms'].astype(float)*(mask&data['valid'])[...,None]).sum(0).mean(0).tolist()))
        for name,mask in phases.items()}


def finetune(output,weight,critic_path=None):
    source=Path('/tmp/microdog_v6/imitation/ppo/checkpoint_0.pt')
    original=json.loads((source.parent/'environment.json').read_text())
    cfg=replace(StandingCfg(**original['config']),pose_reward_weight=weight)
    env=StandingEnv(cfg);config=ppo_config('squashed',terminal_bootstrap=True);config['num_steps_per_env']=32
    runner=OnPolicyRunner(env,config,device=env.device)
    infos=copy.deepcopy(runner.load(str(source)));restore(runner.alg,infos)
    infos.update(pose_reward_weight=weight,completed_updates=0)
    expected=snapshot(runner.alg);source_state=torch.load(source,weights_only=False,map_location='cpu')
    assert len(runner.alg.optimizer.state)==0
    for key,value in runner.alg.critic.state_dict().items():assert torch.equal(value.cpu(),source_state['critic_state_dict'][key])
    if critic_path is not None:
        runner.alg.critic.load_state_dict(torch.load(critic_path,weights_only=False,map_location=env.device))
        assert_unchanged(runner.alg,expected)
        infos['critic_initialization']='same-policy MC warm-start'
    infos['experiment']='v7b' if critic_path is not None else 'v7a'
    valid=torch.load('/tmp/microdog_v6/imitation/dataset.pt',weights_only=False)['observation'][3].reshape(-1,42).to(env.device)
    td=TensorDict({'actor':valid},batch_size=[len(valid)])
    with torch.no_grad():initial_mu=runner.alg.actor.mlp(runner.alg.actor.get_latent(td)).clone()
    directory=output/'ppo';directory.mkdir(parents=True,exist_ok=True)
    original['config']=asdict(cfg);save(directory/'environment.json',original);save(directory/'config.json',config)
    save(directory/'normalization.json',infos['normalization'])
    runner.logger.writer=None;native_save=runner.save
    def checkpoint(path,metadata=None):native_save(str(path),infos=dict(infos,**(metadata or {})))
    checkpoint(directory/'checkpoint_0.pt');runner.save=checkpoint
    updates=[];evaluations=[];training=[]
    original_step=env.step
    def logged_step(action):
        obs,reward,done,extras=original_step(action);d=extras['diagnostics']
        training.append(dict(self_collision=int(d['reasons']['self_collision'].sum()),
            abnormal=int(d['reasons']['abnormal_ground'].sum()),invalid=int(d['reasons']['invalid'].sum()),
            saturation=int(d['saturation_steps'].sum()),peak_torque=float(d['peak_torque'].max())))
        return obs,reward,done,extras
    env.step=logged_step
    original_update=runner.alg.update
    def checked_update():
        before=pre_update_metrics(runner.alg)
        if before['exact_kl']>1e-7 or before['clip_fraction']>1e-7:raise RuntimeError('V7 갱신 전 분포 불일치')
        diagnostic=instrument_update(runner.alg,update_fn=original_update);assert_unchanged(runner.alg,expected)
        updates.append(dict(update=len(updates)+1,pre_update=before,diagnostic=diagnostic,
            training=dict(self_collision=sum(t['self_collision'] for t in training),abnormal=sum(t['abnormal'] for t in training),
                invalid=sum(t['invalid'] for t in training),saturation=sum(t['saturation'] for t in training),
                peak_torque=max(t['peak_torque'] for t in training))))
        training.clear();save(output/'updates.json',updates);return diagnostic['loss']
    runner.alg.update=checked_update
    def evaluate_checkpoint(i):
        target=output/'evaluation'/f'checkpoint_{i}';target.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,'-m','rl.standing_v7','--evaluate-checkpoint',str(directory/f'checkpoint_{i}.pt'),
            '--eval-output',str(target)],check=True)
        result=json.loads((target/'summary.json').read_text())
        with torch.no_grad():
            drift=runner.alg.actor.mlp(runner.alg.actor.get_latent(td))-initial_mu
        result['common_observation_drift']=dict(mean=drift.mean(0).cpu().tolist(),rms=drift.square().mean(0).sqrt().cpu().tolist(),
            total_rms=float(drift.square().mean().sqrt()))
        result['reward_components']=components(result);save(target/'summary.json',result);evaluations.append(result)
        return result
    try:
        env._rng.manual_seed(42424);torch.manual_seed(42424);env.reset();evaluate_checkpoint(0)
        for i in range(1,6):
            runner.current_learning_iteration=i-1;runner.learn(1,init_at_random_ep_len=False)
            checkpoint(directory/f'checkpoint_{i}.pt',dict(completed_updates=i));evaluate_checkpoint(i)
        zero=json.loads(Path('/tmp/microdog_v6/fix/evaluation/summary.json').read_text())['baselines']['zero']
        old=json.loads(Path('/tmp/microdog_v6/results.json').read_text())['updates'][5]
        old_drift=float(np.sqrt(np.mean(np.square(old['common_observation_delta_mu_rms']))))
        if critic_path is not None:
            parent=json.loads(Path('/tmp/microdog_v7/v7a/decision.json').read_text())['per_update'][5]
            old=dict(metrics=parent['metrics']);old_drift=parent['drift']['total_rms']
        last=evaluations[-1]
        seed_success=all(row['peak_orientation_error']<z['peak_orientation_error'] and
            row['integrated_orientation_error']<z['integrated_orientation_error'] for row,z in zip(last['per_seed'],zero['per_seed']))
        reduction=1-last['aggregate']['integrated_orientation_error']/old['metrics']['integrated_orientation_error']
        drift_improved=last['common_observation_drift']['total_rms']<old_drift
        extend=safe(last) and seed_success and correct_feedback(last) and reduction>=.15 and drift_improved
        eligible=[(i,row) for i,row in enumerate(evaluations) if safe(row)]
        best_i,best=min(eligible,key=lambda pair:(pair[1]['aggregate']['integrated_orientation_error'],
            pair[1]['aggregate']['peak_orientation_error'],pair[1]['aggregate']['recovery_seconds'] or 0))
        best_ppo_i,best_ppo=min([pair for pair in eligible if pair[0]>0],key=lambda pair:(
            pair[1]['aggregate']['integrated_orientation_error'],pair[1]['aggregate']['peak_orientation_error'],
            pair[1]['aggregate']['recovery_seconds'] or 0,not correct_feedback(pair[1])))
        result=dict(pose_reward_weight=weight,only_experimental_change='critic warm-start' if critic_path is not None else 'pose reward weight',
            per_update=[dict(update=i,metrics=row['aggregate'],drift=row['common_observation_drift']) for i,row in enumerate(evaluations)],
            final_seed_success=seed_success,safe=safe(last),feedback_correct=correct_feedback(last),
            integrated_reduction_vs_parent=reduction,parent='v7a_ppo5' if critic_path is not None else 'v6_ppo5',
            common_mu_drift_improved=drift_improved,
            old_common_mu_drift_rms=old_drift,new_common_mu_drift_rms=last['common_observation_drift']['total_rms'],
            from_scratch_allowed=extend,threshold_integrated_reduction=.15,best_checkpoint=best_i,best=best['aggregate'],
            best_ppo_checkpoint=best_ppo_i,best_ppo=best_ppo['aggregate'])
        save(output/'decision.json',result)
        shutil.copy2(directory/f'checkpoint_{best_i}.pt',output/'best_checkpoint.pt')
        shutil.copy2(directory/f'checkpoint_{best_ppo_i}.pt',output/'best_ppo_checkpoint.pt')
        print('V7 후보 판정: from-scratch 실행 조건',extend,'누적 오차 개선',reduction,flush=True)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description='V7 pose 단일 변경과 좋은 정책 유지 검증')
    parser.add_argument('--output',type=Path,default=Path('/tmp/microdog_v7/v7a'))
    parser.add_argument('--pose-weight',type=float,default=.1)
    parser.add_argument('--critic-warm-start',type=Path)
    parser.add_argument('--evaluate-checkpoint',type=Path);parser.add_argument('--eval-output',type=Path)
    args=parser.parse_args()
    if args.evaluate_checkpoint:
        cfg=StandingCfg(**json.loads((args.evaluate_checkpoint.parent/'environment.json').read_text())['config'])
        infos=torch.load(args.evaluate_checkpoint,weights_only=False,map_location='cpu')['infos']
        summary=measure(cfg,args.eval_output,infos.get('experiment','v7a'),args.evaluate_checkpoint);save(args.eval_output/'summary.json',summary)
    else:finetune(args.output,args.pose_weight,args.critic_warm_start)


if __name__=='__main__':main()
