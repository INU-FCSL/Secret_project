"""환경 검증 후에만 실행하는 짧은 PPO 학습 진입점."""
import argparse
from pathlib import Path
from dataclasses import asdict
import hashlib
import subprocess
import json
import random
import sys
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg, ppo_config, MODEL_PATH, REWARD_WEIGHTS
from .env import StandingEnv
from .normalization import warmup, assert_unchanged, validate_rollout, pre_update_metrics, configure
from .diagnose_ppo import instrument_update


def main():
    parser = argparse.ArgumentParser(description='MicroDog 기립 PPO 초기 실행')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--log-dir', type=Path, default=Path('/tmp/microdog_standing_smoke'))
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--episode-seconds', type=float, default=20.)
    parser.add_argument('--smoothing-tau', type=float, default=.75)
    parser.add_argument('--v2-stage', choices=['A','B','C','D'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--normalization', choices=['running', 'warmup-frozen'], default='running')
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--warmup-seed', type=int)
    parser.add_argument('--orientation-reward-scale', type=float, default=.05)
    parser.add_argument('--pose-reward-weight',type=float,default=REWARD_WEIGHTS['pose'])
    parser.add_argument('--gate-after-25', action='store_true')
    parser.add_argument('--v2-baseline', type=Path)
    parser.add_argument('--v3a-baseline', type=Path)
    parser.add_argument('--action-mapping',choices=['clip','tanh','identity'],default='clip')
    parser.add_argument('--policy-distribution',choices=['gaussian','squashed'],default='gaussian')
    parser.add_argument('--mean-regularization',type=float,default=0.)
    parser.add_argument('--terminal-bootstrap',action='store_true')
    parser.add_argument('--critic-warm-start',type=Path)
    parser.add_argument('--rollout',type=int,choices=[16,32],default=16)
    parser.add_argument('--ablation-parent',type=Path)
    parser.add_argument('--ablation-name',choices=['v3c','v3d','v3e','v4a','v4b','v5a','v5b','v6fix','v7a','v7b'])
    parser.add_argument('--previous-action-normalization',choices=['running','identity'],default='running')
    parser.add_argument('--standing-basis',type=Path)
    args = parser.parse_args()
    if (args.policy_distribution=='squashed') != (args.action_mapping=='identity'):
        parser.error('squashed distribution과 identity mapping은 함께 사용해야 합니다.')
    if args.iterations < 1:
        parser.error('iteration은 양수여야 합니다.')
    if args.ablation_parent and (args.iterations!=100 or args.normalization!='warmup-frozen' or
                                args.ablation_name is None or args.gate_after_25):
        parser.error('후보 판정에는 100회 상한·정규화 고정·후보 이름이 필요하며 이전 gate와 동시에 사용할 수 없습니다.')
    if args.gate_after_25 and (args.iterations != 100 or args.normalization != 'warmup-frozen'
                              or args.v2_baseline is None or args.v3a_baseline is None):
        parser.error('25회 판정은 100회 상한, 정규화 고정, V2/V3-A 기준 경로가 필요합니다.')
    env_cfg = StandingCfg(num_envs=args.num_envs, seed=args.seed, stage=2 if args.v2_stage else args.stage,
                          v2_stage=args.v2_stage, smoothing_seconds=args.smoothing_tau,
                          episode_seconds=args.episode_seconds,
                          orientation_reward_scale=args.orientation_reward_scale,pose_reward_weight=args.pose_reward_weight,
                          action_mapping=args.action_mapping,
                          previous_action_normalization=args.previous_action_normalization,
                          action_basis='standing' if args.standing_basis else 'joint',
                          standing_basis=json.loads(args.standing_basis.read_text())['matrix'] if args.standing_basis else None)
    frozen = args.normalization == 'warmup-frozen'
    if frozen:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    env = StandingEnv(env_cfg)
    cfg = ppo_config(args.policy_distribution,args.mean_regularization,args.terminal_bootstrap)
    cfg['num_steps_per_env']=args.rollout
    if frozen:
        cfg['seed'] = args.seed
    cfg['max_iterations'] = args.iterations
    cfg['save_interval'] = 25
    args.log_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = json.loads(json.dumps(cfg))
    (args.log_dir / 'config.json').write_text(json.dumps(config_snapshot, indent=2))
    (args.log_dir / 'environment.json').write_text(json.dumps({
        'config': asdict(env_cfg), 'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'reference_calibration': env.reference_calibration,
        'reference_orientation': env.reference_standing_orientation.cpu().tolist(),
        'reference_height': env.reference_standing_height,
        'gpu': torch.cuda.get_device_name(env.device)}, indent=2))
    runner = OnPolicyRunner(env, cfg, log_dir=str(args.log_dir), device=env.device)
    configure(runner.alg,env_cfg.previous_action_normalization)
    if args.critic_warm_start is not None:
        # critic의 가중치만 교체하고 actor·새 warm-up의 정규화·PPO optimizer는 유지한다.
        state=torch.load(args.critic_warm_start,weights_only=False,map_location=env.device)
        runner.alg.critic.mlp.load_state_dict({key.removeprefix('mlp.'):value for key,value in state.items() if key.startswith('mlp.')})
    normalization = None
    frozen_state = None
    if frozen:
        normalization, frozen_state = warmup(env, runner.alg, steps=args.warmup_steps,
            seed=args.warmup_seed if args.warmup_seed is not None else args.seed+10000,
            output=args.log_dir / 'warmup_observations.pt')
        (args.log_dir / 'normalization.json').write_text(json.dumps(normalization, indent=2, allow_nan=False))
        preflight = validate_rollout(env, runner.alg, frozen_state)
        preflight['storage']=dict(steps=runner.alg.storage.num_transitions_per_env,
            actions_shape=list(runner.alg.storage.actions.shape),returns_finite=bool(torch.isfinite(runner.alg.storage.returns).all()),
            advantages_finite=bool(torch.isfinite(runner.alg.storage.advantages).all()),
            mini_batch_count=cfg['algorithm']['num_mini_batches'],epochs=cfg['algorithm']['num_learning_epochs'],
            gamma=cfg['algorithm']['gamma'],lambda_value=cfg['algorithm']['lam'])
        (args.log_dir / 'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
        # 관측 준비·사전 검증 경로와 학습 시작 경로를 구분한다.
        torch.manual_seed(args.seed)
        env._rng.manual_seed(args.seed)
        env.reset()
        print(f"관측 준비 검증 통과: {normalization['observations']}개, 갱신 전 KL={preflight['exact_kl']:.9g}", flush=True)
    initial_state = runner.alg.save()
    checkpoint_infos = {'completed_updates': 0}
    checkpoint_infos['policy_distribution'] = args.policy_distribution
    checkpoint_infos['mean_regularization'] = args.mean_regularization
    checkpoint_infos['terminal_bootstrap'] = args.terminal_bootstrap
    checkpoint_infos['critic_initialization'] = 'MC warm-start' if args.critic_warm_start is not None else 'default'
    if args.critic_warm_start is not None:
        checkpoint_infos['critic_warm_start_sha256']=hashlib.sha256(args.critic_warm_start.read_bytes()).hexdigest()
    checkpoint_infos['orientation_reward_scale'] = env_cfg.orientation_reward_scale
    checkpoint_infos['pose_reward_weight'] = env_cfg.pose_reward_weight
    checkpoint_infos['action_mapping'] = env_cfg.action_mapping
    checkpoint_infos['previous_action_normalization'] = env_cfg.previous_action_normalization
    checkpoint_infos['action_basis'] = env_cfg.action_basis
    checkpoint_infos['standing_basis'] = env_cfg.standing_basis
    if frozen:
        checkpoint_infos['normalization'] = normalization
    initial_state.update(iter=0, infos=checkpoint_infos)
    torch.save(initial_state, args.log_dir / 'checkpoint_0.pt')
    losses = []
    metrics = []
    if frozen:
        native_save = runner.save
        def save_frozen_checkpoint(path, infos=None):
            metadata = dict(infos or {})
            metadata['policy_distribution'] = args.policy_distribution
            metadata['mean_regularization'] = args.mean_regularization
            metadata['terminal_bootstrap'] = args.terminal_bootstrap
            metadata['critic_initialization'] = checkpoint_infos['critic_initialization']
            if args.critic_warm_start is not None:
                metadata['critic_warm_start_sha256']=checkpoint_infos['critic_warm_start_sha256']
            metadata['normalization'] = normalization
            metadata['orientation_reward_scale'] = env_cfg.orientation_reward_scale
            metadata['pose_reward_weight'] = env_cfg.pose_reward_weight
            metadata['action_mapping'] = env_cfg.action_mapping
            metadata['previous_action_normalization'] = env_cfg.previous_action_normalization
            metadata['action_basis'] = env_cfg.action_basis
            metadata['standing_basis'] = env_cfg.standing_basis
            metadata.setdefault('completed_updates', len(metrics))
            native_save(path, infos=metadata)
        runner.save = save_frozen_checkpoint
    batch = []
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device)
    original_step = env.step
    def logged_step(actions):
        obs, reward, done, extras = original_step(actions)
        d = extras['diagnostics']
        returns.add_(reward)
        lengths.add_(1)
        completed = done.bool()
        batch.append(dict(step_reward=float(reward.mean()),
            completed_returns=returns[completed].cpu().tolist(),
            completed_lengths=lengths[completed].cpu().tolist(),
            self_collision_failures=int(d['reasons']['self_collision'].sum()),
            failures=int(d['terminated'].sum()), completed=int(completed.sum()),
            self_contact_env_steps=int((d['self_contacts']>0).sum()),
            unexpected_contact_env_steps=int(d['reasons']['abnormal_ground'].sum()),
            saturation_samples=int(d['saturation_steps'].sum()),
            peak_torque=float(d['peak_torque'].max()),
            reward_terms={k: float(v.mean()) for k, v in d['reward_terms'].items()}))
        returns[completed] = 0
        lengths[completed] = 0
        return obs, reward, done, extras
    env.step = logged_step
    update = runner.alg.update
    def checked_update():
        diagnostic = None
        if frozen:
            assert_unchanged(runner.alg, frozen_state)
            before = pre_update_metrics(runner.alg)
            if abs(before['exact_kl']) > 1e-7 or before['clip_fraction'] > 1e-7:
                raise RuntimeError('학습 중 갱신 전 정책 분포가 변경되었습니다.')
            diagnostic = instrument_update(runner.alg, update_fn=update)
            result = diagnostic['loss']
            diagnostic['pre_update'] = before
            assert_unchanged(runner.alg, frozen_state)
        else:
            result = update()
        if not all(torch.isfinite(torch.as_tensor(v)).all() for v in result.values()):
            raise RuntimeError('PPO loss에 NaN 또는 무한값이 있습니다.')
        losses.append({k: float(v) for k, v in result.items()})
        completed_returns = [x for row in batch for x in row['completed_returns']]
        completed_lengths = [x for row in batch for x in row['completed_lengths']]
        completed = sum(row['completed'] for row in batch)
        failures = sum(row['failures'] for row in batch)
        metrics.append(dict(iteration=len(losses),
            mean_step_reward=sum(row['step_reward'] for row in batch)/len(batch),
            mean_episode_return=sum(completed_returns)/len(completed_returns) if completed_returns else None,
            mean_episode_length=sum(completed_lengths)/len(completed_lengths) if completed_lengths else None,
            completed_episodes=completed, failures=failures,
            termination_rate=failures/completed if completed else None,
            termination_step_fraction=failures/(len(batch)*env.num_envs),
            self_collision_failures=sum(row['self_collision_failures'] for row in batch),
            self_contact_env_steps=sum(row['self_contact_env_steps'] for row in batch),
            unexpected_contact_env_steps=sum(row['unexpected_contact_env_steps'] for row in batch),
            saturation_samples=sum(row['saturation_samples'] for row in batch),
            peak_torque=max(row['peak_torque'] for row in batch),
            reward_terms={k: sum(row['reward_terms'][k] for row in batch)/len(batch)
                          for k in batch[0]['reward_terms']},
            learning_rate=runner.alg.learning_rate, **losses[-1]))
        batch.clear()
        if diagnostic is not None:
            metrics[-1]['diagnostic'] = diagnostic
        (args.log_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        if len(losses) in (25, 50, 75, 100):
            infos = {'completed_updates': len(losses)}
            if frozen:
                infos['normalization'] = normalization
            runner.save(str(args.log_dir / f'checkpoint_{len(losses)}.pt'), infos=infos)
        return result
    runner.alg.update = checked_update
    try:
        if args.ablation_parent:
            runner.learn(25,init_at_random_ep_len=False)
            evaluation=args.log_dir.parent/'evaluation'
            command=[sys.executable,'-m','rl.ablation','--candidate',str(args.log_dir),
                '--parent',str(args.ablation_parent),'--name',args.ablation_name,'--checkpoints','0','25',
                '--output',str(evaluation)]
            subprocess.run(command,check=True)
            decision=json.loads((evaluation/'decision.json').read_text())
            (args.log_dir/'gate_25.json').write_text(json.dumps(decision,indent=2,allow_nan=False))
            if decision['status'] in ('SUCCESS','PARTIAL'):
                runner.current_learning_iteration=len(metrics)
                runner.learn(75,init_at_random_ep_len=False)
                command[command.index('--checkpoints')+1:command.index('--output')]=['50','75','100']
                subprocess.run(command,check=True)
                decision=json.loads((evaluation/'decision.json').read_text())
            print(f"후보 판정: {args.ablation_name}, {decision['status']}, 완료={len(metrics)}회",flush=True)
        elif args.gate_after_25:
            runner.learn(25, init_at_random_ep_len=False)
            gate_output=args.log_dir.parent/'gate_evaluation'
            command=[sys.executable,'-m','rl.compare_standing','--v2',str(args.v2_baseline),
                '--v3',str(args.v3a_baseline),'--v3b',str(args.log_dir),'--checkpoints','0','25',
                '--legacy-checkpoints','0','25','100','--orientation-reward-scale',str(env_cfg.orientation_reward_scale),
                '--output',str(gate_output)]
            # 별도 프로세스에서 평가해 학습 환경·optimizer·난수 상태를 보존한다.
            subprocess.run(command,check=True)
            gate=json.loads((gate_output/'gate.json').read_text())
            (args.log_dir/'gate.json').write_text(json.dumps(gate,indent=2,allow_nan=False))
            print(f"25회 판정: 100회 연장={gate['extend_to_100']}",flush=True)
            if gate['extend_to_100']:
                runner.current_learning_iteration=len(metrics)
                runner.learn(75, init_at_random_ep_len=False)
        else:
            runner.learn(args.iterations, init_at_random_ep_len=False)
        if not all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()):
            raise RuntimeError('학습된 actor에 유한하지 않은 값이 있습니다.')
        (args.log_dir / 'losses.json').write_text(json.dumps(losses, indent=2))
        if frozen:
            assert_unchanged(runner.alg, frozen_state)
            (args.log_dir / 'normalization_final.json').write_text(json.dumps({
                'statistics_unchanged': True, 'actor_count': int(runner.alg.actor.obs_normalizer.count),
                'critic_count': int(runner.alg.critic.obs_normalizer.count),
                'completed_updates': len(metrics),
                'training_transitions': len(metrics)*cfg['num_steps_per_env']*env.num_envs,
                'normalization': normalization}, indent=2, allow_nan=False))
    finally:
        env.close()
    print(f'PPO 검증 완료: {args.log_dir}')

if __name__ == '__main__':
    main()
