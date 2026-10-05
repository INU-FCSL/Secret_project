"""환경 검증 후에만 실행하는 짧은 PPO 학습 진입점."""
import argparse
from pathlib import Path
from dataclasses import asdict
import hashlib
import subprocess
import json
import random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg, ppo_config, MODEL_PATH
from .env import StandingEnv
from .normalization import warmup, assert_unchanged, validate_rollout, pre_update_metrics
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
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iteration은 양수여야 합니다.')
    env_cfg = StandingCfg(num_envs=args.num_envs, seed=args.seed, stage=2 if args.v2_stage else args.stage,
                          v2_stage=args.v2_stage, smoothing_seconds=args.smoothing_tau,
                          episode_seconds=args.episode_seconds)
    frozen = args.normalization == 'warmup-frozen'
    if frozen:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    env = StandingEnv(env_cfg)
    cfg = ppo_config()
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
    normalization = None
    frozen_state = None
    if frozen:
        normalization, frozen_state = warmup(env, runner.alg, steps=args.warmup_steps,
            seed=args.warmup_seed if args.warmup_seed is not None else args.seed+10000,
            output=args.log_dir / 'warmup_observations.pt')
        (args.log_dir / 'normalization.json').write_text(json.dumps(normalization, indent=2, allow_nan=False))
        preflight = validate_rollout(env, runner.alg, frozen_state)
        (args.log_dir / 'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
        # 관측 준비·사전 검증 경로와 학습 시작 경로를 구분한다.
        torch.manual_seed(args.seed)
        env._rng.manual_seed(args.seed)
        env.reset()
        print(f"관측 준비 검증 통과: {normalization['observations']}개, 갱신 전 KL={preflight['exact_kl']:.9g}", flush=True)
    initial_state = runner.alg.save()
    checkpoint_infos = {'completed_updates': 0}
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
            metadata['normalization'] = normalization
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
        runner.learn(args.iterations, init_at_random_ep_len=False)
        if not all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()):
            raise RuntimeError('학습된 actor에 유한하지 않은 값이 있습니다.')
        (args.log_dir / 'losses.json').write_text(json.dumps(losses, indent=2))
        if frozen:
            assert_unchanged(runner.alg, frozen_state)
            (args.log_dir / 'normalization_final.json').write_text(json.dumps({
                'statistics_unchanged': True, 'actor_count': int(runner.alg.actor.obs_normalizer.count),
                'critic_count': int(runner.alg.critic.obs_normalizer.count),
                'training_transitions': args.iterations*cfg['num_steps_per_env']*env.num_envs,
                'normalization': normalization}, indent=2, allow_nan=False))
    finally:
        env.close()
    print(f'PPO 검증 완료: {args.log_dir}')

if __name__ == '__main__':
    main()
