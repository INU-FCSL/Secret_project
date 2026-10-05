"""환경 검증 후에만 실행하는 짧은 PPO 학습 진입점."""
import argparse
from pathlib import Path
from dataclasses import asdict
import hashlib
import subprocess
import json
import torch
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg, ppo_config, MODEL_PATH
from .env import StandingEnv


def main():
    parser = argparse.ArgumentParser(description='MicroDog 기립 PPO 초기 실행')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--log-dir', type=Path, default=Path('/tmp/microdog_standing_smoke'))
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--episode-seconds', type=float, default=20.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iteration은 양수여야 합니다.')
    env_cfg = StandingCfg(num_envs=args.num_envs, seed=args.seed, stage=args.stage,
                          episode_seconds=args.episode_seconds)
    env = StandingEnv(env_cfg)
    cfg = ppo_config()
    cfg['max_iterations'] = args.iterations
    cfg['save_interval'] = 25
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
    (args.log_dir / 'environment.json').write_text(json.dumps({
        'config': asdict(env_cfg), 'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'gpu': torch.cuda.get_device_name(env.device)}, indent=2))
    runner = OnPolicyRunner(env, cfg, log_dir=str(args.log_dir), device=env.device)
    initial_state = runner.alg.save()
    initial_state.update(iter=0, infos={'completed_updates': 0})
    torch.save(initial_state, args.log_dir / 'checkpoint_0.pt')
    losses = []
    metrics = []
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
            self_contact_env_steps=sum(row['self_contact_env_steps'] for row in batch),
            unexpected_contact_env_steps=sum(row['unexpected_contact_env_steps'] for row in batch),
            saturation_samples=sum(row['saturation_samples'] for row in batch),
            peak_torque=max(row['peak_torque'] for row in batch),
            reward_terms={k: sum(row['reward_terms'][k] for row in batch)/len(batch)
                          for k in batch[0]['reward_terms']},
            learning_rate=runner.alg.learning_rate, **losses[-1]))
        batch.clear()
        (args.log_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        if len(losses) in (25, 50, 75, 100):
            runner.save(str(args.log_dir / f'checkpoint_{len(losses)}.pt'),
                        infos={'completed_updates': len(losses)})
        return result
    runner.alg.update = checked_update
    try:
        runner.learn(args.iterations, init_at_random_ep_len=False)
        if not all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()):
            raise RuntimeError('학습된 actor에 유한하지 않은 값이 있습니다.')
        (args.log_dir / 'losses.json').write_text(json.dumps(losses, indent=2))
    finally:
        env.close()
    print(f'PPO 검증 완료: {args.log_dir}')

if __name__ == '__main__':
    main()
