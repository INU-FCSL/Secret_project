"""환경 검증 후에만 실행하는 짧은 PPO 학습 진입점."""
import argparse
from pathlib import Path
import json
import torch
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg, ppo_config
from .env import StandingEnv


def main():
    parser = argparse.ArgumentParser(description='MicroDog 기립 PPO 초기 실행')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--log-dir', type=Path, default=Path('/tmp/microdog_standing_smoke'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iteration은 양수여야 합니다.')
    env = StandingEnv(StandingCfg(num_envs=args.num_envs, seed=args.seed))
    cfg = ppo_config()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
    runner = OnPolicyRunner(env, cfg, log_dir=str(args.log_dir), device=env.device)
    losses = []
    update = runner.alg.update
    def checked_update():
        result = update()
        if not all(torch.isfinite(torch.as_tensor(v)).all() for v in result.values()):
            raise RuntimeError('PPO loss에 NaN 또는 무한값이 있습니다.')
        losses.append({k: float(v) for k, v in result.items()})
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
