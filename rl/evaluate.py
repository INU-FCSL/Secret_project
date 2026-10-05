"""공통 평가 조건에서 기립 복구 지표를 집계한다."""
import argparse
import json
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg, ppo_config
from .env import StandingEnv


@torch.inference_mode()
def evaluate(cfg, checkpoint=None, random=False):
    env = StandingEnv(cfg)
    policy = None
    if checkpoint is not None:
        runner = OnPolicyRunner(env, ppo_config(), log_dir=None, device=cfg.device)
        runner.load(str(checkpoint), load_cfg={'actor': True, 'critic': True})
        policy = runner.get_inference_policy(device=cfg.device)
    # 정책 구성으로 난수 상태가 바뀌어도 동일한 초기 조건을 사용한다.
    env._rng.manual_seed(cfg.seed)
    obs = env.reset()
    rng = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 100000)
    n = cfg.num_envs
    device = cfg.device
    alive = torch.ones(n, dtype=torch.bool, device=device)
    fall = torch.zeros(n, dtype=torch.bool, device=device)
    totals = {k: torch.zeros(n, device=device) for k in
              ('return', 'steps', 'roll', 'pitch', 'height_error', 'feet', 'saturation', 'self', 'unexpected')}
    maxroll = torch.zeros(n, device=device)
    maxpitch = maxroll.clone()
    peak = maxroll.clone()
    minheight = torch.full((n,), float('inf'), device=device)
    recovery = torch.full((n,), float('nan'), device=device)
    stable = torch.zeros(n, dtype=torch.long, device=device)
    reset_stable = stable.clone()
    reset_recovery = torch.full((n,), float("nan"), device=device)
    post_push_maxroll = torch.zeros(n, device=device)
    post_push_maxpitch = post_push_maxroll.clone()
    push_start = env.push_start.clone()
    push_end = push_start + cfg.push_seconds
    push_present = env.push_vector.norm(dim=-1) > 0
    try:
        for step in range(env.max_episode_length):
            if policy is not None:
                action = policy(obs)
            elif random:
                action = 2 * torch.rand((n, 12), generator=rng, device=device) - 1
            else:
                action = torch.zeros((n, 12), device=device)
            obs, reward, done, extras = env.step(action)
            d = extras['diagnostics']
            roll = d['roll'].abs().rad2deg()
            pitch = d['pitch'].abs().rad2deg()
            totals['return'] += reward * alive
            totals['steps'] += alive
            for name, value in [('roll', roll), ('pitch', pitch),
                                ('height_error', (d['height']-.19272).abs()),
                                ('feet', d['feet'].all(-1)),
                                ('saturation', d['saturation_steps'].sum(-1)/(12*cfg.decimation)),
                                ('self', d['self_contacts'] > 0),
                                ('unexpected', d['reasons']['abnormal_ground'])]:
                totals[name] += value * alive
            maxroll = torch.maximum(maxroll, torch.where(alive, roll, 0))
            maxpitch = torch.maximum(maxpitch, torch.where(alive, pitch, 0))
            peak = torch.maximum(peak, torch.where(alive, d['peak_torque'].max(-1).values, 0))
            minheight = torch.minimum(minheight, torch.where(alive, d['height'], float('inf')))
            time = (step+1)*cfg.step_dt
            # 0.6도, 높이 5mm, 네 발 접지 상태를 0.2초 연속 유지해야 복구로 센다.
            reset_good = alive & (time < push_start) & (roll < .6) & (pitch < .6)
            reset_good &= (d['height']-.19272).abs() < .005
            reset_good &= d['feet'].all(-1)
            reset_stable = torch.where(reset_good, reset_stable+1, 0)
            reset_done = (reset_stable >= 10) & reset_recovery.isnan()
            reset_recovery[reset_done] = max(0., time-.18)
            post_push = alive & push_present & (time >= push_start)
            post_push_maxroll = torch.maximum(post_push_maxroll, torch.where(post_push, roll, 0))
            post_push_maxpitch = torch.maximum(post_push_maxpitch, torch.where(post_push, pitch, 0))
            good = alive & push_present & (time >= push_end) & (roll < .6) & (pitch < .6)
            good &= (d['height']-.19272).abs() < .005
            good &= d['feet'].all(-1)
            stable = torch.where(good, stable+1, 0)
            recovered = (stable >= 10) & recovery.isnan()
            recovery[recovered] = (time-push_end[recovered]-.18).clamp_min(0)
            fall |= alive & d['terminated']
            alive &= ~done.bool()
        steps = totals['steps'].clamp_min(1)
        safe = (~fall) & (totals['self'] == 0) & (totals['unexpected'] == 0)
        safe &= totals['saturation']/steps < .01
        safe &= ~reset_recovery.isnan()
        safe &= ~push_present | ~recovery.isnan()
        result = dict(episodes=n, success_rate=float(safe.float().mean()),
            reset_recovery_rate=float((~reset_recovery.isnan()).float().mean()),
            reset_recovery_seconds=float(reset_recovery.nanmean()) if (~reset_recovery.isnan()).any() else None,
            post_push_max_roll=float(post_push_maxroll.max()), post_push_max_pitch=float(post_push_maxpitch.max()),
            survival_rate=float((~fall).float().mean()),
            termination_rate=float(fall.float().mean()), mean_return=float(totals['return'].mean()),
            episode_seconds=float((steps*cfg.step_dt).mean()), max_roll=float(maxroll.max()),
            max_pitch=float(maxpitch.max()), mean_abs_roll=float((totals['roll']/steps).mean()),
            mean_abs_pitch=float((totals['pitch']/steps).mean()),
            mean_height_error=float((totals['height_error']/steps).mean()), min_height=float(minheight.min()),
            four_foot_fraction=float((totals['feet']/steps).mean()),
            saturation_fraction=float((totals['saturation']/steps).mean()), peak_torque=float(peak.max()),
            self_contact_env_steps=int(totals['self'].sum()), unexpected_contact_env_steps=int(totals['unexpected'].sum()),
            push_recovery_rate=float((~recovery.isnan()).float().mean()) if push_present.any() else None,
            recovery_seconds=float(recovery.nanmean()) if push_present.any() and (~recovery.isnan()).any() else None,
            seed=cfg.seed, stage=cfg.stage, push_force=cfg.push_force,
            per_episode_return=totals['return'].cpu().tolist(),
            per_episode_recovery=[None if x != x else x for x in recovery.cpu().tolist()])
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='동일 조건의 기립 정책 평가')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--push-force', type=float)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(StandingCfg(num_envs=args.num_envs, stage=args.stage, seed=args.seed,
        episode_seconds=args.seconds, push_force=args.push_force), args.checkpoint, args.random)
    args.output.write_text(json.dumps(result, indent=2))
    print(f'평가 결과 저장: {args.output}')

if __name__ == '__main__':
    main()
