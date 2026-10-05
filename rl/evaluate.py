"""동일한 초기 오차와 외력 조건에서 자연 평형 복구를 평가한다."""
import argparse
import json
from pathlib import Path
import math
import torch
from rsl_rl.runners import OnPolicyRunner
from .config import StandingCfg, ppo_config
from .env import StandingEnv
from .normalization import restore


class ScriptedController:
    """기존 자세·각속도 관측만 사용하는 두 knee 차등 기저의 기준 제어기."""
    def __init__(self, env):
        from .control_authority import AuthorityProbe, patterns
        probe = AuthorityProbe()
        columns = []
        names = ('roll_knee', 'pitch_knee')
        for name in names:
            plus = probe.run(action=.5*patterns()[name], seconds=4.)
            minus = probe.run(action=-.5*patterns()[name], seconds=4.)
            columns.append([a-b for a,b in zip(plus['final_delta'][:2],minus['final_delta'][:2])])
        response = torch.tensor(columns, device=env.device).T
        self.inverse = torch.linalg.inv(response)
        self.basis = torch.tensor([patterns()[name].tolist() for name in names], device=env.device).T
        self.reference = env.reference_standing_orientation[:2]

    def __call__(self, obs):
        gravity = obs['actor'][:, :3]
        rpy = torch.stack((torch.atan2(-gravity[:,1],-gravity[:,2]),
                           torch.asin(gravity[:,0].clamp(-1,1))), -1)
        omega = obs['actor'][:,3:5]
        coefficients = torch.rad2deg(2*(self.reference-rpy)-.2*omega) @ self.inverse.T
        return coefficients.clamp(-1,1) @ self.basis.T


@torch.inference_mode()
def evaluate(cfg, checkpoint=None, random=False, scripted=False, *, step_callback=None):
    if sum((checkpoint is not None,random,scripted)) > 1:
        raise ValueError('평가 정책은 하나만 선택해야 합니다.')
    env = StandingEnv(cfg)
    try:
        policy = None
        if checkpoint is not None:
            runner = OnPolicyRunner(env, ppo_config(), log_dir=None, device=cfg.device)
            infos = runner.load(str(checkpoint), load_cfg={'actor': True, 'critic': True})
            restore(runner.alg, infos)
            policy = runner.get_inference_policy(device=cfg.device)
        elif scripted:
            policy = ScriptedController(env)
        # 정책 구성에 따른 난수 소비와 초기 조건을 분리한다.
        env._rng.manual_seed(cfg.seed)
        obs = env.reset()
        rng = torch.Generator(device=cfg.device).manual_seed(cfg.seed+100000)
        n,device = cfg.num_envs,cfg.device
        zero = lambda: torch.zeros(n, device=device)
        alive = torch.ones(n,dtype=torch.bool,device=device)
        fall = torch.zeros(n,dtype=torch.bool,device=device)
        totals = {k:zero() for k in ('return','steps','roll','pitch','height_error','feet','saturation','self','unexpected')}
        peak = zero();episode_peak = zero();integrated = zero();episode_integrated = zero()
        during_peak = zero();during_integrated = zero();peak_torque = zero()
        maxroll,maxpitch = zero(),zero()
        minheight = torch.full((n,),float('inf'),device=device)
        recovery = torch.full((n,),float('nan'),device=device)
        reset_recovery = recovery.clone();legacy_recovery = recovery.clone()
        stable = torch.zeros(n,dtype=torch.long,device=device)
        reset_stable,legacy_stable = stable.clone(),stable.clone()
        push_start = env.push_start.clone()
        push_end = push_start+cfg.disturbance_seconds
        push_present = env.push_vector.norm(dim=-1)>0
        directions = env.push_vector.cpu().tolist()
        reference = env.reference_standing_orientation[:2]
        height_reference = env.reference_standing_height
        reasons = {name:zero() for name in ('fall','self_collision','invalid_state','abnormal_ground_contact','timeout')}
        for step in range(env.max_episode_length):
            evaluation_observation = obs
            if policy is not None:
                action = policy(obs)
            elif random:
                action = 2*torch.rand((n,12),generator=rng,device=device)-1
            else:
                action = torch.zeros((n,12),device=device)
            obs,reward,done,extras = env.step(action)
            d = extras['diagnostics']
            if step_callback is not None:
                step_callback(env,evaluation_observation,action,reward,done,extras,alive.clone(),(step+1)*cfg.step_dt)
            rpy = torch.stack((d['roll'],d['pitch']),-1)
            error = torch.rad2deg(rpy-reference)
            tilt = torch.nan_to_num(error.norm(dim=-1),nan=180.,posinf=180.)
            time = (step+1)*cfg.step_dt
            after = alive & ((time>=push_start)|~push_present)
            during = alive & push_present & (time>=push_start) & (time<=push_end)
            episode_peak = torch.maximum(episode_peak,torch.where(alive,tilt,0))
            episode_integrated += torch.where(alive,tilt,0)*cfg.step_dt
            peak = torch.maximum(peak,torch.where(after,tilt,0))
            integrated += torch.where(after,tilt,0)*cfg.step_dt
            during_peak = torch.maximum(during_peak,torch.where(during,tilt,0))
            during_integrated += torch.where(during,tilt,0)*cfg.step_dt
            totals['return'] += reward*alive
            totals['steps'] += alive
            for name,value in [('roll',d['roll'].abs().rad2deg()),('pitch',d['pitch'].abs().rad2deg()),
                ('height_error',(d['height']-height_reference).abs()),('feet',d['feet'].all(-1)),
                ('saturation',d['saturation_steps'].sum(-1)/(12*cfg.decimation)),
                ('self',d['self_contacts']>0),('unexpected',d['reasons']['abnormal_ground'])]:
                totals[name] += torch.where(alive,torch.nan_to_num(value.float()),0)
            maxroll = torch.maximum(maxroll,torch.where(alive,d['roll'].abs().rad2deg(),0))
            maxpitch = torch.maximum(maxpitch,torch.where(alive,d['pitch'].abs().rad2deg(),0))
            peak_torque = torch.maximum(peak_torque,torch.where(alive,d['peak_torque'].max(-1).values,0))
            minheight = torch.minimum(minheight,torch.where(alive,d['height'],float('inf')))
            contact_good = d['feet'].all(-1)&((d['height']-height_reference).abs()<.005)
            good = alive & contact_good & (error.abs().max(-1).values<cfg.recovery_degrees)
            reset_good = good & (time<push_start)
            reset_stable = torch.where(reset_good,reset_stable+1,0)
            reset_done = (reset_stable>=10)&reset_recovery.isnan()
            reset_recovery[reset_done] = max(0.,time-.18)
            stable = torch.where(good & push_present & (time>=push_end),stable+1,0)
            recovered = (stable>=10)&recovery.isnan()
            recovery[recovered] = (time-push_end[recovered]-.18).clamp_min(0)
            legacy_good = alive & contact_good & (rpy.abs().rad2deg().max(-1).values<.6)
            legacy_stable = torch.where(legacy_good & push_present & (time>=push_end),legacy_stable+1,0)
            legacy_done = (legacy_stable>=10)&legacy_recovery.isnan()
            legacy_recovery[legacy_done] = (time-push_end[legacy_done]-.18).clamp_min(0)
            normalized = dict(fall=d['reasons']['low_height']|d['reasons']['tilt'],
                self_collision=d['reasons']['self_collision'],invalid_state=d['reasons']['invalid'],
                abnormal_ground_contact=d['reasons']['abnormal_ground'],timeout=extras['time_outs'])
            for name,value in normalized.items():
                reasons[name] += value&alive
            fall |= alive & d['terminated']
            alive &= ~done.bool()
        steps = totals['steps'].clamp_min(1)
        safe = (~fall)&(totals['self']==0)&(totals['unexpected']==0)&(totals['saturation']/steps<.01)
        safe &= ~reset_recovery.isnan()
        safe &= ~push_present|~recovery.isnan()
        def mean_recovery(value):
            return float(value.nanmean()) if (~value.isnan()).any() else None
        def optional_list(value):
            return [None if math.isnan(x) else x for x in value.cpu().tolist()]
        return dict(episodes=n,success_rate=float(safe.float().mean()),survival_rate=float((~fall).float().mean()),
            termination_rate=float(fall.float().mean()),mean_return=float(totals['return'].mean()),
            episode_seconds=float((steps*cfg.step_dt).mean()),peak_orientation_error=float(peak.mean()),
            worst_peak_orientation_error=float(peak.max()),integrated_orientation_error=float(integrated.mean()),
            peak_during_push=float(during_peak.mean()),integrated_during_push=float(during_integrated.mean()),
            episode_peak_orientation_error=float(episode_peak.mean()),episode_integrated_orientation_error=float(episode_integrated.mean()),
            max_roll=float(maxroll.max()),max_pitch=float(maxpitch.max()),
            mean_abs_roll=float((totals['roll']/steps).mean()),mean_abs_pitch=float((totals['pitch']/steps).mean()),
            mean_height_error=float((totals['height_error']/steps).mean()),min_height=float(minheight.min()),
            four_foot_fraction=float((totals['feet']/steps).mean()),saturation_fraction=float((totals['saturation']/steps).mean()),
            peak_torque=float(peak_torque.max()),self_contact_env_steps=int(totals['self'].sum()),
            unexpected_contact_env_steps=int(totals['unexpected'].sum()),termination_reasons={k:int(v.sum()) for k,v in reasons.items()},
            reset_recovery_rate=float((~reset_recovery.isnan()).float().mean()),reset_recovery_seconds=mean_recovery(reset_recovery),
            push_recovery_rate=float((~recovery.isnan()).float().mean()) if push_present.any() else None,
            recovery_seconds=mean_recovery(recovery) if push_present.any() else None,
            legacy_recovery_seconds=mean_recovery(legacy_recovery),seed=cfg.seed,stage=cfg.stage,v2_stage=cfg.v2_stage,
            smoothing_seconds=cfg.smoothing_seconds,reference_orientation=env.reference_standing_orientation.cpu().tolist(),
            reference_height=height_reference,force=cfg.disturbance_force,duration=cfg.disturbance_seconds,
            push_directions=directions,per_episode_return=totals['return'].cpu().tolist(),
            per_episode_peak=peak.cpu().tolist(),per_episode_integrated=integrated.cpu().tolist(),
            per_episode_recovery=optional_list(recovery))
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description='자연 평형 기준 기립 복구 평가')
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--stage',type=int,default=2)
    parser.add_argument('--v2-stage',choices=['A','B','C','D'])
    parser.add_argument('--smoothing-tau',type=float,default=.75)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--push-force',type=float)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--random',action='store_true')
    parser.add_argument('--scripted',action='store_true')
    parser.add_argument('--orientation-reward-scale',type=float,default=.05)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=StandingCfg(num_envs=args.num_envs,stage=args.stage,v2_stage=args.v2_stage,
        smoothing_seconds=args.smoothing_tau,seed=args.seed,episode_seconds=args.seconds,
        push_force=args.push_force,balanced_push_directions=True,
        orientation_reward_scale=args.orientation_reward_scale)
    result=evaluate(cfg,args.checkpoint,args.random,args.scripted)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(f'평가 결과 저장: {args.output}')

if __name__=='__main__':
    main()
