"""Standing V2의 동결 checkpoint와 학습 신호를 계측한다. 본 학습 설정은 바꾸지 않는다."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from .config import StandingCfg, ppo_config, MODEL_PATH, LEG_JOINT_NAMES
from .env import StandingEnv
from .evaluate import ScriptedController
from .reference import gravity_from_rpy, calibrate_reference
from .rewards import standing_rewards

TERMS = ('upright', 'height', 'pose', 'joint_velocity', 'action_rate', 'effort')
GROUPS = {'gravity': (0, 3), 'angular_velocity': (3, 6), 'joint_position': (6, 18),
          'joint_velocity': (18, 30), 'previous_action': (30, 42)}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def stats(value, axis=None):
    x = np.asarray(value, dtype=np.float64)
    if x.size == 0:
        return {'count': 0}
    return dict(mean=x.mean(axis=axis).tolist(), std=x.std(axis=axis).tolist(),
                min=x.min(axis=axis).tolist(), max=x.max(axis=axis).tolist(),
                positive_fraction=(x > 0).mean(axis=axis).tolist(),
                negative_fraction=(x < 0).mean(axis=axis).tolist(), count=int(x.shape[0]))


def cfg(seed=2026, num_envs=16):
    return StandingCfg(num_envs=num_envs, seed=seed, stage=2, v2_stage='A',
                       smoothing_seconds=.15, episode_seconds=10., balanced_push_directions=True)


def load_runner(env, directory, iteration):
    runner = OnPolicyRunner(env, ppo_config(), device=env.device)
    runner.load(str(directory / f'checkpoint_{iteration}.pt'),
                load_cfg={'actor': True, 'critic': True, 'optimizer': True})
    runner.alg.actor.eval()
    runner.alg.critic.eval()
    return runner


@torch.no_grad()
def trace(env, policy, actor=None):
    env._rng.manual_seed(env.cfg.seed)
    obs = env.reset()
    start = env.push_start.cpu().numpy().copy()
    push_vector = env.push_vector.cpu().numpy().copy()
    rows = []
    normalizer_before = None if actor is None else copy.deepcopy(actor.obs_normalizer.state_dict())
    for i in range(env.max_episode_length):
        raw = obs['actor'].clone()
        norm = raw if actor is None else actor.obs_normalizer(raw)
        action = policy(obs)
        obs, reward, done, extras = env.step(action)
        d = extras['diagnostics']
        rpy = torch.stack((d['roll'], d['pitch']), -1)
        error = (rpy-env.reference_standing_orientation[:2]).rad2deg()
        rows.append({k: v.detach().cpu().numpy().copy() for k, v in dict(
            raw=raw, normalized=norm, requested=action, applied=d['applied_actions'],
            target=d['targets'], actual=extras['terminal_observation']['actor'][:,6:18]+env.default_joint_position, omega=raw[:, 3:5],
            error=error, reward=reward, terms=torch.stack([d['reward_terms'][k] for k in TERMS], -1)*.02,
            done=done, failure=d['terminated'], self_contact=d['self_contacts'],
            saturation=d['saturation_steps'].sum(-1), feet=d['feet'].all(-1)).items()})
        if i < env.max_episode_length-1 and done.any():
            raise RuntimeError('조기 종료가 있어 첫 episode 분석을 다시 구성해야 합니다.')
    data = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    data['push_start'] = start
    data['push_vector'] = push_vector
    t = np.arange(1, 501)[:, None]*.02
    data['phase'] = np.where(t < start, 0, np.where(t <= start+.5, 1, 2))
    data['after'] = t >= start
    if actor is not None:
        assert all(torch.equal(v, actor.obs_normalizer.state_dict()[k]) for k, v in normalizer_before.items())
    return data


def correlation(x, y):
    return None if np.std(x) < 1e-10 or np.std(y) < 1e-10 else float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def summarize_trace(data):
    result = {}
    discount = .99**np.arange(500)[:, None]
    for name, mask in [('pre', data['phase'] == 0), ('during', data['phase'] == 1),
                       ('post', data['phase'] == 2), ('all', np.ones_like(data['phase'], dtype=bool))]:
        components = (data['terms'].astype(float)*mask[..., None]).sum(0).mean(0)
        relative_discount = np.zeros_like(discount*mask)
        for j in range(mask.shape[1]):
            indices = np.flatnonzero(mask[:,j])
            if len(indices):
                relative_discount[indices,j] = .99**(indices-indices[0])
        result[name] = dict(components=dict(zip(TERMS, components.tolist())),
            total=float((data['reward'].astype(float)*mask).sum(0).mean()),
            discounted=float((data['reward']*mask*discount).sum(0).mean()),
            discounted_from_phase_start=float((data['reward']*relative_discount).sum(0).mean()),
            fraction=float(mask.mean()))
    after = data['after']
    error = np.linalg.norm(data['error'], axis=-1)
    result['physical'] = dict(peak=float(np.where(after, error, 0).max(0).mean()),
        integrated=float((error*after*.02).sum(0).mean()),
        failure=int(data['failure'].sum()), self_contact=int(data['self_contact'].sum()),
        saturation=int(data['saturation'].sum()), four_feet=float(data['feet'].mean()))
    result['requested'] = stats(data['requested'].reshape(-1, 12), axis=0)
    result['applied'] = stats(data['applied'].reshape(-1, 12), axis=0)
    result['target_degrees'] = stats(np.rad2deg(data['target'].reshape(-1, 12)), axis=0)
    result['actual_degrees'] = stats(np.rad2deg(data['actual'].reshape(-1, 12)), axis=0)
    result['tracking_error_degrees'] = stats(np.rad2deg((data['target']-data['actual']).reshape(-1, 12)), axis=0)
    result['request_filter_rmse'] = float(np.sqrt(np.mean((data['requested']-data['applied'])**2)))
    knees = data['requested'][..., 2::3]
    differential = np.stack(((knees[..., 0]-knees[..., 1]+knees[..., 2]-knees[..., 3])/4,
                             (knees[..., 0]+knees[..., 1]-knees[..., 2]-knees[..., 3])/4), -1)
    result['feedback'] = {}
    for j, axis in enumerate(('roll', 'pitch')):
        # 초기화의 영향을 제외하고 외란 시작 이후의 관계를 측정한다.
        mask = after
        result['feedback'][axis] = dict(error_correlation=correlation(data['error'][..., j][mask], differential[..., j][mask]),
            velocity_correlation=correlation(data['omega'][..., j][mask], differential[..., j][mask]),
            differential=stats(differential[..., j][mask]))
    result['observations'] = {name: {kind: stats(data[key][..., a:b].reshape(-1, b-a), axis=0)
        for kind, key in [('raw', 'raw'), ('normalized', 'normalized')]} for name, (a,b) in GROUPS.items()}
    return result


def trajectories(directory, output):
    results = {}
    for name in ['zero', 'scripted']+[f'ppo_{i}' for i in (0,25,50,75,100)]+['ppo_0_sample', 'ppo_100_sample']:
        all_data = []
        normalizer = None
        for seed in (2026, 2027, 2028):
            env = StandingEnv(cfg(seed))
            try:
                actor = None
                if name == 'zero':
                    policy = lambda obs: torch.zeros((env.num_envs,12), device=env.device)
                elif name == 'scripted':
                    policy = ScriptedController(env)
                else:
                    iteration = int(name.split('_')[1])
                    runner = load_runner(env, directory, iteration)
                    actor = runner.alg.actor
                    stochastic = name.endswith('sample')
                    torch.manual_seed(seed+100000)
                    policy = lambda obs: actor(obs, stochastic_output=stochastic)
                    state = torch.load(directory/f'checkpoint_{iteration}.pt', weights_only=False, map_location=env.device)
                    loaded = actor.state_dict()
                    actor_key = 'actor_state_dict'
                    normalizer = dict(count=int(actor.obs_normalizer.count), mean=actor.obs_normalizer.mean.cpu().tolist(),
                        std=actor.obs_normalizer.std.cpu().tolist(), eps=actor.obs_normalizer.eps,
                        reload_exact=all(torch.equal(v, loaded[k]) for k,v in state[actor_key].items()),
                        action_std=actor.distribution.std_param.detach().cpu().tolist())
                all_data.append(trace(env, policy, actor))
            finally:
                env.close()
        data = {key: np.concatenate([d[key] for d in all_data], axis=1 if all_data[0][key].ndim >= 2 and key not in ('push_vector',) else 0) for key in all_data[0]}
        np.savez_compressed(output/f'{name}.npz', **data)
        results[name] = summarize_trace(data)
        results[name]['normalizer'] = normalizer
        save(output/'trajectories.json', results)
        print(f'경로 계측 완료: {name}', flush=True)


def sensitivity(output):
    ref = calibrate_reference()
    rows = []
    zeros = torch.zeros((1,12))
    # 다른 항목은 자연 평형에서 관측된 동일 값으로 고정한다.
    from .control_authority import AuthorityProbe
    probe = AuthorityProbe()
    offsets = torch.tensor((probe.eq_joints-probe.default)[None], dtype=torch.float32)
    torque = torch.tensor(probe.eq_torque[None,:12], dtype=torch.float32)
    for axis in (0,1):
        for degrees in (0,.05,.1,.2,.3,.5,.75,1,2):
            rpy = list(ref['orientation']);rpy[axis] += math.radians(degrees)
            total, terms = standing_rewards(torch.tensor([gravity_from_rpy(rpy)]), torch.tensor([ref['height']]),
                offsets, zeros, zeros, torque, reference_gravity=torch.tensor(gravity_from_rpy(ref['orientation'])), reference_height=ref['height'])
            rows.append(dict(axis=['roll','pitch'][axis], degrees=degrees,
                             orientation=float(terms['upright']*.02), total=float(total*.02)))
    save(output/'sensitivity.json', rows)


def instrument_update(alg):
    """기존 갱신 함수를 유지하며 입출력과 clipping 전 기울기를 관측한다."""
    original_generator = alg.storage.mini_batch_generator
    original_log_prob = alg.actor.get_output_log_prob
    original_clip = torch.nn.utils.clip_grad_norm_
    current = [None]
    batches, gradients = [], []
    def generator(*args, **kwargs):
        for batch in original_generator(*args, **kwargs):
            current[0] = batch
            yield batch
    def log_prob(actions):
        value = original_log_prob(actions)
        b = current[0]
        with torch.no_grad():
            logratio = value-b.old_actions_log_prob.squeeze(-1)
            ratio = logratio.exp()
            analytic = alg.actor.get_kl_divergence(b.old_distribution_params, alg.actor.output_distribution_params)
            batches.append(dict(ratio=stats(ratio.cpu().numpy()), approximate_kl=float(((ratio-1)-logratio).mean()),
                analytic_kl=float(analytic.mean()), clip_fraction=float(((ratio-1).abs()>alg.clip_param).float().mean()),
                entropy=float(alg.actor.output_entropy.mean())))
        return value
    def clip(parameters, *args, **kwargs):
        value = original_clip(parameters, *args, **kwargs)
        gradients.append(float(value))
        return value
    alg.storage.mini_batch_generator = generator
    alg.actor.get_output_log_prob = log_prob
    torch.nn.utils.clip_grad_norm_ = clip
    try:
        loss = alg.update()
    finally:
        alg.storage.mini_batch_generator = original_generator
        alg.actor.get_output_log_prob = original_log_prob
        torch.nn.utils.clip_grad_norm_ = original_clip
    return dict(loss=loss, minibatches=batches, actor_gradient_norm=gradients[::2],
                critic_gradient_norm=gradients[1::2], learning_rate=alg.learning_rate,
                action_std=alg.actor.distribution.std_param.detach().cpu().tolist())


def updates(directory, output):
    results = []
    # 독립 checkpoint 복사본에서 시간 창마다 한 번만 갱신한다.
    for iteration in (0,100):
        for onset in (0., 2.24, 2.88, 5.12):
            env = StandingEnv(cfg(314,64))
            try:
                runner = load_runner(env,directory,iteration);alg=runner.alg
                torch.manual_seed(9000+iteration+round(onset*100))
                env._rng.manual_seed(314);obs=env.reset()
                with torch.no_grad():
                    for _ in range(round(onset/.02)):
                        obs,_,_,_=env.step(alg.actor(obs,stochastic_output=True))
                alg.actor.train();alg.critic.train()
                # 저장된 optimizer의 실제 learning rate를 이어서 사용한다.
                alg.learning_rate=alg.optimizer.param_groups[0]['lr']
                lr_before=alg.learning_rate
                phases=[];raw_obs=[];normalized_obs=[]
                norm0=copy.deepcopy(alg.actor.obs_normalizer.state_dict())
                with torch.no_grad():
                    for _ in range(16):
                        action=alg.act(obs)
                        raw_obs.append(obs['actor'].cpu().numpy().copy())
                        normalized_obs.append(alg.actor.get_latent(obs).cpu().numpy().copy())
                        obs,reward,done,extra=env.step(action)
                        d=extra['diagnostics'];time=d['episode_steps']*.02
                        phase=torch.where(time<d['push_start'],0,torch.where(time<=d['push_start']+.5,1,torch.where(time<d['push_start']+1.,2,3)))
                        phases.append(phase.cpu().numpy())
                        alg.process_env_step(obs,reward,done,extra)
                    alg.compute_returns(obs)
                st=alg.storage
                phase=np.stack(phases)
                raw_adv=(st.returns-st.values).squeeze(-1).cpu().numpy().copy()
                normalized_adv=st.advantages.squeeze(-1).cpu().numpy().copy()
                row=dict(iteration=iteration,onset=onset,learning_rate_before=lr_before,
                    advantage={name:dict(raw=stats(raw_adv[phase==j]),normalized=stats(normalized_adv[phase==j]),fraction=float((phase==j).mean()))
                               for j,name in enumerate(('pre','during','post','stable'))},
                    advantage_all=dict(raw=stats(raw_adv.ravel()),normalized=stats(normalized_adv.ravel())),
                    explained_variance=float(1-torch.var(st.returns-st.values)/torch.var(st.returns)))
                # 갱신 전 정규화 통계 변화가 만드는 KL을 따로 측정한다.
                with torch.no_grad():
                    flat=st.observations.flatten(0,1)
                    alg.actor(flat,stochastic_output=True)
                    kl_updated=float(alg.actor.get_kl_divergence(tuple(p.flatten(0,1) for p in st.distribution_params),alg.actor.output_distribution_params).mean())
                    norm1=copy.deepcopy(alg.actor.obs_normalizer.state_dict())
                    alg.actor.obs_normalizer.load_state_dict(norm0)
                    alg.actor(flat,stochastic_output=True)
                    kl_before=float(alg.actor.get_kl_divergence(tuple(p.flatten(0,1) for p in st.distribution_params),alg.actor.output_distribution_params).mean())
                    alg.actor.obs_normalizer.load_state_dict(norm1)
                row['normalizer_kl']=dict(final_stats=kl_updated,start_stats=kl_before)
                row['observations']={name:{kind:stats(np.stack(values)[...,a:b].reshape(-1,b-a),axis=0)
                    for kind,values in [('raw',raw_obs),('normalized',normalized_obs)]} for name,(a,b) in GROUPS.items()}
                row['update']=instrument_update(alg)
                results.append(row);save(output/'updates.json',results)
                print(f'갱신 계측 완료: checkpoint={iteration}, 시작={onset}초',flush=True)
            finally:
                env.close()


def supervised(output):
    data=np.load(output/'scripted.npz')
    # 평가 seed 단위로 분리하여 다른 episode에서 검증한다.
    x=torch.tensor(data['raw'][:,:32].reshape(-1,42),device='cuda')
    y=torch.tensor(np.clip(data['requested'][:,:32],-1,1).reshape(-1,12),device='cuda')
    xt=torch.tensor(data['raw'][:,32:].reshape(-1,42),device='cuda')
    yt=torch.tensor(np.clip(data['requested'][:,32:],-1,1).reshape(-1,12),device='cuda')
    mean=x.mean(0);scale=x.std(0).clamp_min(.01)
    torch.manual_seed(1234)
    net=torch.nn.Sequential(torch.nn.Linear(42,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,12)).cuda()
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    # 외란 및 초기 복구 상태를 함께 학습한다. 안정 구간의 영 action만 맞추는 해를 피한다.
    needed=(y.abs().amax(-1)>.01).nonzero().squeeze(-1)
    for step in range(6000):
        idx=torch.cat((torch.randint(len(x),(256,),device='cuda'),needed[torch.randint(len(needed),(256,),device='cuda')]))
        pred=net((x[idx]-mean)/scale)
        loss=(pred-y[idx]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        pred=net((xt-mean)/scale)
        mse=float((pred-yt).square().mean())
        kk=(pred-yt)[:,2::3]
        differential=torch.stack(((kk[:,0]-kk[:,1]+kk[:,2]-kk[:,3])/4,(kk[:,0]+kk[:,1]-kk[:,2]-kk[:,3])/4),-1)
        result=dict(test_mse=mse,train_mse=float((net((x-mean)/scale)-y).square().mean()),
            differential_rmse=differential.square().mean(0).sqrt().cpu().tolist(),updates=6000,train_episodes=32,test_episodes=16)
    torch.save(dict(state=net.state_dict(),mean=mean,scale=scale),output/'supervised.pt')
    net.eval()
    policy=lambda obs: net((obs['actor']-mean)/scale)
    parts=[]
    for seed in (2026,2027,2028):
        env=StandingEnv(cfg(seed))
        try:
            parts.append(trace(env,policy))
        finally:
            env.close()
    merged={k:np.concatenate([d[k] for d in parts],axis=1 if parts[0][k].ndim>=2 and k!='push_vector' else 0) for k in parts[0]}
    np.savez_compressed(output/'supervised.npz',**merged)
    result['environment']=summarize_trace(merged)
    result['held_out_environment']=summarize_trace(parts[2])
    save(output/'supervised.json',result)
    print('지도 회귀와 환경 평가 완료',flush=True)


def supervised_augmented(output):
    """상관된 경로 자료의 한계를 구분하기 위해 독립 관측 변화를 추가한다."""
    data=np.load(output/'scripted.npz')
    x=torch.tensor(data['raw'][:,:32].reshape(-1,42),device='cuda')
    y=torch.tensor(np.clip(data['requested'][:,:32],-1,1).reshape(-1,12),device='cuda')
    xt=torch.tensor(data['raw'][:,32:].reshape(-1,42),device='cuda')
    yt=torch.tensor(np.clip(data['requested'][:,32:],-1,1).reshape(-1,12),device='cuda')
    env=StandingEnv(cfg())
    try:
        teacher=ScriptedController(env)
        reference=env.reference_standing_orientation[:2].clone()
    finally:
        env.close()
    mean=x.mean(0);scale=x.std(0).clamp_min(.01)
    torch.manual_seed(4321)
    net=torch.nn.Sequential(torch.nn.Linear(42,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,12)).cuda()
    optimizer=torch.optim.Adam(net.parameters(),lr=.0005)
    def synthetic(count, radius):
        raw=mean+torch.randn(count,42,device='cuda')*scale
        raw[:,6:18]=mean[6:18]+torch.randn(count,12,device='cuda')*.025
        raw[:,18:30]=torch.randn(count,12,device='cuda')*.3
        raw[:,30:42]=torch.rand(count,12,device='cuda')*2-1
        rpy=reference+(torch.rand(count,2,device='cuda')*2-1)*math.radians(radius)
        raw[:,0]=torch.sin(rpy[:,1]);raw[:,1]=-torch.sin(rpy[:,0])*torch.cos(rpy[:,1])
        raw[:,2]=-torch.cos(rpy[:,0])*torch.cos(rpy[:,1])
        raw[:,3:5]=(torch.rand(count,2,device='cuda')*2-1)*(.02 if radius<1 else .2)
        return raw
    for step in range(12000):
        idx=torch.randint(len(x),(256,),device='cuda')
        raw=torch.cat((x[idx],synthetic(384,.5),synthetic(384,2.)))
        with torch.no_grad():
            target=teacher({'actor':raw}).clamp(-1,1)
        predicted=net((raw-mean)/scale)
        loss=(predicted-target).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    net.eval()
    with torch.no_grad():
        pred=net((xt-mean)/scale)
        kk=(pred-yt)[:,2::3]
        differential=torch.stack(((kk[:,0]-kk[:,1]+kk[:,2]-kk[:,3])/4,(kk[:,0]+kk[:,1]-kk[:,2]-kk[:,3])/4),-1)
        result=dict(test_mse=float((pred-yt).square().mean()),train_mse=float((net((x-mean)/scale)-y).square().mean()),
                    differential_rmse=differential.square().mean(0).sqrt().cpu().tolist(),updates=12000,
                    synthetic_reason='관측 간 상관에 의존한 해와 표현력의 한계를 구분하는 진단',network=[42,64,64,12])
        test=synthetic(16000,.5)
        result['synthetic_mse']=float((net((test-mean)/scale)-teacher({'actor':test}).clamp(-1,1)).square().mean())
    torch.save(dict(state=net.state_dict(),mean=mean,scale=scale),output/'supervised_augmented.pt')
    policy=lambda obs:net((obs['actor']-mean)/scale)
    parts=[]
    for seed in (2026,2027,2028):
        env=StandingEnv(cfg(seed))
        try:
            parts.append(trace(env,policy))
        finally:
            env.close()
    merged={k:np.concatenate([d[k] for d in parts],axis=1 if parts[0][k].ndim>=2 and k!='push_vector' else 0) for k in parts[0]}
    np.savez_compressed(output/'supervised_augmented.npz',**merged)
    result['environment']=summarize_trace(merged)
    result['held_out_environment']=summarize_trace(parts[2])
    save(output/'supervised_augmented.json',result)
    print('독립 관측 변화를 추가한 지도 회귀 검증 완료',flush=True)


def finite_horizon(output):
    """전체 적분 상태와 필터 상태를 복원하고 같은 외란 상태에서 분기한다."""
    import mujoco
    from .control_authority import AuthorityProbe,orientation,patterns
    probe=AuthorityProbe();m,d=probe.model,probe.data
    jac=[]
    for name in ('roll_knee','pitch_knee'):
        a=probe.run(action=.5*patterns()[name],seconds=4.)
        b=probe.run(action=-.5*patterns()[name],seconds=4.)
        jac.append(np.array(a['final_delta'][:2])-np.array(b['final_delta'][:2]))
    inverse=np.linalg.inv(np.array(jac).T)
    basis=np.stack([patterns()[k] for k in ('roll_knee','pitch_knee')],axis=1)
    def script():
        err=orientation(d.qpos[3:7])[:2]-probe.eq[:2]
        coeff=np.rad2deg(-2*err-.2*d.qvel[3:5])@inverse.T
        return np.clip(np.clip(coeff,-1,1)@basis.T,-1,1)
    result=[]
    for axis in (0,1):
        for sign in (-1,1):
            mujoco.mj_setState(m,d,probe.state,probe.state_spec)
            d.xfrc_applied[:]=0
            for _ in range(150):
                d.xfrc_applied[1,axis]=sign*.5
                mujoco.mj_step(m,d)
            state=np.empty_like(probe.state);mujoco.mj_getState(m,d,state,probe.state_spec)
            initial_error=np.rad2deg(orientation(d.qpos[3:7])[:2]-probe.eq[:2]).tolist()
            for name in ('zero','scripted','wrong'):
                mujoco.mj_setState(m,d,state,probe.state_spec);mujoco.mj_forward(m,d)
                applied=np.zeros(12);total=0.;integrated=0.;values={}
                for step in range(50):
                    requested=np.zeros(12) if name=='zero' else script()*(1 if name=='scripted' else -1)
                    delta=(-math.expm1(-.02/.15))*(requested-applied);applied+=delta
                    d.ctrl[:12]=probe.default+probe.scale*applied;d.ctrl[12:]=0
                    for sub in range(10):
                        d.xfrc_applied[:]=0
                        if .3+(step*10+sub)*.002<.5:
                            d.xfrc_applied[1,axis]=sign*.5
                        mujoco.mj_step(m,d)
                    rpy=orientation(d.qpos[3:7]);err=np.rad2deg(rpy[:2]-probe.eq[:2]);integrated+=float(np.linalg.norm(err))*.02
                    r,_=standing_rewards(torch.tensor([gravity_from_rpy(rpy)]),torch.tensor([d.qpos[2]]),
                        torch.tensor((d.qpos[probe.qids]-probe.default)[None]),torch.tensor(d.qvel[probe.vids][None]),
                        torch.tensor(delta[None]),torch.tensor(d.actuator_force[None,:12]),
                        reference_gravity=torch.tensor(gravity_from_rpy(probe.eq[:3])),reference_height=float(probe.eq[3]))
                    total+=.99**step*float(r)*.02
                    if step+1 in (1,5,10,25,50):
                        values[str(step+1)]=dict(discounted_return=total,error=float(np.linalg.norm(err)),integrated=integrated)
                result.append(dict(axis=axis,sign=sign,controller=name,initial_error=initial_error,horizons=values))
    save(output/'finite_horizon.json',result)


def signals(directory, output):
    """보상 대비, 외란 밀도, 정책의 국소 피드백 부호를 계산한다."""
    zero=np.load(output/'zero.npz');script=np.load(output/'scripted.npz')
    assert np.array_equal(zero['raw'][0],script['raw'][0])
    assert np.array_equal(zero['push_start'],script['push_start'])
    error=np.abs(zero['error']).max(-1)
    after=zero['after']
    necessary=(error>.1)&after
    contrast=script['reward'].astype(float)-zero['reward'].astype(float)
    gae_contrast=np.zeros_like(contrast);running=np.zeros(48)
    for i in range(499,-1,-1):
        running=contrast[i]+.99*.95*running;gae_contrast[i]=running
    phases=zero['phase']
    component_delta=(script['terms'].astype(float)-zero['terms'].astype(float)).sum(0).mean(0)
    result=dict(component_delta=dict(zip(TERMS,component_delta.tolist())),
        matched_initial_state=True,matched_push=True,
        density=dict(push_fraction=float((phases==1).mean()),corrective_fraction=float(necessary.mean()),
                     corrective_seconds=float(necessary.sum(0).mean()*.02),
                     push_plus_recovery_fraction=(.5+.126748865)/10,
                     five_seconds=(.5+.126748865)/5,three_pushes_ten_seconds=3*(.5+.126748865)/10),
        reward_contrast={name:dict(step=stats(contrast[mask]),gae=stats(gae_contrast[mask]))
                         for name,mask in [('pre',phases==0),('during',phases==1),('post',phases==2),('all',np.ones_like(phases,dtype=bool))]},
        horizons=dict(discount_seconds=.02/(1-.99),discount_efold=-.02/math.log(.99),
            gae_seconds=.02/(1-.99*.95),gae_efold=-.02/math.log(.99*.95),
            discount_half=-.02*math.log(2)/math.log(.99),gae_half=-.02*math.log(2)/math.log(.99*.95),
            attenuation={str(t):dict(gamma=.99**(t/.02),gae=(.99*.95)**(t/.02)) for t in (.15,.32,.5,.64,1.,1.28)}),
        policies={})
    env=StandingEnv(cfg(num_envs=4))
    try:
        for iteration in (0,25,50,75,100):
            runner=load_runner(env,directory,iteration);actor=runner.alg.actor
            data=np.load(output/f'ppo_{iteration}.npz')
            raw=torch.tensor(data['raw'].reshape(-1,42),device=env.device)
            torch.manual_seed(7000+iteration)
            with torch.no_grad():
                mean=actor({'actor':raw})
                sample=actor({'actor':raw},stochastic_output=True)
                row=dict(deterministic=stats(mean.cpu().numpy(),axis=0),
                    stochastic_same_states=stats(sample.cpu().numpy(),axis=0),
                    mean_outside_ctrl_fraction=float((mean.abs()>1).float().mean()),
                    sample_outside_ctrl_fraction=float((sample.abs()>1).float().mean()),
                    std=actor.output_std[0].cpu().tolist(),entropy=float(actor.output_entropy.mean()))
                # 초기 정착 이후 관측에서 다른 입력을 고정한 유한 차분을 측정한다.
                selected=raw[torch.tensor(data['after'].ravel(),device=env.device)][::16]
                g=selected[:,:3]
                rpy=torch.stack((torch.atan2(-g[:,1],-g[:,2]),torch.asin(g[:,0].clamp(-1,1))),-1)
                derivatives={}
                def differentials(obs):
                    knees=actor({'actor':obs}).clamp(-1,1)[:,2::3]
                    return torch.stack(((knees[:,0]-knees[:,1]+knees[:,2]-knees[:,3])/4,
                                        (knees[:,0]+knees[:,1]-knees[:,2]-knees[:,3])/4),-1)
                for axis,name in enumerate(('roll','pitch')):
                    outputs=[]
                    for sign in (-1,1):
                        o=selected.clone();q=rpy.clone();q[:,axis]+=sign*math.radians(.05)
                        o[:,0]=q[:,1].sin();o[:,1]=-q[:,0].sin()*q[:,1].cos();o[:,2]=-q[:,0].cos()*q[:,1].cos()
                        outputs.append(differentials(o)[:,axis])
                    pose=(outputs[1]-outputs[0])/.1
                    outputs=[]
                    for sign in (-1,1):
                        o=selected.clone();o[:,3+axis]+=sign*.01
                        outputs.append(differentials(o)[:,axis])
                    derivatives[name]=dict(position_per_degree=stats(pose.cpu().numpy()),
                        velocity_per_rad_s=stats(((outputs[1]-outputs[0])/.02).cpu().numpy()))
                row['local_feedback']=derivatives
                row['observation_near_zero_running_std_dims']=(actor.obs_normalizer.std<1e-3).nonzero().flatten().cpu().tolist()
                row['near_zero_std_threshold']=1e-3
                row['normalizer_count']=int(actor.obs_normalizer.count)
                row['normalizer_stats_equal_actor_critic']=all(torch.equal(v,runner.alg.critic.obs_normalizer.state_dict()[k])
                   for k,v in actor.obs_normalizer.state_dict().items())
                row['settled_mean_applied']=data['applied'][250:450].mean((0,1)).tolist()
                row['settled_mean_requested']=data['requested'][250:450].mean((0,1)).tolist()
                row['settled_mean_tracking_error_deg']=np.rad2deg(data['target'][250:450]-data['actual'][250:450]).mean((0,1)).tolist()
                row['max_filter_residual_settled']=float(np.max(np.abs(data['requested'][250:450].clip(-1,1)-data['applied'][250:450])))
                result['policies'][str(iteration)]=row
    finally:
        env.close()
    reference=calibrate_reference()['orientation'][:2]
    gref=np.array(gravity_from_rpy((*reference,0.)))
    hypothetical=[]
    for scale in (.05,.02,.01):
        row={'scale':scale}
        for name,data in [('zero',zero),('scripted',script)]:
            rpy=np.deg2rad(data['error'])+reference
            r,p=rpy[...,0],rpy[...,1]
            g=np.stack((np.sin(p),-np.sin(r)*np.cos(p),-np.cos(r)*np.cos(p)),-1)
            orientation=.03*np.exp(-np.sum((g-gref)**2,axis=-1)/scale**2)*np.clip(np.sum(g*gref,axis=-1),0,1)
            reward=data['reward']-data['terms'][...,0]+orientation
            row[name]=float(reward.sum(0).mean())
        row['difference']=row['scripted']-row['zero']
        row['percentage']=100*row['difference']/row['zero']
        hypothetical.append(row)
    result['hypothetical_orientation_scales']=hypothetical
    save(output/'signals.json',result)


def normalization_control(directory, output):
    """초기 한 갱신에서 통계 고정의 영향을 비교한다. 설정과 가중치를 저장하지 않는다."""
    rows=[]
    for frozen in (False,True):
        env=StandingEnv(cfg(314,64))
        try:
            runner=load_runner(env,directory,0);alg=runner.alg
            alg.actor.train();alg.critic.train()
            if frozen:
                alg.actor.obs_normalizer.eval();alg.critic.obs_normalizer.eval()
            env._rng.manual_seed(314);obs=env.reset();torch.manual_seed(9000)
            with torch.no_grad():
                for _ in range(16):
                    action=alg.act(obs);obs,reward,done,extra=env.step(action)
                    alg.process_env_step(obs,reward,done,extra)
                alg.compute_returns(obs)
                st=alg.storage
                alg.actor(st.observations.flatten(0,1),stochastic_output=True)
                initial_kl=float(alg.actor.get_kl_divergence(tuple(p.flatten(0,1) for p in st.distribution_params),alg.actor.output_distribution_params).mean())
            torch.manual_seed(8888)
            rows.append(dict(frozen=frozen,pre_update_kl=initial_kl,normalizer_count=int(alg.actor.obs_normalizer.count),update=instrument_update(alg)))
        finally:
            env.close()
    save(output/'normalization_control.json',rows)


def main():
    parser=argparse.ArgumentParser(description='Standing V2 PPO 신호 진단')
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--part',choices=['trajectories','sensitivity','updates','supervised','supervised_augmented','finite','signals','normalization_control','all'],default='all')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/'manifest.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),checkpoint_directory=str(args.checkpoints),
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.checkpoints.glob('checkpoint_*.pt')},
        ppo_config=ppo_config(),seeds=[2026,2027,2028],leg_order=LEG_JOINT_NAMES))
    for name,fn in [('sensitivity',lambda:sensitivity(args.output)),('trajectories',lambda:trajectories(args.checkpoints,args.output)),
                    ('updates',lambda:updates(args.checkpoints,args.output)),('supervised',lambda:supervised(args.output)),
                    ('supervised_augmented',lambda:supervised_augmented(args.output)),('finite',lambda:finite_horizon(args.output)),
                    ('signals',lambda:signals(args.checkpoints,args.output)),('normalization_control',lambda:normalization_control(args.checkpoints,args.output))]:
        if args.part in (name,'all'):
            fn()


if __name__=='__main__':
    main()
