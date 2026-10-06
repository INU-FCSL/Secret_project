"""관측 통계를 먼저 준비하고 PPO의 수집·갱신·재로딩 동안 고정한다."""
import hashlib
import torch
from rsl_rl.modules import EmpiricalNormalization


class SemanticNormalization(EmpiricalNormalization):
    """앞 30차원만 정규화하고 controller의 12차원 필터 상태를 그대로 전달한다."""
    previous_action_normalization = 'identity'

    def forward(self, x):
        state = (x[..., :30]-self._mean[..., :30])/(self._std[..., :30]+self.eps)
        return torch.cat((state,x[..., 30:]),dim=-1)

    def inverse(self, y):
        state = y[..., :30]*(self._std[..., :30]+self.eps)+self._mean[..., :30]
        return torch.cat((state,y[..., 30:]),dim=-1)


def configure_model(model, mode='running'):
    """buffer 이름·차원을 보존해 기존 checkpoint와 의미 정규화 모드를 구분한다."""
    if mode not in ('running','identity'):
        raise ValueError('previous action 정규화는 running 또는 identity여야 합니다.')
    old=model.obs_normalizer
    current=getattr(old,'previous_action_normalization','running')
    if current==mode:
        return
    if old._mean.shape[-1]!=42:
        raise ValueError('의미 정규화에는 기존 42차원 관측이 필요합니다.')
    cls=SemanticNormalization if mode=='identity' else EmpiricalNormalization
    new=cls(42,eps=old.eps,until=old.until).to(old._mean.device)
    new.load_state_dict(old.state_dict())
    new.train(old.training)
    model.obs_normalizer=new


def configure(alg, mode='running'):
    for model in (alg.actor,alg.critic):
        configure_model(model,mode)


def checkpoint_mode(infos):
    return (infos or {}).get('previous_action_normalization',
        (infos or {}).get('normalization',{}).get('previous_action_normalization','running'))


@torch.no_grad()
def previous_input_statistics(alg, raw):
    transformed=alg.actor.obs_normalizer(raw)
    critic=alg.critic.obs_normalizer(raw)
    values=transformed[:,30:]
    identity=getattr(alg.actor.obs_normalizer,'previous_action_normalization','running')=='identity'
    if identity and (not torch.equal(values,raw[:,30:]) or not torch.equal(values,critic[:,30:]) or
                     values.abs().max()>1.000001):
        raise RuntimeError('previous action 의미 범위 또는 actor/critic 입력 일치가 깨졌습니다.')
    return dict(mean=values.double().mean(0).cpu().tolist(),std=values.double().std(0,unbiased=False).cpu().tolist(),
        min=values.min(0).values.cpu().tolist(),max=values.max(0).values.cpu().tolist(),
        identity_exact=bool(torch.equal(values,raw[:,30:])),actor_critic_equal=bool(torch.equal(values,critic[:,30:])))


def parameter_hash(alg):
    digest = hashlib.sha256()
    for model in (alg.actor, alg.critic):
        for name, value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def snapshot(alg):
    return [{key: value.detach().clone() for key, value in model.obs_normalizer.state_dict().items()}
            for model in (alg.actor, alg.critic)]


def assert_unchanged(alg, expected):
    for model, state in zip((alg.actor, alg.critic), expected):
        actual = model.obs_normalizer.state_dict()
        if not all(torch.equal(actual[key], value) for key, value in state.items()):
            raise RuntimeError('고정한 관측 정규화 통계가 변경되었습니다.')


def freeze(alg):
    # rsl_rl의 학습 모드 전환과 관계없이 update의 표본 수 제한으로 고정한다.
    for model in (alg.actor, alg.critic):
        normalizer = model.obs_normalizer
        normalizer.until = int(normalizer.count)


def restore(alg, infos):
    """기존 checkpoint 형식을 유지하며 V3-A 고정 설정을 복원한다."""
    configure(alg,checkpoint_mode(infos))
    if infos and infos.get('normalization', {}).get('mode') == 'warmup_frozen':
        freeze(alg)
        return True
    return False


def initialize(alg, raw):
    """전체 관측의 모집단 분산을 계산하고 두 normalizer에 동일하게 적용한다."""
    if raw.ndim != 2 or raw.shape[0] < 1 or not torch.isfinite(raw).all():
        raise ValueError('유한한 2차원 원시 관측 자료가 필요합니다.')
    # 작은 g_z 분산이 초기 분산 1과의 뺄셈에서 사라지지 않도록 직접 계산한다.
    precise = raw.double()
    mean = precise.mean(0, keepdim=True)
    variance = precise.var(0, unbiased=False, keepdim=True)
    with torch.no_grad():
        for model in (alg.actor, alg.critic):
            normalizer = model.obs_normalizer
            if normalizer.eps != .01:
                raise RuntimeError('검증된 정규화 eps=0.01을 유지해야 합니다.')
            normalizer._mean.copy_(mean)
            normalizer._var.copy_(variance)
            normalizer._std.copy_(variance.sqrt())
            if getattr(normalizer,'previous_action_normalization','running')=='identity':
                normalizer._mean[:,30:].zero_()
                normalizer._var[:,30:].fill_(1.)
                normalizer._std[:,30:].fill_(1.)
            normalizer.count.fill_(raw.shape[0])
    freeze(alg)
    expected = snapshot(alg)
    if not all(torch.equal(expected[0][key], expected[1][key]) for key in expected[0]):
        raise RuntimeError('actor와 critic의 관측 정규화 통계가 다릅니다.')
    return expected


@torch.no_grad()
def warmup(env, alg, steps=500, seed=12718, output=None):
    """초기 정규화와 가중치를 고정한 확률정책으로 실제 관측을 수집한다."""
    if steps < 1:
        raise ValueError('관측 수집 step은 양수여야 합니다.')
    freeze(alg)
    initial_statistics = snapshot(alg)
    weights = parameter_hash(alg)
    torch.manual_seed(seed)
    env._rng.manual_seed(seed)
    obs = env.reset()
    rows = []
    events = dict(failures=0, self_collision=0, abnormal_ground_contact=0, saturation_samples=0,
                  peak_torque=0., completed_episodes=0, push_env_steps=0)
    for _ in range(steps):
        if not torch.equal(obs['actor'], obs['critic']):
            raise RuntimeError('actor와 critic의 원시 관측이 다릅니다.')
        rows.append(obs['actor'].clone())
        actions = alg.actor(obs, stochastic_output=True)
        obs, _, done, extra = env.step(actions)
        d = extra['diagnostics']
        events['failures'] += int(d['terminated'].sum())
        events['self_collision'] += int(d['termination_reasons']['self_collision'].sum())
        events['abnormal_ground_contact'] += int(d['termination_reasons']['abnormal_ground_contact'].sum())
        events['saturation_samples'] += int(d['saturation_steps'].sum())
        events['completed_episodes'] += int(done.sum())
        events['push_env_steps'] += int(d['push_active'].sum())
        events['peak_torque'] = max(events['peak_torque'], float(d['peak_torque'].max()))
    raw = torch.stack(rows).reshape(-1, rows[0].shape[-1])
    assert_unchanged(alg, initial_statistics)
    expected = initialize(alg, raw)
    if parameter_hash(alg) != weights:
        raise RuntimeError('관측 통계 준비 중 policy weight가 변경되었습니다.')
    # 같은 관측의 반복 평가와 학습 모드 전환 후에도 통계가 고정되는지 검증한다.
    probe = raw[:env.num_envs]
    for model in (alg.actor, alg.critic):
        first = model.obs_normalizer(probe).clone()
        model.train()
        model.obs_normalizer.update(probe+1)
        if not torch.equal(first, model.obs_normalizer(probe)):
            raise RuntimeError('같은 관측의 정규화 결과가 변경되었습니다.')
    assert_unchanged(alg, expected)
    normalizer = alg.actor.obs_normalizer
    metadata = dict(mode='warmup_frozen', environments=env.num_envs, steps=steps,
                    observations=int(raw.shape[0]), seed=seed, eps=normalizer.eps,
                    mean=normalizer.mean.cpu().tolist(), variance=normalizer._var[0].cpu().tolist(),
                    std=normalizer.std.cpu().tolist(), actor_critic_equal=True,
                    collection_statistics_unchanged=True, weights_unchanged=True,
                    repeat_observation_exact=True, events=events)
    metadata.update(previous_action_normalization=getattr(normalizer,'previous_action_normalization','running'),
        normalization_mask=[True]*30+[getattr(normalizer,'previous_action_normalization','running')!='identity']*12,
        previous_action_input=previous_input_statistics(alg,raw),
        previous_action_raw=dict(mean=raw[:,30:].double().mean(0).cpu().tolist(),
            std=raw[:,30:].double().std(0,unbiased=False).cpu().tolist(),
            min=raw[:,30:].min(0).values.cpu().tolist(),max=raw[:,30:].max(0).values.cpu().tolist()))
    if output is not None:
        torch.save(raw.cpu(), output)
    return metadata, expected


@torch.no_grad()
def validate_rollout(env, alg, expected):
    """가중치 갱신 없이 실제 rollout 재평가의 KL을 검증한다."""
    weights = parameter_hash(alg)
    obs = env.get_observations()
    for _ in range(alg.storage.num_transitions_per_env):
        action = alg.act(obs)
        obs, reward, done, extra = env.step(action)
        alg.process_env_step(obs, reward, done, extra)
    alg.compute_returns(obs)
    metrics = pre_update_metrics(alg)
    assert_unchanged(alg, expected)
    if abs(metrics['exact_kl']) > 1e-7 or metrics['clip_fraction'] > 1e-7:
        raise RuntimeError('가중치 갱신 전 정책 분포가 rollout 시점과 다릅니다.')
    if parameter_hash(alg) != weights:
        raise RuntimeError('사전 검증 중 policy weight가 변경되었습니다.')
    alg.storage.clear()
    return metrics


@torch.no_grad()
def pre_update_metrics(alg):
    st = alg.storage
    actor = alg.actor
    actor.distribution.update(actor(st.observations.flatten(0, 1)))
    exact_kl = actor.get_kl_divergence(tuple(p.flatten(0, 1) for p in st.distribution_params),
                                         actor.output_distribution_params)
    logratio = actor.get_output_log_prob(st.actions.flatten(0, 1))-st.actions_log_prob.flatten().detach()
    ratio = logratio.exp()
    return dict(exact_kl=float(exact_kl.mean()), max_exact_kl=float(exact_kl.abs().max()),
                approximate_kl=float(((ratio-1)-logratio).mean()),
                clip_fraction=float(((ratio-1).abs()>alg.clip_param).float().mean()),
                ratio_mean=float(ratio.mean()), ratio_std=float(ratio.std()),
                ratio_min=float(ratio.min()), ratio_max=float(ratio.max()))
