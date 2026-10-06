"""검증된 물리 모델과 초기 기립 탐색 설정."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math

MODEL_PATH = Path(__file__).resolve().parents[1] / 'microdog.xml'
LEG_JOINT_NAMES = tuple(
    f'{leg}_{joint}_joint'
    for leg in ('fl', 'fr', 'rl', 'rr')
    for joint in ('hip_roll', 'hip_pitch', 'knee')
)
ACTION_SCALE = (math.radians(.25), math.radians(2), math.radians(2)) * 4
REWARD_WEIGHTS = dict(upright=1.5, height=.5, pose=.2, joint_velocity=-.02,
                      action_rate=-.01, effort=-.01)

V2_DISTURBANCES = {'A': (.5, .5), 'B': (.5, 1.), 'C': (.75, .5), 'D': (.75, 1.)}

@dataclass(frozen=True)
class StandingCfg:
    num_envs: int = 16
    device: str = 'cuda:0'
    seed: int = 42
    timestep: float = .002
    decimation: int = 10
    episode_seconds: float = 20.
    smoothing_seconds: float = .75
    stage: int = 0
    push_force: float | None = None
    push_seconds: float = .15
    v2_stage: str | None = None
    balanced_push_directions: bool = False
    reference_standing_orientation: tuple[float, float, float] | None = None
    reference_standing_height: float | None = None
    self_collision_failure: bool = True
    recovery_degrees: float = .1
    orientation_reward_scale: float = .05
    action_mapping: str = 'clip'
    previous_action_normalization: str = 'running'
    action_basis: str = 'joint'
    standing_basis: tuple[tuple[float, ...], ...] | None = None
    min_height: float = .10
    max_tilt_degrees: float = 45.

    def __post_init__(self):
        if self.action_basis not in ('joint','standing'):
            raise ValueError('action basis는 joint 또는 standing이어야 합니다.')
        if self.action_basis=='standing' and self.standing_basis is None:
            raise ValueError('standing basis의 측정된 행렬이 필요합니다.')
        if self.previous_action_normalization not in ('running','identity'):
            raise ValueError('previous action 정규화는 running 또는 identity여야 합니다.')
        if self.action_mapping not in ('clip','tanh'):
            raise ValueError('action mapping은 clip 또는 tanh여야 합니다.')
        if self.v2_stage is not None and self.v2_stage not in V2_DISTURBANCES:
            raise ValueError('V2 외란 단계는 A, B, C, D 중 하나여야 합니다.')
        if self.reference_standing_orientation is not None and (len(self.reference_standing_orientation) != 3 or not all(math.isfinite(v) for v in self.reference_standing_orientation)):
            raise ValueError('목표 자세는 유한한 rad 값 세 개여야 합니다.')
        if self.reference_standing_height is not None and (not math.isfinite(self.reference_standing_height) or self.reference_standing_height <= 0):
            raise ValueError('목표 높이는 유한한 양수여야 합니다.')
        if self.recovery_degrees <= 0:
            raise ValueError('복구 각도 문턱은 양수여야 합니다.')
        if not math.isfinite(self.orientation_reward_scale) or self.orientation_reward_scale <= 0:
            raise ValueError('자세 보상 scale은 유한한 양수여야 합니다.')
        if self.stage not in (0, 1, 2, 3) or self.push_seconds <= 0 or (self.push_force is not None and self.push_force < 0):
            raise ValueError('단계는 0~3, 외력은 음수가 아니어야 합니다.')
        if self.num_envs < 1 or self.decimation < 1 or self.episode_seconds <= 0:
            raise ValueError('환경 수, decimation, episode 길이는 양수여야 합니다.')
        if self.timestep != .002 or self.smoothing_seconds <= 0:
            raise ValueError('검증된 timestep과 양수 smoothing 시간이 필요합니다.')

    @property
    def disturbance_force(self):
        if self.push_force is not None:
            return self.push_force
        if self.v2_stage is not None:
            return V2_DISTURBANCES[self.v2_stage][0]
        return (0., 0., 1., 2.)[self.stage]

    @property
    def disturbance_seconds(self):
        return V2_DISTURBANCES[self.v2_stage][1] if self.v2_stage is not None else self.push_seconds

    @property
    def step_dt(self):
        return self.timestep * self.decimation


def ppo_config():
    from mjlab.rl.config import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
    cfg = asdict(RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(64, 64), obs_normalization=True,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .3, 'std_type': 'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(64, 64), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(num_learning_epochs=2, num_mini_batches=2,
                                     learning_rate=3e-4, entropy_coef=.005),
        num_steps_per_env=16, max_iterations=3, save_interval=1,
        experiment_name='microdog_standing', logger='tensorboard', upload_model=False,
    ))
    # rsl_rl의 MLPModel에는 CNN/RNN 전용 설정을 전달하지 않는다.
    for model in ('actor', 'critic'):
        for key in ('cnn_cfg', 'rnn_type', 'rnn_hidden_dim', 'rnn_num_layers'):
            cfg[model].pop(key, None)
    return cfg
