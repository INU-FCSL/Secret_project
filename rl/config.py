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
    min_height: float = .10
    max_tilt_degrees: float = 45.

    def __post_init__(self):
        if self.stage not in (0, 1, 2, 3) or self.push_seconds <= 0 or (self.push_force is not None and self.push_force < 0):
            raise ValueError('단계는 0~3, 외력은 음수가 아니어야 합니다.')
        if self.num_envs < 1 or self.decimation < 1 or self.episode_seconds <= 0:
            raise ValueError('환경 수, decimation, episode 길이는 양수여야 합니다.')
        if self.timestep != .002 or self.smoothing_seconds <= 0:
            raise ValueError('검증된 timestep과 양수 smoothing 시간이 필요합니다.')

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
