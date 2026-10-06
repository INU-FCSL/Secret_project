"""bounded PPO action과 그 density를 함께 정의한다."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules import GaussianDistribution


def squash(latent):
    # 정확히 ±1로 반올림되는 극단 표본만 한 ULP 안쪽으로 보호한다.
    limit = 1 - torch.finfo(latent.dtype).eps / 2
    return latent.tanh().clamp(-limit, limit)


def inverse_squash(action):
    limit = 1 - torch.finfo(action.dtype).eps / 2
    return torch.atanh(action.clamp(-limit, limit))


def log_jacobian(latent):
    return 2 * (math.log(2) - latent - F.softplus(-2 * latent))


class SquashedGaussianDistribution(GaussianDistribution):
    """global σ와 tanh 변환. storage는 a와 Normal의 (μ, σ)를 유지한다.

    log_prob은 rollout·update 모두 atanh(a)를 사용한다. KL은 공통 가역
    변환 아래 불변이므로 Normal의 analytic KL을 사용한다. 수치 guard가
    발동하는 극단 꼬리에서는 underlying continuous distribution 기준이다.
    """
    def update(self, mlp_output):
        super().update(mlp_output)
        self._entropy_estimate = None

    def sample(self):
        return squash(self._distribution.sample())

    def deterministic_output(self, mlp_output):
        return squash(mlp_output)

    def as_deterministic_output_module(self):
        return _SquashOutput()

    def log_prob(self, outputs):
        latent = inverse_squash(outputs)
        return (self._distribution.log_prob(latent) - log_jacobian(latent)).sum(-1)

    @property
    def entropy(self):
        # H(tanh u)=H(u)+E[log|da/du|]. 한 표본/관측의 pathwise MC로
        # mean·std gradient를 전달한다. 반복 조회는 같은 tensor를 반환한다.
        if self._entropy_estimate is None:
            latent = self._distribution.rsample()
            self._entropy_estimate = (
                self._distribution.entropy() + log_jacobian(latent)
            ).sum(-1)
        return self._entropy_estimate


class _SquashOutput(nn.Module):
    def forward(self, latent):
        return squash(latent)


def checkpoint_distribution(infos):
    return (infos or {}).get('policy_distribution', 'gaussian')
