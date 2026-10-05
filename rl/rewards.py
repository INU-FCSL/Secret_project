"""양의 점수와 비음수 비용을 분리한 기립 보상."""
import torch
from .config import REWARD_WEIGHTS


def standing_rewards(gravity, height, position_offsets, velocities, action_delta, torque):
    height_error = torch.clamp(torch.abs(height - .1927) - .005, min=0)
    terms = {
        'upright': torch.exp(-gravity[:, :2].square().sum(-1) / .2**2)
                   * (-gravity[:, 2]).clamp(0, 1),
        'height': torch.exp(-height_error.square() / .02**2),
        'pose': torch.exp(-position_offsets.square().mean(-1) / .1**2),
        'joint_velocity': velocities.square().mean(-1),
        'action_rate': action_delta.square().mean(-1),
        'effort': (torque / .52).square().mean(-1),
    }
    weighted = {name: value * REWARD_WEIGHTS[name] for name, value in terms.items()}
    return sum(weighted.values()), weighted
