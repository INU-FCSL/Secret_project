"""양의 점수와 비음수 비용을 분리한 기립 보상."""
import torch
from .config import REWARD_WEIGHTS


def standing_rewards(gravity, height, position_offsets, velocities, action_delta, torque,
                     reference_gravity=None, reference_height=.1927, orientation_reward_scale=.05,
                     pose_reward_weight=None):
    if reference_gravity is None:
        reference_gravity = gravity.new_tensor([0., 0., -1.])
    height_error = torch.clamp(torch.abs(height - reference_height) - .005, min=0)
    terms = {
        'upright': torch.exp(-(gravity-reference_gravity).square().sum(-1) / orientation_reward_scale**2)
                   * (gravity*reference_gravity).sum(-1).clamp(0, 1),
        'height': torch.exp(-height_error.square() / .02**2),
        'pose': torch.exp(-position_offsets.square().mean(-1) / .1**2),
        'joint_velocity': velocities.square().mean(-1),
        'action_rate': action_delta.square().mean(-1),
        'effort': (torque / .52).square().mean(-1),
    }
    weighted = {name: value * REWARD_WEIGHTS[name] for name, value in terms.items()}
    if pose_reward_weight is not None:
        weighted['pose']=terms['pose']*pose_reward_weight
    return sum(weighted.values()), weighted
