"""pose 한 항의 교체와 동률을 포함한 순위 상관을 확인한다."""
import unittest
import numpy as np
import torch
from rl.config import StandingCfg
from rl.rewards import standing_rewards
from rl.reward_alignment import ranks,correlation


class RewardAlignmentTests(unittest.TestCase):
    def test_pose_weight_changes_only_pose_component(self):
        args=(torch.tensor([[.002,.003,-.99999]]),torch.tensor([.1927]),
            torch.full((1,12),.01),torch.full((1,12),.02),torch.full((1,12),.03),torch.full((1,12),.1))
        old,terms=standing_rewards(*args)
        explicit,legacy=standing_rewards(*args,pose_reward_weight=.2)
        torch.testing.assert_close(old,explicit,rtol=0,atol=0)
        new,reweighted=standing_rewards(*args,pose_reward_weight=.1)
        for key in terms:
            torch.testing.assert_close(reweighted[key],terms[key]*(.5 if key=='pose' else 1),rtol=0,atol=0)
        torch.testing.assert_close(new-old,-.5*terms['pose'])
        self.assertEqual(StandingCfg().pose_reward_weight,.2)
        with self.assertRaises(ValueError):StandingCfg(pose_reward_weight=float('nan'))
        with self.assertRaises(ValueError):StandingCfg(pose_reward_weight=-.1)

    def test_spearman_uses_average_ranks_for_ties(self):
        np.testing.assert_equal(ranks([4,1,1,2]),[4,1.5,1.5,3])
        self.assertAlmostEqual(correlation([1,2,3],[9,3,1],True),-1.)
        self.assertAlmostEqual(correlation([1,1,2],[7,7,8],True),1.)
        self.assertIsNone(correlation([1,1],[1,2],True))


if __name__=='__main__':unittest.main()
