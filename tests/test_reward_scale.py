"""보상 scale 선택이 물리 상태와 기존 기본 경로를 보존하는지 검증한다."""
import copy
import unittest
import numpy as np
import torch
from rl.config import StandingCfg, REWARD_WEIGHTS
from rl.env import StandingEnv
from rl.rewards import standing_rewards
from rl.reward_validation import rescore, gravity_tensor
from rl.compare_standing import gate_decision


class RewardScaleTests(unittest.TestCase):
    def test_scale_changes_reward_and_preserves_physics(self):
        cfg=dict(num_envs=4,seed=31415,stage=2,v2_stage='A',smoothing_seconds=.15,episode_seconds=10.)
        a,b=StandingEnv(StandingCfg(**cfg)),StandingEnv(StandingCfg(**cfg,orientation_reward_scale=.01))
        try:
            action=torch.zeros(4,12,device=a.device)
            observed_difference=False
            for _ in range(20):
                oa,ra,da,ea=a.step(action);ob,rb,db,eb=b.step(action)
                self.assertTrue(torch.equal(a.qpos,b.qpos))
                self.assertTrue(torch.equal(a.qvel,b.qvel))
                self.assertTrue(torch.equal(oa['actor'],ob['actor']))
                self.assertTrue(torch.equal(da,db))
                for name in REWARD_WEIGHTS:
                    if name!='upright':
                        self.assertTrue(torch.equal(ea['diagnostics']['reward_terms'][name],eb['diagnostics']['reward_terms'][name]))
                observed_difference |= not torch.equal(ra,rb)
            self.assertTrue(observed_difference)
        finally:
            a.close();b.close()
        for scale in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                StandingCfg(orientation_reward_scale=scale)

    def test_rescoring_matches_direct_reward_without_changing_other_terms(self):
        angles=torch.tensor([[[0.,0.],[.3,0.]]],dtype=torch.float64)
        gravity=gravity_tensor(angles.deg2rad())
        zeros=torch.zeros(1,2,12,dtype=torch.float64)
        inputs=(gravity,torch.full((1,2),.1927),zeros,zeros,zeros,zeros)
        reward,terms=standing_rewards(*inputs)
        data=dict(error=angles.numpy(),terms=torch.stack(list(terms.values()),-1).numpy()*.02,
                  reward=reward.numpy()*.02,valid=np.ones((1,2),dtype=bool))
        result=rescore(data,[0.,0.,0.])
        direct,_=standing_rewards(*inputs,orientation_reward_scale=.01)
        self.assertAlmostEqual(result['0.01']['total_return'],float(direct.mean()*.02),places=10)
        self.assertAlmostEqual(result['0.05']['total_return'],float(reward.mean()*.02),places=10)
        for name in REWARD_WEIGHTS:
            if name!='upright':
                self.assertEqual(result['0.05']['components'][name],result['0.01']['components'][name])

    def test_gate_prioritizes_physical_failure_over_old_policy_improvement(self):
        base=dict(aggregate=dict(peak_orientation_error=.3,integrated_orientation_error=.2,
            survival_rate=1.,self_contact_env_steps=0,unexpected_contact_env_steps=0,saturation_fraction=0.),
            actions=dict(outside_range_fraction=.45),policy_probe=dict(local_feedback={
                'roll':{'position_per_degree':{'mean':-.1}},'pitch':{'position_per_degree':{'mean':.1}}}))
        results={'zero':copy.deepcopy(base),'v3a_25':copy.deepcopy(base),'v3b_25':copy.deepcopy(base)}
        results['v3a_25']['aggregate'].update(peak_orientation_error=.9,integrated_orientation_error=6.)
        results['v3b_25']['aggregate'].update(peak_orientation_error=.6,integrated_orientation_error=4.)
        gate=gate_decision(results)
        self.assertTrue(gate['promising'])
        self.assertTrue(gate['early_failure'])
        self.assertFalse(gate['extend_to_100'])
        results['v3b_25']['aggregate'].update(peak_orientation_error=.25,integrated_orientation_error=.15)
        results['v3b_25']['actions']['outside_range_fraction']=.05
        results['v3b_25']['policy_probe']['local_feedback']['roll']['position_per_degree']['mean']=.1
        results['v3b_25']['policy_probe']['local_feedback']['pitch']['position_per_degree']['mean']=-.1
        self.assertTrue(gate_decision(results)['extend_to_100'])


if __name__=='__main__':
    unittest.main()
