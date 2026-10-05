"""기립 환경의 물리 보존, 입출력, 종료와 짧은 GPU 동작 검증."""
import unittest
import numpy as np
import mujoco
import torch
from rl.config import StandingCfg, ppo_config
from rl.env import StandingEnv
from rl.rewards import standing_rewards

class StandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = StandingEnv(StandingCfg(num_envs=4, episode_seconds=20))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_mapping_observation_and_reset(self):
        e = self.env
        o = e.get_observations()
        self.assertEqual(tuple(o['actor'].shape), (4, 42))
        self.assertEqual(e.num_actions, 12)
        self.assertTrue(torch.allclose(o['actor'][:, :3], torch.tensor([0., 0., -1.], device=e.device)))
        self.assertTrue(torch.allclose(o['actor'][:, 6:18], torch.zeros((4, 12), device=e.device)))
        self.assertAlmostEqual(float(e.mj_model.body_mass.sum()), 1.173)
        self.assertEqual((e.mj_model.nq, e.mj_model.nv, e.mj_model.nu), (24, 23, 17))
        e.step(torch.ones((4, 12), device=e.device))
        e.reset([0])
        self.assertTrue(torch.equal(e.qpos[0], e.initial_qpos))
        self.assertEqual(int(e.episode_length_buf[1]), 1)
        self.assertFalse(torch.equal(e.previous_actions[1], torch.zeros(12, device=e.device)))

    def test_action_safety_and_validation(self):
        e = self.env
        for _ in range(20):
            obs, reward, done, info = e.step(torch.full((4, 12), 100., device=e.device))
            self.assertTrue(torch.all(e.joint_targets <= e.default_joint_position + e.action_scale + 1e-7))
            self.assertTrue(torch.all(e.joint_targets >= e.target_low))
            self.assertTrue(torch.all(e.joint_targets <= e.target_high))
            self.assertTrue(torch.isfinite(reward).all())
            self.assertTrue(torch.equal(e.ctrl[:, 12:], torch.zeros((4, 5), device=e.device)))
        with self.assertRaises(ValueError): e.step(torch.zeros((4, 17), device=e.device))
        with self.assertRaises(ValueError): e.step(torch.full((4, 12), float('nan'), device=e.device))

    def test_termination_and_timeout(self):
        e = self.env
        e.qpos[0, 2] = .06
        e.qpos[1, 3:7] = torch.tensor([.70710678, .70710678, 0, 0], device=e.device)
        e.qvel[2, 0] = float('nan')
        e.episode_length_buf[3] = e.max_episode_length - 1
        obs, reward, done, info = e.step(torch.zeros((4, 12), device=e.device))
        self.assertTrue(done.bool().all())
        self.assertTrue(info['diagnostics']['reasons']['invalid'][2])
        self.assertTrue(info['time_outs'][3])
        self.assertFalse(info['time_outs'][:3].any())
        self.assertTrue(torch.isfinite(obs['actor']).all())
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue((e.episode_length_buf == 0).all())

    def test_zero_action_ten_seconds(self):
        e = self.env
        peak=0.
        for _ in range(500):
            obs, reward, done, info = e.step(torch.zeros((4, 12), device=e.device))
            d = info['diagnostics']
            self.assertFalse(done.any())
            self.assertFalse(d['self_contacts'].any())
            self.assertFalse(d['saturation_steps'].any())
            self.assertTrue(torch.isfinite(reward).all())
            peak=max(peak,float(d['peak_torque'].max()))
        self.assertTrue(d['feet'].all())
        self.assertLess(abs(float(d['height'].mean()) - .19272), .002)
        self.assertLess(float(torch.acos(-d['gravity'][:, 2].clamp(-1,1)).max()), .035)
        self.assertLess(peak,.15)

    def test_random_actions_ten_seconds(self):
        e=self.env
        torch.manual_seed(43)
        for _ in range(500):
            obs,reward,done,info=e.step(torch.rand((4,12),device=e.device)*2-1)
            d=info['diagnostics']
            self.assertFalse(done.any())
            self.assertFalse(d['self_contacts'].any())
            self.assertFalse(d['saturation_steps'].any())
            self.assertTrue(torch.isfinite(obs['actor']).all())
            self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(d['feet'].all())

    def test_reward_signs(self):
        z=torch.zeros((4,12),device=self.env.device)
        g=torch.tensor([[0.,0.,-1.]]*4,device=self.env.device)
        h=torch.full((4,),.1927,device=self.env.device)
        baseline,_=standing_rewards(g,h,z,z,z,z)
        worse,terms=standing_rewards(g,h,z,torch.ones_like(z),torch.ones_like(z),torch.ones_like(z)*.4)
        self.assertTrue((worse < baseline).all())
        for term in ['joint_velocity','action_rate','effort']:
            self.assertTrue((terms[term] < 0).all())
        _, upside = standing_rewards(-g,h,z,z,z,z)
        self.assertTrue((upside['upright'] == 0).all())

    def test_ppo_configuration(self):
        from rsl_rl.runners import OnPolicyRunner
        runner=OnPolicyRunner(self.env,ppo_config(),device=self.env.device)
        obs=self.env.get_observations()
        with torch.inference_mode():
            action=runner.alg.act(obs)
        self.assertEqual(tuple(action.shape),(4,12))
        self.assertTrue(torch.isfinite(action).all())

if __name__=='__main__': unittest.main()
