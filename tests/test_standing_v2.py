"""자연 평형 목표, 자기접촉 종료와 V2 평가의 회귀 검증."""
import math
import tempfile
from pathlib import Path
import unittest
import torch
from rsl_rl.runners import OnPolicyRunner
from rl.config import StandingCfg, ppo_config
from rl.env import StandingEnv
from rl.evaluate import evaluate
from rl.reference import calibrate_reference, gravity_from_rpy
from rl.rewards import standing_rewards


class StandingV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = StandingEnv(StandingCfg(num_envs=4,seed=123,stage=2,v2_stage='A',
            balanced_push_directions=True,smoothing_seconds=.15))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_reference_reward_and_explicit_level_goal(self):
        ref = calibrate_reference()
        gravity = torch.tensor([gravity_from_rpy(ref['orientation']),[0.,0.,-1.]],device=self.env.device)
        zeros = torch.zeros((2,12),device=self.env.device)
        height = torch.full((2,),ref['height'],device=self.env.device)
        _,terms = standing_rewards(gravity,height,zeros,zeros,zeros,zeros,
            reference_gravity=gravity[0],reference_height=ref['height'])
        self.assertGreater(float(terms['upright'][0]),float(terms['upright'][1]))
        self.assertAlmostEqual(float(terms['upright'][0]),1.5,places=5)
        self.assertTrue(torch.equal(terms['height'],torch.full((2,),.5,device=self.env.device)))
        env = StandingEnv(StandingCfg(num_envs=1,reference_standing_orientation=(0.,0.,0.),
            reference_standing_height=.19))
        try:
            self.assertTrue(torch.equal(env.reference_gravity,torch.tensor([0.,0.,-1.],device=env.device)))
            self.assertEqual(env.reference_standing_height,.19)
        finally:
            env.close()

    def test_self_collision_failure_reaches_ppo(self):
        e=self.env
        runner=OnPolicyRunner(e,ppo_config(),device=e.device)
        with torch.inference_mode():
            action=runner.alg.act(e.get_observations())
            e.qpos[0,e.qpos_ids[0]]=math.radians(-4.)
            obs,reward,done,extras=e.step(action)
            self.assertTrue(done[0])
            self.assertTrue(extras['termination_reasons']['self_collision'][0])
            self.assertFalse(extras['time_outs'][0])
            self.assertEqual(int(e.episode_length_buf[0]),0)
            runner.alg.process_env_step(obs,reward,done,extras)
            self.assertTrue(runner.alg.storage.dones[0,0])

    def test_termination_reason_and_normal_foot_contact(self):
        e=self.env
        e.qpos[0,2]=.06
        e.qvel[1,0]=float('nan')
        e.episode_length_buf[2]=e.max_episode_length-1
        _,_,done,extras=e.step(torch.zeros((4,12),device=e.device))
        self.assertTrue(done[:3].bool().all())
        self.assertTrue(extras['termination_reasons']['fall'][0])
        self.assertTrue(extras['termination_reasons']['invalid_state'][1])
        self.assertTrue(extras['termination_reasons']['timeout'][2])
        self.assertFalse(extras['termination_reasons']['self_collision'][3])

    def test_filter_config_and_actual_response(self):
        e=self.env
        e.reset()
        requested=torch.ones((4,12),device=e.device)
        _,_,_,extras=e.step(requested)
        expected=1-math.exp(-.02/.15)
        self.assertTrue(torch.allclose(extras['diagnostics']['applied_actions'],torch.full_like(requested,expected)))
        for tau in [.75,.3,.15,.05]:
            self.assertEqual(StandingCfg(smoothing_seconds=tau).smoothing_seconds,tau)
        with self.assertRaises(ValueError):
            StandingCfg(smoothing_seconds=0.)

    def test_combined_reset_and_balanced_push(self):
        e=self.env
        self.assertTrue(e.qvel[:,e.qvel_ids].abs().max()<=.010001)
        self.assertTrue(torch.equal(e.push_vector,torch.tensor([[.5,0,0],[-.5,0,0],[0,.5,0],[0,-.5,0]],device=e.device)))
        e.push_start[:]=0
        _,_,_,extras=e.step(torch.zeros((4,12),device=e.device))
        self.assertTrue(extras['diagnostics']['push_active'].all())
        self.assertTrue(torch.allclose(e.external_force[:,e.base_id,:3],e.push_vector))
        self.assertEqual(e.cfg.disturbance_seconds,.5)

    def test_evaluation_metrics_and_checkpoint_reload(self):
        cfg=StandingCfg(num_envs=4,stage=0,episode_seconds=4.,balanced_push_directions=True)
        result=evaluate(cfg)
        self.assertEqual(result['survival_rate'],1.)
        self.assertEqual(result['self_contact_env_steps'],0)
        self.assertGreaterEqual(result['integrated_orientation_error'],0.)
        self.assertLess(result['mean_height_error'],.001)
        self.assertAlmostEqual(result['reference_height'],self.env.reference_standing_height)
        e=self.env
        runner=OnPolicyRunner(e,ppo_config(),device=e.device)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            state=runner.alg.save();state.update(iter=0,infos={});torch.save(state,path)
            result=evaluate(StandingCfg(num_envs=4,episode_seconds=.5),checkpoint=path)
            self.assertEqual(result['episodes'],4)
            self.assertTrue(math.isfinite(result['mean_return']))

if __name__=='__main__':
    unittest.main()
