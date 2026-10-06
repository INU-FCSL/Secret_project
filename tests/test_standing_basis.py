"""대칭 모드와 reduced basis의 안전 범위·필터 상태·PPO 저장을 검증한다."""
import unittest
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rl.config import StandingCfg, ppo_config
from rl.env import StandingEnv
from rl.action_diagnostics import mode_decomposition
from rl.normalization import configure,initialize,validate_rollout,assert_unchanged
from rl.diagnose_ppo import instrument_update


class StandingBasisTests(unittest.TestCase):
    def test_orthogonal_symmetry_modes_preserve_energy(self):
        common=np.ones((20,12))
        front=np.tile(np.array([1,1,-1,-1])[:,None],(1,3)).ravel()[None]
        for value,name in [(common,'common'),(front,'front_rear')]:
            r=mode_decomposition(value)
            self.assertTrue(r['energy_preserved'])
            self.assertAlmostEqual(r['modes'][name]['energy_fraction'],1.)
        random=np.random.default_rng(42).normal(size=(100,12))
        self.assertTrue(mode_decomposition(random)['energy_preserved'])

    def test_two_dimension_targets_observation_and_native_update(self):
        matrix=np.zeros((12,2))
        matrix[2::3,0]=.55*np.array([1,1,-1,-1])
        matrix[2::3,1]=.45*np.array([1,-1,1,-1])
        self.assertLessEqual(np.abs(matrix).sum(-1).max(),1.000001)
        env=StandingEnv(StandingCfg(num_envs=4,episode_seconds=10.,smoothing_seconds=.15,
            previous_action_normalization='identity',action_basis='standing',standing_basis=matrix.tolist()))
        try:
            self.assertEqual(env.num_actions,2);self.assertEqual(env.num_obs,42)
            raw=torch.tensor([[5.,5.],[5.,-5.],[-5.,5.],[-5.,-5.]],device=env.device)
            obs,_,_,extra=env.step(raw);d=extra['diagnostics']
            expected=raw.clamp(-1,1)@env.basis_matrix.T
            self.assertTrue(torch.allclose(d['bounded_actions'],expected))
            self.assertLessEqual(float(d['bounded_actions'].abs().max()),1.000001)
            self.assertTrue(torch.equal(obs['actor'][:,30:],d['applied_actions']))
            self.assertTrue((d['targets']<=env.target_high).all() and (d['targets']>=env.target_low).all())
            cfg=ppo_config();cfg['num_steps_per_env']=32
            alg=OnPolicyRunner(env,cfg,device=env.device).alg;configure(alg,'identity')
            values=torch.randn(128,42,device=env.device);values[:,30:]=values[:,30:].clamp(-1,1)
            expected_stats=initialize(alg,values)
            self.assertEqual(validate_rollout(env,alg,expected_stats)['exact_kl'],0.)
            self.assertEqual(alg.storage.actions.shape,(32,4,2))
            obs=env.reset()
            with torch.no_grad():
                for i in range(32):
                    action=alg.act(obs);saved=action.clone()
                    obs,reward,done,extra=env.step(action);alg.process_env_step(obs,reward,done,extra)
                    self.assertTrue(torch.equal(alg.storage.actions[i],saved))
                alg.compute_returns(obs)
            self.assertTrue(torch.isfinite(alg.storage.returns).all())
            result=instrument_update(alg)
            self.assertEqual(len(result['minibatches']),4)
            self.assertEqual(result['minibatches'][0]['analytic_kl'],0.)
            assert_unchanged(alg,expected_stats)
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
