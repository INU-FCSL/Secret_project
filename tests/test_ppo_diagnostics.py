"""진단 계측이 실제 PPO 갱신을 바꾸지 않는지 검증한다."""
import copy
import unittest
import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from rl.config import StandingCfg, ppo_config
from rl.env import StandingEnv
from rl.diagnose_ppo import instrument_update, stats


class PPODiagnosticsTests(unittest.TestCase):
    def test_instrumentation_preserves_native_update(self):
        env=StandingEnv(StandingCfg(num_envs=4,episode_seconds=10.))
        try:
            runner=OnPolicyRunner(env,ppo_config(),device=env.device)
            alg=runner.alg
            obs=env.reset()
            with torch.no_grad():
                for _ in range(16):
                    actions=alg.act(obs)
                    obs,reward,done,extras=env.step(actions)
                    alg.process_env_step(obs,reward,done,extras)
                alg.compute_returns(obs)
            twin=copy.deepcopy(alg)
            native_update=twin.update
            # 학습 진입점의 wrapper에서도 원래 update를 직접 계측해야 한다.
            twin.update=lambda:instrument_update(twin,update_fn=native_update)
            torch.manual_seed(7654)
            expected=alg.update()
            torch.manual_seed(7654)
            measured=twin.update()
            for key in expected:
                self.assertEqual(expected[key],measured['loss'][key])
            for model_a,model_b in ((alg.actor,twin.actor),(alg.critic,twin.critic)):
                for a,b in zip(model_a.parameters(),model_b.parameters()):
                    self.assertTrue(torch.equal(a,b))
            self.assertEqual(alg.learning_rate,twin.learning_rate)
            self.assertEqual(len(measured['minibatches']),4)
            self.assertEqual(len(measured['actor_gradient_norm']),4)
            self.assertTrue(all(np.isfinite(x['approximate_kl']) for x in measured['minibatches']))
        finally:
            env.close()

    def test_empty_phase_and_signed_statistics(self):
        self.assertEqual(stats(np.array([])),{'count':0})
        value=stats(np.array([-2.,0.,1.,3.]))
        self.assertEqual(value['positive_fraction'],.5)
        self.assertEqual(value['negative_fraction'],.25)
        self.assertEqual(value['min'],-2.)
        self.assertEqual(value['max'],3.)


if __name__=='__main__':
    unittest.main()
