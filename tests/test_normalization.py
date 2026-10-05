"""통계 준비·고정과 기존 checkpoint 형식의 호환성을 검증한다."""
import tempfile
from pathlib import Path
import unittest
import torch
from rsl_rl.runners import OnPolicyRunner
from rl.config import StandingCfg, ppo_config
from rl.env import StandingEnv
from rl.normalization import initialize, freeze, snapshot, assert_unchanged, warmup, restore, validate_rollout, parameter_hash


class FrozenNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = StandingEnv(StandingCfg(num_envs=4, stage=2, v2_stage='A',
            seed=2718, smoothing_seconds=.15, episode_seconds=10.))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def runner(self):
        return OnPolicyRunner(self.env, ppo_config(), device=self.env.device)

    def test_population_statistics_preserve_small_variance(self):
        alg = self.runner().alg
        torch.manual_seed(10)
        raw = torch.randn(64,42,device=self.env.device)
        raw[:,2] = -1+torch.linspace(-.0001,.0001,64,device=self.env.device)
        weights = parameter_hash(alg)
        state = initialize(alg,raw)
        self.assertEqual(parameter_hash(alg),weights)
        expected = raw.double().var(0,unbiased=False).float()
        self.assertTrue(torch.equal(alg.actor.obs_normalizer._var[0],expected))
        self.assertGreater(float(alg.actor.obs_normalizer._var[0,2]),0.)
        self.assertEqual(int(alg.actor.obs_normalizer.count),64)
        alg.train_mode()
        alg.actor.update_normalization({'actor':raw+100})
        alg.critic.update_normalization({'critic':raw-100})
        assert_unchanged(alg,state)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(state[0].values(),state[1].values())))

    def test_warmup_and_actual_rollout_have_no_weight_update(self):
        alg = self.runner().alg
        metadata, state = warmup(self.env,alg,steps=160,seed=12345)
        self.assertEqual(metadata['observations'],640)
        self.assertTrue(metadata['weights_unchanged'])
        self.assertTrue(metadata['collection_statistics_unchanged'])
        self.assertTrue(metadata['repeat_observation_exact'])
        self.assertGreater(metadata['events']['push_env_steps'],0)
        self.env._rng.manual_seed(2718);self.env.reset()
        weights = parameter_hash(alg)
        metrics = validate_rollout(self.env,alg,state)
        self.assertLess(abs(metrics['exact_kl']),1e-7)
        self.assertEqual(metrics['clip_fraction'],0.)
        self.assertEqual(parameter_hash(alg),weights)
        self.assertEqual(alg.storage.step,0)

    def test_checkpoint_statistics_and_fixed_mode_restore(self):
        runner = self.runner()
        raw = torch.randn(128,42,device=self.env.device)
        state = initialize(runner.alg,raw)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            saved=runner.alg.save()
            saved.update(iter=0,infos={'normalization':{'mode':'warmup_frozen'}})
            torch.save(saved,path)
            loaded=self.runner()
            infos=loaded.load(str(path))
            self.assertTrue(restore(loaded.alg,infos))
            loaded.alg.train_mode()
            loaded.alg.actor.update_normalization({'actor':raw+10})
            loaded.alg.critic.update_normalization({'critic':raw-10})
            assert_unchanged(loaded.alg,state)
            self.assertTrue(torch.equal(runner.alg.actor({'actor':raw}),loaded.alg.actor({'actor':raw})))
        self.assertFalse(restore(runner.alg,None))
        self.assertFalse(restore(runner.alg,{'completed_updates':100}))


if __name__=='__main__':
    unittest.main()
