"""실제 MLP 입력·PPO 갱신·checkpoint에서 previous action의 의미를 검증한다."""
import tempfile
import unittest
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
from rl.config import StandingCfg, ppo_config
from rl.env import StandingEnv
from rl.normalization import configure, initialize, snapshot, assert_unchanged, restore, validate_rollout
from rl.diagnose_ppo import instrument_update


class SemanticNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=StandingEnv(StandingCfg(num_envs=4,stage=2,v2_stage='A',smoothing_seconds=.15,episode_seconds=10.))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def runner(self):
        return OnPolicyRunner(self.env,ppo_config(),device=self.env.device)

    def test_state_equality_and_actual_mlp_identity_input(self):
        legacy=self.runner();semantic=self.runner()
        configure(semantic.alg,'identity')
        raw=torch.randn(256,42,device=self.env.device)
        raw[:,30:]=raw[:,30:].clamp(-1,1)
        initialize(legacy.alg,raw);initialize(semantic.alg,raw)
        probe=raw[:5].clone()
        probe[:,30:]=torch.tensor([-1.,-.5,0.,.5,1.],device=self.env.device)[:,None]
        observation={'actor':probe,'critic':probe}
        for model,old in [(semantic.alg.actor,legacy.alg.actor),(semantic.alg.critic,legacy.alg.critic)]:
            actual=model.get_latent(observation)
            self.assertTrue(torch.equal(actual[:,:30],old.get_latent(observation)[:,:30]))
            self.assertTrue(torch.equal(actual[:,30:],probe[:,30:]))
            captured=[]
            hook=model.mlp.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
            model(observation);hook.remove()
            self.assertTrue(torch.equal(captured[0],actual))
        self.assertEqual(StandingCfg().previous_action_normalization,'running')

    def test_freeze_across_native_ppo_update(self):
        runner=self.runner();alg=runner.alg;configure(alg,'identity')
        raw=torch.randn(256,42,device=self.env.device);raw[:,30:]=raw[:,30:].clamp(-1,1)
        expected=initialize(alg,raw)
        preflight=validate_rollout(self.env,alg,expected)
        self.assertEqual(preflight['exact_kl'],0.)
        alg.train_mode();obs=self.env.reset()
        with torch.no_grad():
            for _ in range(alg.storage.num_transitions_per_env):
                action=alg.act(obs);obs,reward,done,extra=self.env.step(action)
                alg.process_env_step(obs,reward,done,extra)
            alg.compute_returns(obs)
        measured=instrument_update(alg)
        self.assertEqual(measured['minibatches'][0]['analytic_kl'],0.)
        assert_unchanged(alg,expected)

    def test_checkpoint_mask_mode_and_output_roundtrip(self):
        runner=self.runner();configure(runner.alg,'identity')
        raw=torch.randn(128,42,device=self.env.device);raw[:,30:]=raw[:,30:].clamp(-1,1)
        expected=initialize(runner.alg,raw)
        metadata=dict(previous_action_normalization='identity',normalization=dict(mode='warmup_frozen',
            previous_action_normalization='identity',normalization_mask=[True]*30+[False]*12))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            state=runner.alg.save();state.update(iter=0,infos=metadata);torch.save(state,path)
            loaded=self.runner();infos=loaded.load(str(path))
            self.assertEqual(infos['normalization']['normalization_mask'],[True]*30+[False]*12)
            self.assertTrue(restore(loaded.alg,infos))
            loaded.alg.train_mode()
            for model in (loaded.alg.actor,loaded.alg.critic):
                model.update_normalization({'actor':raw+10,'critic':raw+10})
                self.assertTrue(torch.equal(model.obs_normalizer(raw)[:,30:],raw[:,30:]))
            assert_unchanged(loaded.alg,expected)
            self.assertTrue(torch.equal(runner.alg.actor({'actor':raw}),loaded.alg.actor({'actor':raw})))
            self.assertTrue(torch.equal(runner.alg.critic({'critic':raw}),loaded.alg.critic({'critic':raw})))
        restore(runner.alg,None)
        self.assertFalse(torch.equal(runner.alg.actor.obs_normalizer(raw)[:,30:],raw[:,30:]))

    def test_legacy_checkpoints_without_identity_metadata(self):
        original=self.runner();raw=torch.randn(128,42,device=self.env.device)
        initialize(original.alg,raw)
        with tempfile.TemporaryDirectory() as directory:
            for frozen in (False,True):
                with self.subTest(frozen=frozen):
                    path=Path(directory)/f'legacy_{frozen}.pt'
                    state=original.alg.save()
                    state.update(iter=0,infos={'normalization':{'mode':'warmup_frozen'}} if frozen else {})
                    torch.save(state,path)
                    runner=self.runner();infos=runner.load(str(path));restore(runner.alg,infos)
                    normalizer=runner.alg.actor.obs_normalizer
                    expected=(raw-normalizer._mean)/(normalizer._std+normalizer.eps)
                    self.assertTrue(torch.equal(normalizer(raw),expected))


if __name__=='__main__':
    unittest.main()
