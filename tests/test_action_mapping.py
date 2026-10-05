"""환경 action 변환과 raw Gaussian PPO 저장 경로를 검증한다."""
import copy
import unittest
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rl.action_mapping import bounded_action,equivalent_raw
from rl.config import StandingCfg,ppo_config
from rl.env import StandingEnv
from rl.normalization import initialize,validate_rollout
from rl.diagnose_ppo import instrument_update
from rl.ablation import classify


class ActionMappingTests(unittest.TestCase):
    def test_mapping_range_linearity_and_equivalent_script(self):
        raw=torch.tensor([0.,.1,-.1,.5,-.5,1.,-1.,2.,-2.,5.,-5.])
        bounded=bounded_action(raw,'tanh')
        self.assertTrue(torch.equal(bounded,raw.tanh()))
        self.assertTrue(torch.isfinite(bounded).all())
        self.assertTrue((bounded.abs()<=1).all())
        self.assertLess(abs(float(bounded[1]-raw[1])),.00034)
        self.assertTrue(torch.equal(bounded_action(raw),raw.clamp(-1,1)))
        self.assertTrue(torch.allclose(bounded_action(equivalent_raw(raw,'tanh'),'tanh'),raw.clamp(-1,1),atol=1.1e-6))
        with self.assertRaises(ValueError):
            StandingCfg(action_mapping='invalid')

    def test_filter_observation_and_safe_targets(self):
        env=StandingEnv(StandingCfg(num_envs=4,smoothing_seconds=.15,action_mapping='tanh'))
        try:
            raw=torch.full((4,12),2.,device=env.device)
            obs,_,_,extras=env.step(raw)
            d=extras['diagnostics'];expected=raw.tanh()*(1-np.exp(-.02/.15))
            self.assertTrue(torch.equal(d['raw_actions'],raw))
            self.assertTrue(torch.equal(d['bounded_actions'],raw.tanh()))
            self.assertTrue(torch.allclose(d['applied_actions'],expected))
            self.assertTrue(torch.equal(obs['actor'][:,30:42],d['applied_actions']))
            self.assertTrue((d['targets']<=env.default_joint_position+env.action_scale+1e-7).all())
            self.assertTrue((d['targets']>=env.default_joint_position-env.action_scale-1e-7).all())
            self.assertTrue((d['targets']<=env.target_high).all())
            self.assertTrue((d['targets']>=env.target_low).all())
        finally:
            env.close()

    def test_raw_gaussian_storage_and_rollout32_native_update(self):
        env=StandingEnv(StandingCfg(num_envs=4,action_mapping='tanh',episode_seconds=10.))
        try:
            cfg=ppo_config();cfg['num_steps_per_env']=32
            alg=OnPolicyRunner(env,cfg,device=env.device).alg
            state=initialize(alg,torch.randn(128,42,device=env.device))
            before=validate_rollout(env,alg,state)
            self.assertEqual(before['exact_kl'],0.)
            self.assertEqual(alg.storage.actions.shape,(32,4,12))
            obs=env.reset()
            with torch.no_grad():
                for step in range(32):
                    raw=alg.act(obs);saved=raw.clone()
                    obs,reward,done,extras=env.step(raw)
                    alg.process_env_step(obs,reward,done,extras)
                    self.assertTrue(torch.equal(alg.storage.actions[step],saved))
                    self.assertTrue(torch.equal(extras['diagnostics']['bounded_actions'],saved.tanh()))
                alg.compute_returns(obs)
            self.assertTrue(torch.isfinite(alg.storage.returns).all())
            self.assertTrue(torch.isfinite(alg.storage.advantages).all())
            measured=instrument_update(alg)
            self.assertEqual(len(measured['minibatches']),4)
            self.assertEqual(measured['minibatches'][0]['analytic_kl'],0.)
            self.assertTrue(all(np.isfinite(v) for v in measured['loss'].values()))
        finally:
            env.close()

    def test_partial_and_success_require_physics_and_safety(self):
        parent=dict(aggregate=dict(peak_orientation_error=1.,integrated_orientation_error=5.,
            survival_rate=1.,saturation_fraction=0.,push_recovery_rate=0.,mean_return=20.),
            actions=dict(boundary_applied_fraction=.5,bounded_99_fraction=.5,feedback={
                'applied_roll':{'error_correlation':-1.},'applied_pitch':{'error_correlation':1.}}),
            per_seed=[dict(seed=i,peak_orientation_error=1.,integrated_orientation_error=5.,survival_rate=1.,
                self_contact_env_steps=0,unexpected_contact_env_steps=0,saturation_fraction=0.) for i in (2026,2027,2028)])
        zero=copy.deepcopy(parent)
        for row in zero['per_seed']:
            row.update(peak_orientation_error=.3,integrated_orientation_error=.2)
        candidate=copy.deepcopy(parent)
        candidate['aggregate'].update(peak_orientation_error=.8,integrated_orientation_error=4.)
        candidate['actions']['boundary_applied_fraction']=.2
        self.assertEqual(classify(candidate,parent,zero)['status'],'PARTIAL')
        candidate['actions']['boundary_applied_fraction']=.5
        self.assertEqual(classify(candidate,parent,zero)['status'],'FAIL')
        for row in candidate['per_seed']:
            row.update(peak_orientation_error=.2,integrated_orientation_error=.1)
        self.assertEqual(classify(candidate,parent,zero)['status'],'SUCCESS')
        candidate['per_seed'][0]['self_contact_env_steps']=1
        self.assertEqual(classify(candidate,parent,zero)['status'],'FAIL')


if __name__=='__main__':
    unittest.main()
