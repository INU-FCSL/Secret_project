"""density·수치 경계·gradient·action 의미를 검증한다."""
import unittest
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Independent
from rl.distributions import SquashedGaussianDistribution, squash, inverse_squash
from rl.action_mapping import bounded_action


class SquashedDistributionTests(unittest.TestCase):
    def test_many_samples_strict_bounds_and_finite_density(self):
        torch.manual_seed(97)
        for sigma in (.1,.3,1.):
            mean=torch.tensor([0.,-.5,.5,-2.,2.,-5.,5.]).repeat(20000,1)
            dist=SquashedGaussianDistribution(7,init_std=sigma)
            dist.update(mean)
            action=dist.sample()
            self.assertTrue((action.abs()<1).all())
            self.assertTrue(torch.isfinite(dist.log_prob(action)).all())

    def test_reference_transformed_density(self):
        mean=torch.tensor([[.5,-2.]],dtype=torch.float64)
        dist=SquashedGaussianDistribution(2,init_std=.3).double()
        dist.update(mean)
        action=torch.tensor([[.2,-.9999]],dtype=torch.float64)
        reference=Independent(TransformedDistribution(Normal(mean,dist.std),[TanhTransform()]),1)
        torch.testing.assert_close(dist.log_prob(action),reference.log_prob(action),atol=1e-10,rtol=1e-10)

    def test_ratio_and_analytic_kl_without_update(self):
        mean=torch.randn(4096,2)*2
        dist=SquashedGaussianDistribution(2,init_std=.3)
        dist.update(mean);action=dist.sample();old_prob=dist.log_prob(action).detach()
        old=tuple(p.clone() for p in dist.params)
        dist.update(mean.clone())
        torch.testing.assert_close((dist.log_prob(action)-old_prob).exp(),torch.ones(4096))
        torch.testing.assert_close(dist.kl_divergence(old,dist.params),torch.zeros(4096))

    def test_inverse_consistency(self):
        u=torch.linspace(-5,5,10000,dtype=torch.float64)
        torch.testing.assert_close(inverse_squash(squash(u)),u,atol=1e-11,rtol=1e-11)
        u32=torch.linspace(-3,3,10000)
        torch.testing.assert_close(inverse_squash(squash(u32)),u32,atol=1e-5,rtol=1e-5)

    def test_boundary_gradient_and_entropy_pathwise_gradient(self):
        for dtype in (torch.float32,torch.float64):
            mean=torch.tensor([[5.,-5.]],dtype=dtype,requires_grad=True)
            dist=SquashedGaussianDistribution(2,init_std=.3).to(dtype)
            dist.update(mean)
            action=torch.tensor([[1.,-1.]],dtype=dtype)
            loss=-dist.log_prob(action).mean()-.005*dist.entropy.mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(mean.grad).all())
            self.assertTrue(torch.isfinite(dist.std_param.grad).all())

    def test_entropy_estimator_matches_mc_negative_log_density(self):
        torch.manual_seed(123)
        mean=torch.tensor([[.5,-.5]]).repeat(100000,1)
        dist=SquashedGaussianDistribution(2,init_std=.3);dist.update(mean)
        estimate=dist.entropy
        self.assertIs(estimate,dist.entropy)
        mc=-dist.log_prob(dist.sample()).mean()
        self.assertLess(abs(float((estimate.mean()-mc).detach())),.015)
        self.assertLess(float(estimate.mean().detach()),float(dist._distribution.entropy().sum(-1).mean().detach()))

    def test_deterministic_and_environment_identity(self):
        dist=SquashedGaussianDistribution(2)
        mean=torch.tensor([[2.,-2.]])
        a=dist.deterministic_output(mean)
        torch.testing.assert_close(a,mean.tanh())
        self.assertIs(bounded_action(a,'identity'),a)
        with self.assertRaises(ValueError):
            bounded_action(mean,'identity')

    def test_native_storage_pre_update_and_instrumented_entropy_gradient(self):
        from tensordict import TensorDict
        from types import SimpleNamespace
        from rsl_rl.algorithms import PPO
        from rl.config import ppo_config
        from rl.normalization import pre_update_metrics
        from rl.diagnose_ppo import instrument_update
        cfg=ppo_config('squashed');cfg['num_steps_per_env']=32;cfg['multi_gpu']=None
        cfg['actor']['obs_normalization']=False
        cfg['critic']['obs_normalization']=False
        obs=TensorDict({'actor':torch.randn(8,42),'critic':torch.randn(8,42)},batch_size=[8])
        alg=PPO.construct_algorithm(obs,SimpleNamespace(num_envs=8,num_actions=2),cfg,'cpu')
        with torch.no_grad():
            for _ in range(32):
                action=alg.act(obs)
                self.assertTrue((action.abs()<1).all())
                obs=TensorDict({'actor':torch.randn(8,42),'critic':torch.randn(8,42)},batch_size=[8])
                alg.process_env_step(obs,torch.rand(8),torch.zeros(8),{})
            alg.compute_returns(obs)
        before=pre_update_metrics(alg)
        self.assertLess(abs(before['exact_kl']),1e-7)
        self.assertEqual(before['clip_fraction'],0)
        self.assertLess(abs(before['ratio_mean']-1),1e-6)
        update=instrument_update(alg)
        self.assertTrue(all(b['entropy_requires_grad'] for b in update['minibatches']))
        self.assertEqual(len(update['head_gradients']),4)


if __name__=='__main__':
    unittest.main()
