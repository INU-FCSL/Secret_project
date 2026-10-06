"""mean penalty가 native PPO gradient·clipping 순서를 보존하는지 확인한다."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rl.config import ppo_config


def make_algorithm(coefficient=None):
    torch.manual_seed(987)
    cfg=ppo_config('squashed');cfg['multi_gpu']=None;cfg['num_steps_per_env']=4
    cfg['algorithm'].update(num_learning_epochs=1,num_mini_batches=1,schedule='fixed')
    if coefficient is not None:
        cfg['algorithm'].update(class_name='rl.mean_regularization:MeanRegularizedPPO',
            mean_regularization=coefficient)
    cfg['actor']['obs_normalization']=False;cfg['critic']['obs_normalization']=False
    obs=TensorDict({'actor':torch.randn(8,42),'critic':torch.randn(8,42)},batch_size=[8])
    alg=PPO.construct_algorithm(obs,SimpleNamespace(num_envs=8,num_actions=2),cfg,'cpu')
    with torch.no_grad():
        for _ in range(4):
            alg.act(obs);alg.process_env_step(obs,torch.rand(8),torch.zeros(8),{})
        alg.compute_returns(obs)
    return alg


class MeanRegularizationTests(unittest.TestCase):
    def test_zero_coefficient_exact_native_equivalence(self):
        native,regularized=make_algorithm(),make_algorithm(0.)
        torch.manual_seed(999);first=native.update()
        torch.manual_seed(999);second=regularized.update()
        self.assertEqual(first,second)
        for key,value in native.actor.state_dict().items():
            self.assertTrue(torch.equal(value,regularized.actor.state_dict()[key]))
        for key,value in native.critic.state_dict().items():
            self.assertTrue(torch.equal(value,regularized.critic.state_dict()[key]))

    def test_penalty_gradient_added_before_native_clipping(self):
        coefficient=.001
        native,regularized=make_algorithm(),make_algorithm(coefficient)
        original_clip=torch.nn.utils.clip_grad_norm_
        original_backward=torch.Tensor.backward
        records=[]
        def backward(tensor,*args,**kwargs):
            kwargs['retain_graph']=True
            return original_backward(tensor,*args,**kwargs)
        for alg in (native,regularized):
            actor_parameters=list(alg.actor.parameters());row={}
            def clip(parameters,*args,**kwargs):
                parameters=list(parameters)
                if parameters[0] is actor_parameters[0]:
                    row['actual']=[p.grad.clone() for p in parameters]
                    penalty=coefficient*alg.actor.distribution.mean.square().mean()
                    row['expected_penalty']=torch.autograd.grad(penalty,parameters,allow_unused=True)
                return original_clip(parameters,*args,**kwargs)
            with patch.object(torch.nn.utils,'clip_grad_norm_',clip),patch.object(torch.Tensor,'backward',backward):
                torch.manual_seed(999);alg.update()
            records.append(row)
        for base,actual,penalty in zip(records[0]['actual'],records[1]['actual'],records[1]['expected_penalty']):
            torch.testing.assert_close(actual,base+(penalty if penalty is not None else 0),atol=1e-6,rtol=1e-5)


if __name__=='__main__':
    unittest.main()
