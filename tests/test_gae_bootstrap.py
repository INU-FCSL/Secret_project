"""손계산 trajectory로 failure·timeout·reset·rollout 경계를 확인한다."""
import unittest
from types import SimpleNamespace
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rl.timeout_bootstrap import TerminalBootstrapPPO


def trajectory(rewards,values,dones,last_value,gamma=.9,lam=.8):
    st=SimpleNamespace(num_transitions_per_env=len(rewards),
        rewards=torch.tensor(rewards,dtype=torch.float32).reshape(-1,1,1),
        values=torch.tensor(values,dtype=torch.float32).reshape(-1,1,1),
        dones=torch.tensor(dones,dtype=torch.float32).reshape(-1,1,1),
        returns=torch.zeros(len(rewards),1,1))
    alg=SimpleNamespace(storage=st,critic=lambda obs:torch.tensor([[last_value]]),
        gamma=gamma,lam=lam,normalize_advantage_per_mini_batch=False)
    PPO.compute_returns(alg,None)
    return st


class DummyModel:
    def update_normalization(self,obs):pass
    def reset(self,dones):pass
    def __call__(self,obs):return obs['critic']


def process(cls,timeout,terminated=False):
    alg=cls.__new__(cls);alg.actor=DummyModel();alg.critic=DummyModel()
    alg.rnd=None;alg.gamma=.9;alg.device='cpu';alg.transition=RolloutStorage.Transition()
    alg.transition.values=torch.tensor([[1.]])
    rows=[];alg.storage=SimpleNamespace(add_transition=lambda t:rows.append(t.rewards.clone()))
    extras=dict(time_outs=torch.tensor([timeout]),terminal_observation={'critic':torch.tensor([[4.]])},
        diagnostics={'terminated':torch.tensor([terminated])})
    alg.process_env_step({'critic':torch.tensor([[99.]])},torch.tensor([.2]),torch.tensor([1]),extras)
    return rows[0]


class GaeBootstrapTests(unittest.TestCase):
    def test_hand_calculated_nonterminal_rollout(self):
        st=trajectory([1.,2.],[.5,1.],[0,0],3.)
        # delta1=3.7, A1=3.7; delta0=1.4, A0=1.4+.72*3.7=4.064.
        torch.testing.assert_close(st.returns.flatten(),torch.tensor([4.564,4.7]))
        raw=st.returns-st.values
        torch.testing.assert_close(st.advantages,(raw-raw.mean())/(raw.std()+1e-8))

    def test_failure_and_self_collision_stop_at_reset(self):
        st=trajectory([.2,10.],[1.,100.],[1,0],200.)
        self.assertAlmostEqual(float(st.returns[0]),.2,places=6)
        self.assertAlmostEqual(float(st.returns[1]),190.,places=5)
        self.assertAlmostEqual(float(process(TerminalBootstrapPPO,False,True)),.2,places=6)

    def test_timeout_uses_terminal_not_current_or_reset_value(self):
        self.assertAlmostEqual(float(process(PPO,True)),1.1,places=6)
        corrected=float(process(TerminalBootstrapPPO,True))
        self.assertAlmostEqual(corrected,3.8,places=6)
        st=trajectory([corrected,10.],[1.,100.],[1,0],200.)
        self.assertAlmostEqual(float(st.returns[0]),3.8,places=5)

    def test_failure_has_priority_over_timeout(self):
        self.assertAlmostEqual(float(process(TerminalBootstrapPPO,True,True)),.2,places=6)


if __name__=='__main__':unittest.main()
