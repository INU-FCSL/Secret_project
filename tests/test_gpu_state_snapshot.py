"""GPU physical·filter·외란·시간 상태를 복원한 분기 재현을 확인한다."""
import unittest
import torch
import numpy as np
from rl.config import StandingCfg
from rl.env import StandingEnv
from rl.credit_audit import capture,restore_state


class GpuStateSnapshotTests(unittest.TestCase):
    def test_same_full_state_replays_after_other_actions(self):
        basis=np.zeros((12,2));basis[2::3,0]=.55*np.array([1,1,-1,-1]);basis[2::3,1]=.45*np.array([1,-1,1,-1])
        env=StandingEnv(StandingCfg(num_envs=4,action_basis='standing',standing_basis=basis.tolist(),
            action_mapping='identity',smoothing_seconds=.15,stage=2,v2_stage='A'))
        try:
            env.push_start.zero_();env.push_vector[:,0]=.5
            for _ in range(5):env.step(torch.full((4,2),.25,device='cuda'))
            state=capture(env,0)
            def replay():
                restore_state(env,state);rows=[]
                for _ in range(5):
                    _,reward,done,extra=env.step(torch.zeros((4,2),device='cuda'))
                    rows.append((env.qpos.clone(),env.qvel.clone(),env.previous_actions.clone(),reward.clone()))
                return rows
            first=replay()
            for _ in range(10):env.step(torch.full((4,2),-.75,device='cuda'))
            second=replay()
            for a,b in zip(first,second):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        finally:env.close()


if __name__=='__main__':unittest.main()
