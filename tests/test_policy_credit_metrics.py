"""손으로 정한 MC value·advantage·Q 방향을 집계 코드와 비교한다."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rl.policy_credit import compare,PHASES


class PolicyCreditMetricTests(unittest.TestCase):
    def test_known_mc_advantage_and_local_gradient(self):
        n=10;r=16;value=np.arange(n,dtype=float);adv=np.arange(n)-4.5
        q=np.broadcast_to(value[:,None,None],(n,9,r)).copy()
        q[:,7]+=adv[:,None]
        q[:,3]+=.1;q[:,4]-=.1;q[:,5]+=.2;q[:,6]-=.2
        arrays={h:q.copy() for h in ('32','64','remainder','extended')}
        arrays.update(fd_pitch=np.full(n,.2),fd_roll=np.full(n,.2))
        gae=dict(value=value,raw_advantage=adv+100,normalized_advantage=adv,
            mu=np.zeros((n,2)),delta_mu=np.tile([.01,.02],(n,1)))
        states=[{'phase':PHASES[i//2]} for i in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            result=compare(states,gae,arrays,Path(directory))
        row=result['extended']
        self.assertEqual(row['critic']['rmse'],0.)
        self.assertAlmostEqual(row['critic']['explained_variance'],1.)
        self.assertEqual(row['advantage']['raw_advantage']['sign_accuracy'],.5)
        self.assertEqual(row['advantage']['normalized_advantage']['sign_accuracy'],1.)
        self.assertAlmostEqual(row['advantage']['raw_advantage']['spearman'],1.)
        self.assertAlmostEqual(row['gradient_alignment'],1.)
        self.assertEqual(row['gradient_sign_agreement'],1.)
        np.testing.assert_allclose(row['mean_q_gradient'],[1.,2.])


if __name__=='__main__':unittest.main()
