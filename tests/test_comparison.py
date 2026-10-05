"""평가 계측이 기존 지표와 첫 episode의 집계를 보존하는지 검증한다."""
import json
import unittest
from rl.config import StandingCfg
from rl.evaluate import evaluate
from rl.compare_standing import ActionTrace, action_summary


class StandingComparisonTests(unittest.TestCase):
    def test_callback_preserves_evaluation_and_timeout_trace(self):
        cfg=StandingCfg(num_envs=4,episode_seconds=1.,seed=2026)
        baseline=evaluate(cfg)
        collector=ActionTrace()
        measured=evaluate(cfg,step_callback=collector)
        self.assertEqual(measured['survival_rate'],baseline['survival_rate'])
        self.assertAlmostEqual(measured['mean_return'],baseline['mean_return'],places=5)
        self.assertAlmostEqual(measured['integrated_orientation_error'],baseline['integrated_orientation_error'],places=4)
        data=collector.arrays()
        self.assertEqual(data['requested'].shape,(50,4,12))
        self.assertEqual(int(data['valid'].sum()),200)
        self.assertEqual(measured['termination_reasons']['timeout'],4)
        self.assertTrue((data['actual'][-1,:,2]>data['target'][-1,:,2]).all())
        summary=action_summary(data)
        self.assertEqual(summary['actual_clipping_fraction'],0.)
        self.assertIsNone(summary['feedback']['requested_roll']['error_correlation'])
        json.dumps(summary,allow_nan=False)


if __name__=='__main__':
    unittest.main()
