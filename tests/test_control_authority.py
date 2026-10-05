"""기립 진단의 독립 초기화, 목표 기준과 관절 조합을 검증한다."""
import unittest
import numpy as np
from rl.control_authority import AuthorityProbe, patterns


class AuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = AuthorityProbe()

    def test_equilibrium_and_independent_trials(self):
        p = self.probe
        a = p.run(seconds=.4)
        p.run(seconds=.4,action=patterns()['pitch_knee'])
        b = p.run(seconds=.4)
        self.assertEqual(a['records'],b['records'])
        self.assertEqual((p.model.nq,p.model.nv,p.model.nu),(24,23,17))
        self.assertAlmostEqual(float(p.model.body_mass.sum()),1.173)
        self.assertGreater(np.rad2deg(p.eq[1]),.39)
        self.assertLess(np.rad2deg(p.eq[1]),.41)

    def test_pattern_independence_and_directions(self):
        pattern = patterns()
        self.assertFalse(np.array_equal(pattern['hip_roll_common'],pattern['hip_roll_differential']))
        roll = self.probe.run(action=.5*pattern['roll_knee'],seconds=2.)
        pitch = self.probe.run(action=.5*pattern['pitch_knee'],seconds=2.)
        self.assertLess(roll['final_delta'][0],-.3)
        self.assertGreater(pitch['final_delta'][1],.3)
        for result in (roll,pitch):
            self.assertEqual(result['self_contact_control_steps'],0)
            self.assertEqual(result['saturation_samples'],0)

    def test_recovery_reference_and_initial_peak(self):
        r = self.probe.run(seconds=.5)
        self.assertIsNone(r['metrics']['level']['recovery_seconds']['0.1'])
        self.assertIsNotNone(r['metrics']['equilibrium']['recovery_seconds']['0.1'])
        tilted=self.probe.run(seconds=.5,disturbance=dict(kind='orientation',axis=1,value=2.))
        self.assertGreaterEqual(tilted['metrics']['equilibrium']['peak_tilt_degrees'],1.999)

if __name__ == '__main__':
    unittest.main()
