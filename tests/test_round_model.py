"""Schedule safety and target metric semantics for experimental rounds."""
from pathlib import Path
import runpy
import unittest
import numpy as np

M = runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/models/backtest_round_model.py'))


class RoundTests(unittest.TestCase):
    def test_schedule_mask(self):
        p = np.full((2,5),.2)
        out = M['schedule_mask'](p,[3,5])
        np.testing.assert_allclose(out[0],[1/3,1/3,1/3,0,0])
        np.testing.assert_array_equal(out[1],p[1])
        np.testing.assert_allclose(out.sum(axis=1),1)
        np.testing.assert_array_equal(p,np.full((2,5),.2))
        for schedules in ([2,5],[3,7]):
            with self.assertRaises(ValueError): M['schedule_mask'](p,schedules)
        with self.assertRaises(ValueError): M['schedule_mask']([[0,0,0,.5,.5]],[3])

    def test_metrics_ordinal(self):
        y = np.arange(5)
        perfect = M['evaluate'](y,np.eye(5))
        self.assertEqual(perfect['mae_rounds'],0)
        self.assertEqual(perfect['brier_score'],0)
        all_first = M['evaluate'](y,np.tile([1,0,0,0,0],(5,1)))
        self.assertEqual(all_first['mae_rounds'],2)

    def test_feature_context_and_redundancy(self):
        for features in M['groups']().values():
            self.assertEqual(len(features),len(set(features)))
            self.assertFalse(set(features)&{'ROUND','TIME','METHOD','outcome','winner','previous_early_finish_wins','previous_times_finished'})
        self.assertEqual(M['groups']()['ROUND-M'],M['groups']()['ROUND-METHOD'])


if __name__=='__main__': unittest.main()
