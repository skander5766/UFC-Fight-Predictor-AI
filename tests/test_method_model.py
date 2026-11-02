"""Candidate orientation and multiclass scoring, independent of fit quality."""
from pathlib import Path
import runpy
import unittest
import numpy as np
import pandas as pd

M = runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/models/backtest_method_model.py'))


class MethodTests(unittest.TestCase):
    def test_candidate_invariance_and_opposing_candidate(self):
        frame = pd.DataFrame({'fighter_1_previous_wins':[5.,2.], 'fighter_2_previous_wins':[3.,7.]})
        x = M['candidate_features'](frame,['previous_wins'],[1,2])
        np.testing.assert_array_equal(x.to_numpy().ravel(),[2.,5.])
        swapped = frame.rename(columns={'fighter_1_previous_wins':'fighter_2_previous_wins','fighter_2_previous_wins':'fighter_1_previous_wins'})
        pd.testing.assert_frame_equal(x,M['candidate_features'](swapped,['previous_wins'],[2,1]))
        pd.testing.assert_frame_equal(-x,M['candidate_features'](frame,['previous_wins'],[2,1]))
        frame['METHOD'] = 'Submission'
        frame['outcome'] = 'fighter_2_win'
        pd.testing.assert_frame_equal(x,M['candidate_features'](frame,['previous_wins'],[1,2]))

    def test_missing_and_layoff(self):
        frame = pd.DataFrame({'fighter_1_layoff_over_365_days':[True,None], 'fighter_2_layoff_over_365_days':[False,True]})
        values = M['candidate_features'](frame,['layoff_over_365_days'],[1,2]).iloc[:,0]
        self.assertEqual(values.iloc[0],1)
        self.assertTrue(pd.isna(values.iloc[1]))
        with self.assertRaises(ValueError):
            M['candidate_features'](frame,['layoff_over_365_days'],[0,2])

    def test_metric_definition_and_groups(self):
        result = M['evaluate'](np.array([0,1,2]),np.full((3,3),1/3))
        self.assertAlmostEqual(result['brier_score'],2/3)
        self.assertAlmostEqual(result['log_loss'],np.log(3))
        for features in M['groups']().values():
            self.assertEqual(len(features),len(set(features)))
            self.assertNotIn('previous_times_finished',features)
            self.assertNotIn('previous_round_1_finish_wins',features)
            self.assertFalse(set(features)&{'METHOD','ROUND','TIME','winner','outcome'})


if __name__=='__main__':
    unittest.main()
