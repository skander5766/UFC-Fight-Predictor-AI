"""Temporal mutation tests and independently calculated V2 expectations."""
from pathlib import Path
import runpy
import unittest

import numpy as np
import pandas as pd

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/features/build_v2_features.py'))


def fixture():
    rows = []
    for i, (date, a, b, method) in enumerate([
        ('2020-01-01', 'A', 'B', 'KO/TKO'),
        ('2020-02-01', 'A', 'C', 'Submission'),
        ('2020-02-01', 'A', 'D', 'Decision - Unanimous'),
        ('2020-03-01', 'A', 'B', 'Submission')]):
        row = dict(fight_id=str(i), URL=str(i), event_date=date, fighter_1=a, fighter_2=b,
                   winner=a, outcome='fighter_1_win', METHOD=method, ROUND=1, TIME='2:00')
        row['TIME FORMAT'] = '3 Rnd (5-5-5)'
        for side in (1, 2):
            for metric in M['HISTORY']['METRICS']:
                row[f'fighter_{side}_{metric}'] = 4.
            row[f'fighter_{side}_td_landed'] = 1.
            row[f'fighter_{side}_stats_complete'] = True
        rows.append(row)
    return pd.DataFrame(rows)


class V2Tests(unittest.TestCase):
    def build(self, frame):
        return M['build_features'](frame).set_index('fight_id').sort_index()

    def test_elo_formula_and_historical_opponents(self):
        r = self.build(fixture())
        self.assertEqual(r.loc['0', 'fighter_1_prefight_elo'], 1500)
        self.assertEqual(M['elo_delta'](1500, 1500, 1), 16)
        self.assertEqual(r.loc['1', 'fighter_1_prefight_elo'], 1516)
        self.assertEqual(r.loc['3', 'fighter_2_prefight_elo'], 1484)
        expected = 1516 + 2 * 32 * (1 - 1/(1+10**((1500-1516)/400)))
        self.assertAlmostEqual(r.loc['3', 'fighter_1_prefight_elo'], expected)
        self.assertEqual(r.loc['3', 'fighter_1_average_previous_opponent_elo'], 1500)
        self.assertEqual(r.loc['3', 'fighter_2_average_previous_opponent_elo'], 1500)

    def test_current_future_and_same_date_mutations(self):
        source = fixture()
        expected = self.build(source)
        changed = source.copy()
        changed.loc[1:, 'outcome'] = 'no_contest'
        changed.loc[1:, 'winner'] = np.nan
        changed.loc[1:, 'METHOD'] = 'Overturned'
        changed.loc[1:, 'ROUND'] = 3
        changed.loc[1:, 'TIME'] = '4:00'
        for col in changed:
            if col.startswith(('fighter_1_', 'fighter_2_')) and col.endswith(tuple(M['HISTORY']['METRICS'])):
                changed.loc[1:, col] = 99.
        actual = self.build(changed)
        pd.testing.assert_frame_equal(expected.loc[['0', '1', '2']], actual.loc[['0', '1', '2']])
        self.assertEqual(expected.loc['2', 'fighter_1_previous_submission_wins'], 0)
        self.assertEqual(expected.loc['3', 'fighter_1_previous_finish_wins'], 2)
        self.assertEqual(expected.loc['3', 'fighter_1_previous_decision_wins'], 1)

    def test_defense_cumulative_and_debut(self):
        source = fixture()
        source.loc[1, 'fighter_2_td_landed'] = 0
        source.loc[1, 'fighter_2_td_attempted'] = 2
        r = self.build(source)
        self.assertAlmostEqual(r.loc['3', 'fighter_1_takedown_defense_rate'], 1-2/10)
        self.assertEqual(r.loc['1', 'fighter_1_significant_strikes_absorbed_per_minute'], 2)
        self.assertTrue(pd.isna(r.loc['0', 'fighter_1_finish_win_rate']))
        source['fighter_2_td_landed'] = 0
        source['fighter_2_td_attempted'] = 0
        self.assertTrue(pd.isna(self.build(source).loc['3', 'fighter_1_takedown_defense_rate']))
        source['fighter_2_stats_complete'] = False
        self.assertTrue(pd.isna(self.build(source).loc['3', 'fighter_1_significant_strikes_absorbed_per_fight']))

    def test_swap_and_duplicates(self):
        source = fixture()
        expected = self.build(source)
        alias = source.iloc[[0]].copy()
        alias['fight_id'] = 'alias'
        duplicate = self.build(pd.concat([source, alias], ignore_index=True))
        pd.testing.assert_frame_equal(expected, duplicate.loc[expected.index])
        swapped = source.copy()
        swapped['fighter_1'], swapped['fighter_2'] = source.fighter_2, source.fighter_1
        swapped['outcome'] = 'fighter_2_win'
        for col in source:
            if col.startswith('fighter_1_'):
                other = col.replace('fighter_1_', 'fighter_2_')
                swapped[col], swapped[other] = source[other], source[col]
        reverse = self.build(swapped)
        diffs = [c for c in expected if c.endswith('_diff')]
        np.testing.assert_allclose(expected[diffs], -reverse[diffs], rtol=0, atol=1e-12, equal_nan=True)

    def test_unsupported_and_conflicting_duplicate(self):
        source = fixture()
        source.loc[0, 'METHOD'] = 'DQ'
        self.assertEqual(self.build(source).loc['1', 'fighter_1_prefight_elo'], 1500)
        alias = source.iloc[[0]].copy()
        alias['fight_id'] = 'alias'
        alias['METHOD'] = 'Submission'
        with self.assertRaises(ValueError):
            self.build(pd.concat([source, alias], ignore_index=True))

    def test_tied_recent_window_and_duration(self):
        entries = [dict(date=pd.Timestamp('2020-01-01'), opponent_elo=x) for x in [1400, 1600]]
        self.assertTrue(pd.isna(M['recent_opponents'](entries, 1)))
        self.assertEqual(M['recent_opponents'](entries, 3), 1500)
        row = fixture().iloc[0].to_dict()
        row.update(ROUND=2, TIME='1:30')
        self.assertEqual(M['duration'](row), 390)


if __name__ == '__main__':
    unittest.main()
