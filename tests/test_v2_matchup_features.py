"""Inference reproduction and temporal mutation checks, without fitting models."""
from pathlib import Path
import runpy
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'src/prediction/build_v2_matchup_features.py'))
FIXTURE = runpy.run_path(str(ROOT/'tests/test_v2_features.py'))['fixture']
Builder = M['V2MatchupFeatureBuilder']


class V2InferenceTests(unittest.TestCase):
    def test_historical_reproduction(self):
        result = M['validate_historical'](Builder(),160)
        print('V2 inference reproduction:', result, flush=True)
        for value in result.values():
            self.assertEqual(value['mismatches'],0)
            self.assertEqual(value['sample'],160)

    def test_current_same_date_future_exclusion(self):
        source = FIXTURE()
        before = Builder(source)
        changed = source.copy()
        changed.loc[1:,'outcome']='no_contest'
        changed.loc[1:,'winner']=np.nan
        changed.loc[1:,'METHOD']='Overturned'
        changed.loc[1:,'ROUND']=3
        changed.loc[1:,'TIME']='4:00'
        for metric in M['V2']['HISTORY']['METRICS']:
            for side in (1,2): changed.loc[1:,f'fighter_{side}_{metric}']=99.
        after = Builder(changed)
        for name in ['build_winner_features','build_method_features']:
            pd.testing.assert_frame_equal(getattr(before,name)('A','C','2020-02-01'),
                                          getattr(after,name)('A','C','2020-02-01'))
            pd.testing.assert_frame_equal(getattr(before,name)('A','C','2020-02-01'),
                                          getattr(before,name)('A','C','2020-02-01 23:59'))
        values=before.fighter_features('A','2020-02-01')
        self.assertEqual(values['prefight_elo'],1516)
        self.assertEqual(values['previous_finish_wins'],1)

    def test_debut_unknown_and_same_fighter(self):
        builder=Builder(FIXTURE())
        values=builder.fighter_features('C','2019-01-01')
        self.assertEqual(values['prefight_elo'],1500)
        self.assertEqual(values['previous_fights'],0)
        self.assertTrue(pd.isna(values['average_previous_opponent_elo']))
        self.assertTrue(pd.isna(values['finish_win_rate']))
        for name,count in [('build_winner_features',62),('build_method_features',47)]:
            fn=getattr(builder,name)
            self.assertEqual(fn('A','C','2019-01-01').shape,(1,count))
            with self.assertRaisesRegex(ValueError,'Unknown fighter'): fn('Unknown','C','2020-01-01')
            with self.assertRaises(ValueError): fn('A',' a ','2020-01-01')

    def test_duplicate_swap_and_source_orientation(self):
        source=FIXTURE()
        base=Builder(source)
        alias=source.iloc[[0]].copy(); alias['fight_id']='alias'
        duplicate=Builder(pd.concat([source,alias],ignore_index=True))
        swapped=source.copy()
        swapped['fighter_1'],swapped['fighter_2']=source.fighter_2,source.fighter_1
        swapped['outcome']='fighter_2_win'
        for col in source:
            if col.startswith('fighter_1_'):
                other=col.replace('fighter_1_','fighter_2_')
                swapped[col],swapped[other]=source[other],source[col]
        reverse_source=Builder(swapped)
        for name in ['build_winner_features','build_method_features']:
            original=getattr(base,name)('A','C','2020-04-01')
            pd.testing.assert_frame_equal(original,getattr(duplicate,name)('A','C','2020-04-01'))
            pd.testing.assert_frame_equal(original,getattr(reverse_source,name)('A','C','2020-04-01'))
            np.testing.assert_allclose(original,-getattr(base,name)('C','A','2020-04-01'),rtol=0,atol=0,equal_nan=True)


if __name__=='__main__': unittest.main()
