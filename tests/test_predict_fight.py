"""Prediction compatibility checks; no outcome-based model evaluation."""
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ENGINE=runpy.run_path(str(ROOT/'src/prediction/predict_fight.py'))
PREDICT=ENGINE['predict_fight']
GLOBALS=PREDICT.__globals__


class PredictionTests(unittest.TestCase):
    def test_normal_matchup_and_probability_sum(self):
        result=PREDICT('Ronda Rousey','Amanda Nunes','2016-12-30')
        self.assertEqual(result['model'],'ufc_v1_logistic')
        self.assertGreaterEqual(result['fighter_a_win_probability'],0)
        self.assertLessEqual(result['fighter_a_win_probability'],1)
        self.assertAlmostEqual(result['fighter_a_win_probability']+result['fighter_b_win_probability'],1,places=14)
        self.assertTrue(any('not an out-of-sample' in w for w in result['warnings']))

    def test_swapping_and_historical_compatibility(self):
        data=pd.read_csv(ROOT/'data/processed/prefight_features_activity.csv',low_memory=False)
        data=data[data.outcome.isin(['fighter_1_win','fighter_2_win'])].sort_values(['event_date','fight_id']).drop_duplicates('URL')
        sample=data.iloc[np.linspace(0,len(data)-1,32,dtype=int)]
        model,metadata=ENGINE['load_artifacts']()
        errors=[]
        for row in sample.itertuples():
            a=PREDICT(row.fighter_1,row.fighter_2,row.event_date)
            b=PREDICT(row.fighter_2,row.fighter_1,row.event_date)
            x=ENGINE['BUILDER']['build_matchup_features'](row.fighter_1,row.fighter_2,row.event_date)
            self.assertEqual(list(x),metadata['features'])
            self.assertEqual(x.shape,(1,50))
            self.assertAlmostEqual(a['fighter_a_win_probability'],model.predict_proba(x)[0,1],places=14)
            errors.append(abs(a['fighter_a_win_probability']-(1-b['fighter_a_win_probability'])))
        self.assertLess(max(errors),1e-12)
        print(f'Historical engine compatibility: {len(sample)} passed; swap mean={np.mean(errors):.16g}, max={max(errors):.16g}')

    def test_unknown_fighter(self):
        with self.assertRaisesRegex(ValueError,'Unknown fighter.*Suggestions only'):
            PREDICT('Ronda Rouse','Amanda Nunes','2016-12-30')

    def test_invalid_dates(self):
        for day in [None,'bad-date','2020-02-30',123,'2020-01-01T00:00Z']:
            with self.subTest(day=day), self.assertRaises(ValueError):
                PREDICT('Brock Lesnar','Mark Hunt',day)

    def test_debut_limited_history_and_missing_statistics(self):
        result=PREDICT('Brock Lesnar','Frank Mir','2008-02-02')
        self.assertTrue(any('Brock Lesnar: limited observed UFC history (0 prior fights)' in w for w in result['warnings']))
        self.assertTrue(any('no historical performance evidence' in w for w in result['warnings']))
        self.assertFalse(any(w.startswith('Brock Lesnar:') and 'days since last' in w for w in result['warnings']))
        self.assertTrue(np.isfinite(result['fighter_a_win_probability']))

    def test_long_layoff(self):
        result=PREDICT('Brock Lesnar','Mark Hunt','2016-07-09')
        warning=next(w for w in result['warnings'] if w.startswith('Brock Lesnar: 1653'))
        self.assertIn('365, 730, 1095',warning)
        self.assertIn('does not establish the reason',warning)

    def test_historical_chronology_and_current_future_mutation(self):
        helpers=runpy.run_path(str(ROOT/'tests/test_matchup_features.py'))
        source=helpers['fixture']()
        first=ENGINE['BUILDER']['MatchupFeatureBuilder'](source)
        changed=source.copy()
        for col in changed:
            if col.startswith(('fighter_1_','fighter_2_')):changed.loc[1:,col]=999.
        changed.loc[1:,'outcome']='no_contest'; changed.loc[1:,'winner']=np.nan
        second=ENGINE['BUILDER']['MatchupFeatureBuilder'](changed)
        probabilities=[]
        for builder in (first,second):
            adapter=dict(ENGINE['BUILDER'])
            adapter['build_matchup_features']=builder.build_matchup_features
            adapter['default_builder']=lambda:builder
            with patch.dict(GLOBALS,{'BUILDER':adapter}):
                probabilities.append(PREDICT('Alpha','Beta','2020-01-02')['fighter_a_win_probability'])
        self.assertEqual(probabilities[0],probabilities[1])
        self.assertEqual(first.fighter_features('Alpha','2020-01-02')['previous_fights'],1)

    def test_feature_order_rejected(self):
        model,metadata=ENGINE['load_artifacts']()
        bad=dict(metadata); bad['features']=list(reversed(metadata['features']))
        with patch.dict(GLOBALS,{'load_artifacts':lambda:(model,bad)}):
            with self.assertRaisesRegex(ValueError,'ordered 50'):
                PREDICT('Brock Lesnar','Mark Hunt','2016-07-09')

    def test_missing_history_warning_for_observed_fights(self):
        result=PREDICT('Royce Gracie','Ken Shamrock','1995-04-07')
        self.assertTrue(any('missing historical statistics' in w for w in result['warnings']))


if __name__=='__main__':unittest.main()
