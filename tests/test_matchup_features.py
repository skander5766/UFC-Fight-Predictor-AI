"""Run: .venv/Scripts/python.exe -B -m unittest discover -s tests -p test_matchup_features.py -v"""
from pathlib import Path
import runpy
import unittest

import numpy as np
import pandas as pd

MODULE=runpy.run_path(str(Path(__file__).resolve().parents[1]/"src/prediction/build_matchup_features.py"))
Builder=MODULE["MatchupFeatureBuilder"]


def fixture():
    rows=[]
    for i,(day,a,b) in enumerate([("2020-01-01","Alpha","Beta"),("2020-01-02","Alpha","Gamma"),
                                 ("2020-01-02","Alpha","Delta"),("2020-01-03","Alpha","Epsilon")]):
        row=dict(fight_id=str(i),URL='url'+str(i),event_date=day,fighter_1=a,fighter_2=b,
                 winner=a,outcome='fighter_1_win')
        for side in (1,2):
            for metric in MODULE['HISTORY']['METRICS']:row[f'fighter_{side}_{metric}']=float(i+1)
        rows.append(row)
    return pd.DataFrame(rows)


class MatchupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=fixture()
        cls.builder=Builder(cls.source)

    def test_chronology_same_date_and_future_mutation(self):
        before=self.builder.build_matchup_features('Alpha','Beta','2020-01-02')
        self.assertEqual(self.builder.fighter_features('Alpha','2020-01-02')['previous_fights'],1)
        self.assertEqual(self.builder.fighter_features('Alpha','2020-01-03')['previous_fights'],3)
        changed=self.source.copy()
        for col in changed:
            if col.startswith(('fighter_1_','fighter_2_')):changed.loc[1:,col]=999.
        changed.loc[1:,'outcome']='no_contest'; changed.loc[1:,'winner']=np.nan
        after=Builder(changed).build_matchup_features('Alpha','Beta','2020-01-02')
        pd.testing.assert_frame_equal(before,after)
        pd.testing.assert_frame_equal(before,self.builder.build_matchup_features('Alpha','Beta','2020-01-02 23:59'))

    def test_swap_and_unseen_matchup(self):
        a=self.builder.build_matchup_features('Beta','Gamma','2020-01-04')
        b=self.builder.build_matchup_features('Gamma','Beta','2020-01-04')
        np.testing.assert_allclose(a,-b,rtol=0,atol=0,equal_nan=True)
        self.assertEqual(a.shape,(1,50))
        self.assertEqual(list(a),list(MODULE['FEATURES']))

    def test_debut(self):
        v=self.builder.fighter_features('Epsilon','2020-01-01')
        self.assertEqual(v['previous_fights'],0)
        self.assertEqual(v['fights_last_365_days'],0)
        self.assertTrue(v['is_ufc_debut'])
        for name in ['sig_str_landed_per_fight','days_since_last_fight','layoff_over_365_days']:
            self.assertTrue(pd.isna(v[name]))

    def test_unknown_name_suggestions(self):
        with self.assertRaisesRegex(ValueError,'Suggestions only.*Alpha'):
            self.builder.build_matchup_features('Alph','Beta','2020-01-03')

    def test_normalization_same_fighter_and_invalid_date(self):
        pd.testing.assert_frame_equal(self.builder.build_matchup_features(' alpha ','BETA','2020-01-03'),
                                      self.builder.build_matchup_features('Alpha','Beta','2020-01-03'))
        with self.assertRaises(ValueError):self.builder.build_matchup_features('Alpha',' alpha ','2020-01-03')
        for day in [None,123,'not-a-date','2020-01-01T00:00Z']:
            with self.assertRaises(ValueError):self.builder.build_matchup_features('Alpha','Beta',day)

    def test_duplicate_sources_count_once(self):
        alias=self.source.iloc[[0]].copy(); alias['fight_id']='alias'
        duplicated=Builder(pd.concat([self.source,alias],ignore_index=True))
        self.assertEqual(duplicated.fighter_features('Alpha','2020-01-04')['previous_fights'],4)

    def test_activity_boundaries(self):
        current=pd.Timestamp('2022-01-01')
        activity=MODULE['ACTIVITY']['calculate_activity']([current-pd.Timedelta(days=730),current-pd.Timedelta(days=365)],current)
        self.assertEqual(activity['fights_last_365_days'],1)
        self.assertEqual(activity['fights_last_730_days'],2)
        self.assertFalse(activity['layoff_over_365_days'])

    def test_historical_reproduction(self):
        result=MODULE['validate_historical'](MODULE['default_builder'](),160)
        self.assertEqual(result['sample_fights'],160)
        self.assertEqual(result['mismatches'],[])
        self.assertEqual(result['exact_rows']+result['tolerance_only_rows'],160)
        self.assertEqual(result['maximum_swap_error'],0)


if __name__=='__main__':unittest.main()
