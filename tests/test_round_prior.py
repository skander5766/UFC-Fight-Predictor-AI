import copy
import json
from pathlib import Path
import runpy
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
API=runpy.run_path(str(ROOT/'src/prediction/round_prior.py'))
BUILD=runpy.run_path(str(ROOT/'src/models/build_round_prior.py'))


class PriorTests(unittest.TestCase):
    def test_artifact_probabilities_and_unsupported(self):
        artifact=json.loads(API['ARTIFACT'].read_text())
        API['validate_prior'](artifact)
        for schedule in (3,5):
            p=API['get_round_prior'](schedule)
            self.assertAlmostEqual(sum(p),1)
            self.assertEqual(p,artifact['schedules'][str(schedule)]['probabilities'])
        self.assertEqual(API['get_round_prior'](3)[3:],[0.,0.])
        for schedule in [None,0,2,4,7,'3','unknown',True,3.5]:
            self.assertIsNone(API['get_round_prior'](schedule))
        broken=copy.deepcopy(artifact); broken['schedules']['3']['probabilities'][4]=.1
        with self.assertRaises(ValueError): API['validate_prior'](broken)

    def test_reproduction(self):
        actual=BUILD['build_prior'](pd.read_csv(ROOT/'data/processed/fights_dated.csv',low_memory=False))
        saved=json.loads(API['ARTIFACT'].read_text())
        self.assertEqual(actual['schedules'],saved['schedules'])
        self.assertEqual(actual['eligible_finishes'],4522)

    def test_cutoff_exclusions_and_duplicates(self):
        rows=[]
        for i,(date,method,fmt,r,outcome) in enumerate([
            ('2026-05-16','KO/TKO','3 Rnd (5-5-5)',2,'fighter_1_win'),
            ('2026-05-16','Submission','5 Rnd (5-5-5-5-5)',4,'fighter_2_win'),
            ('2026-05-17','KO/TKO','3 Rnd (5-5-5)',1,'fighter_1_win'),
            ('2026-05-15','Decision - Unanimous','3 Rnd (5-5-5)',3,'fighter_1_win'),
            ('2026-05-15','KO/TKO','3 Rnd (5-5-5)',1,'no_contest'),
            ('2026-05-15','KO/TKO','No Time Limit',1,'fighter_1_win')]):
            rows.append(dict(fight_id=str(i),URL=str(i),fighter_1='A',fighter_2='B',event_date=date,METHOD=method,ROUND=r,outcome=outcome,**{'TIME FORMAT':fmt}))
        rows.append(dict(rows[0],fight_id='alias'))
        a=BUILD['build_prior'](pd.DataFrame(rows))
        self.assertEqual(a['schedules']['3']['counts'],[0,1,0,0,0])
        self.assertEqual(a['schedules']['5']['counts'],[0,0,0,1,0])
        self.assertEqual(a['excluded_other_formats'],1)


if __name__=='__main__': unittest.main()
