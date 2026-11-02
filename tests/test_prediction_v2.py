import copy
from pathlib import Path
import runpy
import unittest
import numpy as np

M=runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/prediction/predict_fight_v2.py'))


class OrchestratorTests(unittest.TestCase):
    def test_composition_and_swapping(self):
        for schedule in (3,5):
            a=M['predict_fight_v2']('Khabib Nurmagomedov','Conor McGregor','2026-05-17',schedule)
            b=M['predict_fight_v2']('Conor McGregor','Khabib Nurmagomedov','2026-05-17',schedule)
            self.assertAlmostEqual(sum(a['winner_probabilities'].values()),1)
            self.assertAlmostEqual(a['winner_probabilities']['A'],b['winner_probabilities']['B'])
            for side,other in [('A','B'),('B','A')]:
                self.assertEqual(a['method_given_win'][side],b['method_given_win'][other])
                self.assertAlmostEqual(sum(a['method_given_win'][side].values()),1)
                for method in M['METHODS']:
                    self.assertAlmostEqual(a['joint_winner_method'][side][method],a['winner_probabilities'][side]*a['method_given_win'][side][method])
            self.assertAlmostEqual(sum(sum(v.values()) for v in a['joint_winner_method'].values()),1)
            r=a['round_given_finish']['probabilities']
            self.assertAlmostEqual(sum(r.values()),1)
            if schedule==3: self.assertEqual((r['ROUND_4'],r['ROUND_5']),(0,0))
            self.assertEqual(a['round_given_finish']['type'],'empirical_population_prior')

    def test_input_safety(self):
        for s in [None,True,'3',4,3.5]:
            with self.assertRaises(ValueError): M['predict_fight_v2']('A','B','2026-01-01',s)
        for args in [('Brock Lesnar',' brock lesnar ','2026-01-01'),('Unknown Fighter','Brock Lesnar','2026-01-01'),
                     ('Brock Lesnar','Mark Hunt','bad'),('Brock Lesnar','Mark Hunt',None),
                     ('Brock Lesnar','Mark Hunt','2026-01-01T00:00Z')]:
            with self.assertRaises(ValueError): M['predict_fight_v2'](*args,3)

    def test_debut_and_historical_warnings(self):
        r=M['predict_fight_v2']('Brock Lesnar','Mark Hunt','1990-01-01',3)
        self.assertTrue(any('Historical sanity' in w for w in r['warnings']))
        self.assertTrue(any('0 prior fights' in w for w in r['warnings']))
        self.assertTrue(r['metadata']['missing_feature_counts']['winner']>0)

    def test_artifact_schema_and_probability_rejection(self):
        models,_=M['load_artifacts']()
        for kind,(model,metadata) in models.items():
            for field,value in [('version','bad'),('features',metadata['features'][::-1]),('classes',[2,1,0])]:
                altered=copy.deepcopy(metadata); altered[field]=value
                with self.assertRaises(ValueError): M['validate_model'](model,altered,M['SPECS'][kind])
        for values in [[np.nan,1],[-.1,1.1],[.2,.2]]:
            with self.assertRaises(ValueError): M['check_distribution'](values)


if __name__=='__main__': unittest.main()
