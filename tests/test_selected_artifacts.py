"""Artifact schemas and inference compatibility; no training-score evaluation."""
import hashlib
import json
from pathlib import Path
import runpy
import unittest
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'src/prediction/build_v2_matchup_features.py'))


class SelectedArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder=M['V2MatchupFeatureBuilder']()
        cls.models={}
        for kind,stem in [('winner','ufc_winner_d_v2_0_0'),('method','ufc_method_m_v1_0_0')]:
            path=ROOT/'models'/f'{stem}.joblib'
            cls.models[kind]=(joblib.load(path),json.loads((ROOT/'models'/f'{stem}_metadata.json').read_text()))
            assert hashlib.sha256(path.read_bytes()).hexdigest()==cls.models[kind][1]['model_sha256']

    def test_schema_preprocessing_and_counts(self):
        for kind,(model,meta) in self.models.items():
            expected=list(M['WINNER_FEATURES'] if kind=='winner' else M['METHOD_FEATURES'])
            self.assertEqual(meta['features'],expected)
            self.assertEqual(list(model.feature_names_in_),expected)
            self.assertEqual(model.named_steps['imputer'].strategy,'constant')
            self.assertEqual(model.named_steps['imputer'].fill_value,0)
            self.assertFalse(model.named_steps['scaler'].with_mean)
            self.assertEqual(meta['original_training_fights'],8547 if kind=='winner' else 8524)
            self.assertEqual(meta['fitted_training_rows'],17094 if kind=='winner' else 8524)
            self.assertTrue(np.all(model.named_steps['scaler'].n_samples_seen_==meta['fitted_training_rows']))
            np.testing.assert_array_equal(model.classes_,[0,1] if kind=='winner' else [0,1,2])

    def test_inference_probabilities_symmetry_and_debut(self):
        for a,b,date in [('Brock Lesnar','Mark Hunt','2016-07-09'),
                         ('Ronda Rousey','Amanda Nunes','2016-12-30'),
                         ('Khabib Nurmagomedov','Conor McGregor','1990-01-01')]:
            for kind,(model,meta) in self.models.items():
                fn=self.builder.build_winner_features if kind=='winner' else self.builder.build_method_features
                x=fn(a,b,date)
                before=joblib.hash(model)
                p=model.predict_proba(x)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p>=0)&(p<=1)).all())
                np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
                if kind=='winner':
                    q=model.predict_proba(fn(b,a,date))
                    np.testing.assert_allclose(p[:,1],1-q[:,1],atol=1e-5,rtol=0)
                else:
                    np.testing.assert_array_equal(p,model.predict_proba(fn(' '+a.upper()+' ',b,date)))
                self.assertEqual(before,joblib.hash(model))
                with self.assertRaises(ValueError): model.predict_proba(x.iloc[:,::-1])


if __name__=='__main__': unittest.main()
