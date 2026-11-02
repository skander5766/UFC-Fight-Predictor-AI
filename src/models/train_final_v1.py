"""Fit frozen configuration C on all eligible history. No in-sample evaluation."""
import hashlib
import json
from pathlib import Path
import runpy
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/"models/ufc_v1_logistic.joblib"
METADATA=ROOT/"models/ufc_v1_metadata.json"


def main():
    base=runpy.run_path(str(Path(__file__).with_name('train_baseline.py')))
    enriched=runpy.run_path(str(Path(__file__).with_name('train_enriched_symmetric_logistic.py')))
    inference=runpy.run_path(str(ROOT/'src/prediction/build_matchup_features.py'))
    protected=[p for folder in (ROOT/'data',ROOT/'models') for p in folder.rglob('*') if p.is_file() and p not in (MODEL,METADATA)]
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    original=pd.read_csv(base['INPUT'],low_memory=False)
    data=pd.read_csv(enriched['INPUT'],low_memory=False)
    pd.testing.assert_frame_equal(data[original.columns],original)
    selected,filtering=base['load_binary_fights']()
    frame=data.set_index('fight_id').loc[selected.fight_id].copy().reset_index()
    features=list(inference['FEATURES'])
    earlier=json.loads(enriched['REPORT'].read_text())
    backtest=json.loads((ROOT/'models/logistic_backtest_metrics.json').read_text())
    assert features==earlier['experiments']['C']['features']==backtest['features']['C']
    full=enriched['design'](data,base['FEATURES'])[features]
    duplicate=data.URL.duplicated(keep=False)
    if full.loc[duplicate].groupby(data.loc[duplicate,'URL']).nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting duplicate feature rows.')
    x=enriched['design'](frame,base['FEATURES'])[features]
    swapped=enriched['design'](frame,base['FEATURES'],swap=True)[features]
    np.testing.assert_allclose(swapped,-x,rtol=1e-10,atol=1e-10,equal_nan=True)
    y=selected.target.reset_index(drop=True)
    xa=pd.concat([x,-x],ignore_index=True)
    ya=pd.concat([y,1-y],ignore_index=True)
    assert len(xa)==2*len(x) and ya.value_counts().to_dict()=={0:len(x),1:len(x)}
    pipeline=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                       ('scaler',StandardScaler(with_mean=False)),
                       ('logistic',LogisticRegression(max_iter=2000))])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipeline.fit(xa,ya)
    filled=pipeline.named_steps['imputer'].transform(xa)
    np.testing.assert_allclose(pipeline.named_steps['scaler'].var_,filled.var(axis=0),atol=1e-12)
    assert np.all(pipeline.named_steps['scaler'].n_samples_seen_==len(xa))
    # Algebraic symmetry checks only; no training accuracy or other fit metrics.
    z=pipeline[:-1].transform(x)
    zr=pipeline[:-1].transform(-x)
    np.testing.assert_allclose(z,-zr,rtol=0,atol=0)
    intercept=float(pipeline.named_steps['logistic'].intercept_[0])
    assert abs(intercept)<1e-5
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    joblib.dump(pipeline,MODEL)
    loaded=joblib.load(MODEL)
    np.testing.assert_array_equal(loaded.named_steps['logistic'].coef_,pipeline.named_steps['logistic'].coef_)
    metadata={'model_id':'ufc_v1_logistic','configuration':'C','model_type':'LogisticRegression',
              'sklearn_version':sklearn.__version__,'features':features,'feature_count':len(features),
              'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              'training_dataset':'data/processed/prefight_features_activity.csv',
              'training_dataset_sha256':hashes[enriched['INPUT']],
              'history_dataset_sha256':hashes[ROOT/'data/processed/fights_dated.csv'],
              'training_start':str(selected.event_date.min().date()),'training_end':str(selected.event_date.max().date()),
              'original_training_fights':len(x),'augmented_training_rows':len(xa),'filtering':filtering,
              'target_definition':{'1':'fighter_a/fighter_1 wins','0':'fighter_b/fighter_2 wins',
                                   'population':'binary outcomes only; draws and no contests excluded'},
              'preprocessing':'SimpleImputer(constant, fill_value=0, keep_empty_features=True); StandardScaler(with_mean=False)',
              'missing_semantics':'Zero means unknown relative advantage, not zero fighter performance.',
              'logistic_parameters':pipeline.named_steps['logistic'].get_params(),
              'symmetry_strategy':'Paired training (X,y),(-X,1-y); odd preprocessing; trainable intercept near zero.',
              'intercept':intercept,
              'evaluation':{'source':'models/logistic_backtest_metrics.json','configuration_C':backtest['aggregate']['C'],
                            'prior_single_period':earlier['experiments']['C']['metrics'],
                            'note':'Temporal backtests are evidence for the frozen configuration, not an out-of-sample evaluation of this all-history fit. The stronger single 2023-2026 evaluation was repeatedly inspected and is not a pristine holdout.'},
              'training_metrics_note':'Final-model training metrics are NOT out-of-sample performance; none were computed or reported.',
              'limitations':['Incomplete dated observed UFC history; undated fights omitted.',
                             'Name variants can split history and shared names can conflate identity.',
                             'Historical source ordering bias and duplicate source aliases; duplicates counted once.',
                             'Probabilities concern binary outcomes, not draw/no-contest probability.',
                             'Historical inference with this all-history model is not an as-of-date forecast: model fitting includes later outcomes.',
                             'No age, physical attributes, stance, current career records, medical or retirement assumptions.',
                             'All previously examined evaluation periods are exploratory, not untouched final holdouts.']}
    METADATA.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({k:metadata[k] for k in ['original_training_fights','augmented_training_rows','training_start','training_end','feature_count','intercept']})
    print('No training performance evaluated. Retained configuration C temporal-backtest documentation.')
    print('Saved:',MODEL,METADATA)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
