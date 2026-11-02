"""Frozen V1 probabilities. Historical demos are compatibility checks only.

The model is fitted on all available eligible history. Only the features, not
 the model fit, are restricted to a requested historical date. No model refit
 or performance evaluation occurs in this module.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BUILDER=runpy.run_path(str(Path(__file__).with_name('build_matchup_features.py')))
MODEL=ROOT/'models/ufc_v1_logistic.joblib'
METADATA=ROOT/'models/ufc_v1_metadata.json'


@lru_cache(maxsize=1)
def load_artifacts():
    metadata=json.loads(METADATA.read_text(encoding='utf-8'))
    if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Model artifact does not match metadata checksum.')
    if hashlib.sha256((ROOT/'data/processed/fights_dated.csv').read_bytes()).hexdigest()!=metadata['history_dataset_sha256']:
        raise ValueError('History snapshot changed; validate and refresh artifacts before using this engine.')
    model=joblib.load(MODEL)
    if list(model.feature_names_in_)!=metadata['features'] or metadata['features']!=list(BUILDER['FEATURES']):
        raise ValueError('Model, metadata and inference feature schemas disagree.')
    if list(model.classes_)!=[0,1]:
        raise ValueError('Unexpected target class ordering.')
    return model,metadata


def clear_prediction_cache():
    load_artifacts.cache_clear()
    BUILDER['clear_builder_cache']()


def predict_fight(fighter_a: str, fighter_b: str, prediction_date):
    day=BUILDER['calendar_date'](prediction_date)
    features=BUILDER['build_matchup_features'](fighter_a,fighter_b,day)
    model,metadata=load_artifacts()
    if features.shape!=(1,50) or list(features.columns)!=metadata['features']:
        raise ValueError('Expected exactly the ordered 50 configuration C features.')
    probabilities=model.predict_proba(features)[0]
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(),1.0,atol=1e-12):
        raise ValueError('Invalid probability output.')
    builder=BUILDER['default_builder']()
    canonical=[builder.names[builder.resolve(name)] for name in (fighter_a,fighter_b)]
    warnings=['Binary-outcome probabilities: this model does not estimate draws or no contests.',
              'Observed UFC history may be incomplete; name variants and shared names remain identity limitations.']
    for name in canonical:
        history=builder.fighter_features(name,day)
        count=history['previous_fights']
        if count<3:
            warnings.append(f'{name}: limited observed UFC history ({count} prior fights); zero prior fights means an observed debut, not a verified career debut.')
        missing=[metric for metric in BUILDER['HISTORY']['METRICS'] if history[metric+'_history_count']<count]
        if count==0:
            warnings.append(f'{name}: no historical performance evidence; missing relative differences use neutral imputation.')
        elif missing:
            warnings.append(f'{name}: missing historical statistics for some prior fights: '+', '.join(missing)+'.')
        gap=history['days_since_last_fight']
        thresholds=[n for n in (365,730,1095) if pd.notna(gap) and gap>n]
        if thresholds:
            warnings.append(f'{name}: {int(gap)} days since last observed UFC fight (exceeds '+', '.join(map(str,thresholds))+' days). The data does not establish the reason for this gap.')
    if features.isna().any(axis=None):
        warnings.append(f'{int(features.isna().sum().sum())} matchup features lack sufficient evidence and are imputed as zero relative advantage.')
    if day<=pd.Timestamp(metadata['training_end']):
        warnings.append('Historical sanity check only: the all-history model was fitted through '+metadata['training_end']+'; this is not an out-of-sample or as-of-date forecast.')
    else:
        warnings.append('Local history snapshot ends '+metadata['training_end']+'; later fights may be missing. Elapsed-time warnings refer only to observed local history.')
    return {'fighter_a':canonical[0],'fighter_b':canonical[1],'prediction_date':str(day.date()),
            'fighter_a_win_probability':float(probabilities[1]),'fighter_b_win_probability':float(probabilities[0]),
            'model':metadata['model_id'],'warnings':warnings}


def main():
    for a,b,day in [('Brock Lesnar','Mark Hunt','2016-07-09'),('Ronda Rousey','Amanda Nunes','2016-12-30'),
                    ('Khabib Nurmagomedov','Conor McGregor','2018-10-06')]:
        print(json.dumps(predict_fight(a,b,day),indent=2))


if __name__=='__main__':main()
