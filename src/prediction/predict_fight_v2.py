"""Compose versioned winner, conditional method, and empirical round artifacts.

Separate from the existing prediction engine. Cached snapshots require an
explicit cache clear after artifact updates. Historical requests restrict
features, not the all-history fitted models or fixed-cutoff round prior.
"""
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path
import runpy
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BUILD=runpy.run_path(str(Path(__file__).with_name('build_v2_matchup_features.py')))
PRIOR=runpy.run_path(str(Path(__file__).with_name('round_prior.py')))
SPECS={'winner':('ufc_winner_d_v2_0_0','2.0.0','D',BUILD['WINNER_FEATURES'],[0,1]),
       'method':('ufc_method_m_v1_0_0','1.0.0','METHOD-M',BUILD['METHOD_FEATURES'],[0,1,2])}
METHODS=['KO_TKO','SUBMISSION','DECISION']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_model(model,meta,spec):
    stem,version,configuration,features,classes=spec
    if (meta['model_id'],meta['version'],meta['configuration'])!=(stem,version,configuration):
        raise ValueError('Unexpected artifact identity/version/configuration.')
    if meta['features']!=list(features) or list(model.feature_names_in_)!=list(features) or meta['feature_count']!=len(features):
        raise ValueError('Artifact feature order does not match inference schema.')
    if meta['classes']!=classes or list(model.classes_)!=classes:
        raise ValueError('Unexpected artifact class order.')
    labels={'0':'fighter_b_win','1':'fighter_a_win'} if configuration=='D' else {str(i):v for i,v in enumerate(METHODS)}
    if meta['class_labels']!=labels:
        raise ValueError('Unexpected artifact class semantics.')


@lru_cache(maxsize=1)
def load_artifacts():
    history_hash=digest(ROOT/'data/processed/fights_dated.csv')
    builder_hash=digest(Path(__file__).with_name('build_v2_matchup_features.py'))
    loaded={}
    for kind,spec in SPECS.items():
        path=ROOT/'models'/f'{spec[0]}.joblib'
        meta=json.loads((ROOT/'models'/f'{spec[0]}_metadata.json').read_text())
        if digest(path)!=meta['model_sha256']:
            raise ValueError('Artifact checksum mismatch.')
        if meta['history_sha256']!=history_hash or meta['inference_builder_sha256']!=builder_hash:
            raise ValueError('History or inference builder changed; revalidate artifacts.')
        model=joblib.load(path)
        validate_model(model,meta,spec)
        loaded[kind]=(model,meta)
    prior=json.loads(PRIOR['ARTIFACT'].read_text())
    if (prior['model_id'],prior['version'])!=('ufc_round_prior','1.0.0'):
        raise ValueError('Unexpected round-prior version.')
    if prior['source_sha256']!=history_hash or prior['round_order']!=[f'ROUND_{i}' for i in range(1,6)]:
        raise ValueError('Round-prior snapshot or class order mismatch.')
    if any(meta['training_cutoff']!=prior['cutoff'] for _,meta in loaded.values()):
        raise ValueError('Artifact cutoffs disagree.')
    PRIOR['validate_prior'](prior)
    return loaded,prior


def clear_prediction_cache():
    load_artifacts.cache_clear()
    BUILD['clear_builder_cache']()


def check_distribution(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any() or not np.isclose(values.sum(),1,rtol=0,atol=1e-12):
        raise ValueError('Invalid probability distribution.')


def predict_fight_v2(fighter_a, fighter_b, prediction_date, scheduled_rounds):
    if isinstance(scheduled_rounds,bool) or not isinstance(scheduled_rounds,Integral) or scheduled_rounds not in (3,5):
        raise ValueError('scheduled_rounds must be integer 3 or 5, supplied independently of the outcome.')
    day=BUILD['OLD']['calendar_date'](prediction_date)
    builder=BUILD['default_builder']()
    winner_x=builder.build_winner_features(fighter_a,fighter_b,day)
    method_x=[builder.build_method_features(a,b,day) for a,b in [(fighter_a,fighter_b),(fighter_b,fighter_a)]]
    models,prior=load_artifacts()
    winner_model,wmeta=models['winner']
    method_model,mmeta=models['method']
    for x,meta in [(winner_x,wmeta),*( (x,mmeta) for x in method_x)]:
        if list(x)!=meta['features'] or len(x)!=1:
            raise ValueError('Inference feature schema mismatch.')
    wp=winner_model.predict_proba(winner_x)[0]
    check_distribution(wp)
    win={'A':float(wp[1]),'B':float(wp[0])}
    conditional={}
    for side,x in zip(('A','B'),method_x):
        p=method_model.predict_proba(x)[0]
        check_distribution(p)
        conditional[side]=dict(zip(METHODS,map(float,p)))
    joint={side:{method:win[side]*p for method,p in probabilities.items()} for side,probabilities in conditional.items()}
    check_distribution([p for values in joint.values() for p in values.values()])
    rounds=prior['schedules'][str(scheduled_rounds)]['probabilities']
    check_distribution(rounds)
    if scheduled_rounds==3 and rounds[3:]!=[0.,0.]:
        raise ValueError('Forbidden late-round probability.')
    canonical=[builder.base.names[builder.base.resolve(name)] for name in (fighter_a,fighter_b)]
    warnings=[
        'Supported-outcome approximation: draws, no contests, DQs and overturned outcomes have no probability mass. Winner training includes DQs; method training excludes them.',
        'METHOD-M has low submission recall and imperfect calibration; conditional estimates are uncertain.',
        'Round output is an empirical population prior, NOT an ML prediction or fighter/method-specific timing estimate; applies only to KO/TKO or Submission.',
        'Observed local UFC history may be incomplete; fighter-name identity and retrospective result-correction limitations persist.']
    cutoff=prior['cutoff']
    warnings.append('Historical sanity check only: fitted models and the fixed round prior include history through '+cutoff+'; not an as-of-date forecast.' if day<=pd.Timestamp(cutoff)
                    else 'Local artifact cutoff is '+cutoff+'; subsequent fights may be absent from the history snapshot.')
    for name in canonical:
        h=builder.fighter_features(name,day)
        if h['previous_fights']<3:
            warnings.append(f"{name}: limited observed history ({h['previous_fights']} prior fights); zero denotes an observed debut, not a verified career debut.")
        if h['stats_coverage']<1 or h['previous_fights']==0:
            warnings.append(f'{name}: incomplete or absent historical statistics; missing differences use neutral imputation.')
        gap=h['days_since_last_fight']
        if pd.notna(gap) and gap>365:
            warnings.append(f'{name}: {int(gap)} days since last observed UFC fight; this establishes no injury or retirement status.')
    missing={'winner':int(winner_x.isna().sum().sum()),'method_A':int(method_x[0].isna().sum().sum()),'method_B':int(method_x[1].isna().sum().sum())}
    if any(missing.values()): warnings.append(f'Missing relative features imputed as zero unknown advantage: {missing}.')
    return {'fighter_a':canonical[0],'fighter_b':canonical[1],'prediction_date':str(day.date()),'scheduled_rounds':int(scheduled_rounds),
            'winner_probabilities':win,'method_given_win':conditional,'joint_winner_method':joint,
            'round_given_finish':{'type':'empirical_population_prior','applies_to':['KO_TKO','SUBMISSION'],
                                  'probabilities':dict(zip(prior['round_order'],rounds)),'sample_count':prior['schedules'][str(scheduled_rounds)]['total']},
            'metadata':{'winner':wmeta['model_id'],'method':mmeta['model_id'],'round_prior':'ufc_round_prior_v1_0_0',
                        'cutoff':cutoff,'missing_feature_counts':missing},'warnings':warnings}
