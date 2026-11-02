"""Experimental P(method | candidate wins), with fixed temporal folds.

One observed winner per fight; do NOT augment with the losing candidate.
Missing differences mean unknown relative advantage and are imputed to zero.
No evidence-count duplicates, aggregate finish counts, or per-win method-rate
simplex are included. Groups are fixed by semantics, not test performance.
Conditional distributions for opposing candidates need not be complementary.
"""
import hashlib
import json
from pathlib import Path
import runpy
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'models/method_backtest_metrics.json'
V2 = runpy.run_path(str(ROOT/'src/features/build_v2_features.py'))
BASE = runpy.run_path(str(ROOT/'src/models/train_baseline.py'))
CLASSES = ['KO_TKO', 'SUBMISSION', 'DECISION']
METRICS = ['accuracy','balanced_accuracy','macro_f1','log_loss','brier_score']
FOLDS = [(2017,2018),(2019,2020),(2021,2022),(2023,2024)]
CORE = ['previous_wins','previous_losses','previous_draws','previous_no_contests',
        'historical_win_rate','wins_last_3','wins_last_5','current_win_streak','current_loss_streak',
        'days_since_last_fight','stats_coverage',
        'sig_str_landed_per_fight','sig_str_attempted_per_fight','historical_sig_str_accuracy',
        'kd_per_fight','td_landed_per_fight','td_attempted_per_fight','historical_td_accuracy',
        'sub_att_per_fight','ctrl_seconds_per_fight',
        'fights_last_365_days','fights_last_730_days','days_since_second_last_fight',
        'average_days_between_previous_fights','layoff_over_365_days','layoff_over_730_days','layoff_over_1095_days']
DEFENSE = list(V2['DEFENSE']) + ['significant_strikes_absorbed_per_minute','takedown_defense_rate']
METHOD = [f'previous_{m}_{r}' for m in ('ko_tko','submission','decision') for r in ('wins','losses')]
METHOD += ['finish_win_rate','finish_loss_rate','previous_early_finish_wins',
           'average_finish_round','average_fight_duration_seconds','average_finish_duration_seconds']


def groups():
    a = CORE
    e = a + ['prefight_elo']
    d = e + DEFENSE
    return {'METHOD-A':a,'METHOD-E':e,'METHOD-D':d,'METHOD-M':d+METHOD}


def candidate_features(frame, features, candidate_side):
    """Caller supplies candidate side; no current winner/outcome is read here."""
    side = pd.Series(candidate_side, index=frame.index)
    if not side.isin([1,2]).all():
        raise ValueError('Candidate side must be 1 or 2.')
    output = {}
    for feature in features:
        values = []
        for s in (1,2):
            value = frame[f'fighter_{s}_{feature}']
            if feature.startswith('layoff_'):
                text = value.astype('string').str.lower()
                if not text.dropna().isin(['true','false']).all():
                    raise ValueError('Invalid layoff value')
                value = text.map({'true':1.,'false':0.}).astype(float)
            values.append(value)
        output[feature+'_diff'] = (values[0]-values[1]) * side.map({1:1,2:-1})
    return pd.DataFrame(output,index=frame.index)


def evaluate(y, probability):
    y = np.asarray(y)
    prediction = probability.argmax(axis=1)
    onehot = np.eye(3)[y]
    return {'accuracy':float(accuracy_score(y,prediction)),
            'balanced_accuracy':float(balanced_accuracy_score(y,prediction)),
            'macro_f1':float(f1_score(y,prediction,average='macro',zero_division=0)),
            'log_loss':float(log_loss(y,probability,labels=[0,1,2])),
            'brier_score':float(np.mean(np.sum((probability-onehot)**2,axis=1))),
            'confusion_matrix':confusion_matrix(y,prediction,labels=[0,1,2]).tolist(),
            'per_class':classification_report(y,prediction,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0),
            'actual_counts':dict(zip(CLASSES,np.bincount(y,minlength=3).tolist())),
            'predicted_counts':dict(zip(CLASSES,np.bincount(prediction,minlength=3).tolist()))}


def reliability(y,p):
    result = {}
    for k,name in enumerate(CLASSES):
        bins = []
        indexes = np.minimum((p[:,k]*10).astype(int),9)
        for b in range(10):
            mask = indexes == b
            if mask.any():
                bins.append({'lower':b/10,'upper':(b+1)/10,'count':int(mask.sum()),
                             'predicted':float(p[mask,k].mean()),'observed':float((y[mask]==k).mean())})
        result[name] = {'mean_probability':float(p[:,k].mean()),'actual_rate':float((y==k).mean()),
                        'ece_10_bins':sum(b['count']*abs(b['predicted']-b['observed']) for b in bins)/len(y), 'bins':bins}
    return result


def main():
    start = perf_counter()
    if REPORT.exists():
        raise FileExistsError(REPORT)
    protected = [p for folder in ('data','models','app','src/prediction') for p in (ROOT/folder).rglob('*') if p.is_file()]
    hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    frame = pd.read_csv(ROOT/'data/processed/prefight_features_v2.csv',low_memory=False)
    source = pd.read_csv(V2['SOURCE'],low_memory=False)
    assert source.fight_id.is_unique and frame.fight_id.is_unique
    joined = frame.merge(source[['fight_id','METHOD']],on='fight_id',validate='one_to_one')
    assert len(joined)==len(frame)
    binary, counts = BASE['load_binary_fights']()
    joined = joined.set_index('fight_id').loc[binary.fight_id].reset_index()
    joined['event_date'] = pd.to_datetime(joined.event_date)
    normalized = joined.METHOD.map(V2['AUDIT']['normalize_method'])
    population = {'binary_filter':counts,'excluded_methods':joined.loc[normalized.isna(),'METHOD'].value_counts().to_dict(),
                  'eligible_all_dates':int(normalized.notna().sum()),'all_date_distribution':normalized.value_counts().to_dict()}
    assert population['eligible_all_dates']==8524
    joined['target'] = normalized.map({'KO/TKO':0,'Submission':1,'Decision':2})
    frame = joined.loc[normalized.notna() & joined.event_date.lt('2025-01-01')].copy()
    frame['target'] = frame.target.astype(int)
    population['eligible_pre2025'] = len(frame)
    population['excluded_2025_plus_eligible'] = population['eligible_all_dates']-len(frame)
    population['pre2025_distribution'] = frame.METHOD.map(V2['AUDIT']['normalize_method']).value_counts().to_dict()
    # All-date access above is population counting only; no reserved-period fitting/evaluation.
    source['event_date'] = pd.to_datetime(source.event_date)
    regenerated = V2['build_features'](source.loc[source.event_date.lt('2025-01-01')]).set_index('fight_id').loc[frame.fight_id]
    np.testing.assert_allclose(frame[regenerated.columns],regenerated,atol=1e-10,rtol=1e-10,equal_nan=True)
    side = frame.outcome.map({'fighter_1_win':1,'fighter_2_win':2})
    assert frame.winner.eq(frame.fighter_1.where(side.eq(1),frame.fighter_2)).all()
    all_features = groups()['METHOD-M']
    assert not set(all_features)&BASE['FORBIDDEN']
    x = candidate_features(frame,all_features,side)
    swapped = frame.copy()
    for feature in all_features:
        swapped[f'fighter_1_{feature}'],swapped[f'fighter_2_{feature}'] = frame[f'fighter_2_{feature}'],frame[f'fighter_1_{feature}']
    reverse = candidate_features(swapped,all_features,3-side)
    np.testing.assert_allclose(x,reverse,rtol=0,atol=0,equal_nan=True)
    assert not np.isinf(x.to_numpy()).any()
    folds, pooled = [], {name:[] for name in [*groups(),'majority','training_prior']}
    seen = set()
    for number,(first,last) in enumerate(FOLDS,1):
        train = frame.loc[frame.event_date.lt(f'{first}-01-01')]
        test = frame.loc[frame.event_date.between(f'{first}-01-01',f'{last}-12-31')]
        assert train.event_date.max()<test.event_date.min()
        assert set(train.URL).isdisjoint(test.URL) and set(train.fight_id).isdisjoint(test.fight_id)
        assert seen.isdisjoint(test.fight_id)
        seen.update(test.fight_id)
        y = test.target.to_numpy()
        priors = train.target.value_counts(normalize=True).reindex([0,1,2]).to_numpy()
        assert np.isfinite(priors).all()
        fold = {'fold':number,'train_fights':len(train),'test_fights':len(test),
                'train_dates':[str(train.event_date.min().date()),str(train.event_date.max().date())],
                'test_dates':[str(test.event_date.min().date()),str(test.event_date.max().date())],
                'training_priors':dict(zip(CLASSES,priors.tolist())),'models':{}}
        for name,features in groups().items():
            columns = [f+'_diff' for f in features]
            pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                                 ('scaler',StandardScaler(with_mean=False)),
                                 ('logistic',LogisticRegression(solver='lbfgs',max_iter=2000))])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                pipeline.fit(x.loc[train.index,columns],train.target)
            np.testing.assert_array_equal(pipeline.classes_,[0,1,2])
            assert np.all(pipeline.named_steps['scaler'].n_samples_seen_==len(train))
            np.testing.assert_allclose(pipeline.named_steps['scaler'].var_,x.loc[train.index,columns].fillna(0).var(ddof=0),atol=1e-12)
            state = joblib.hash(pipeline)
            p = pipeline.predict_proba(x.loc[test.index,columns])
            q = pipeline.predict_proba(reverse.loc[test.index,columns])
            np.testing.assert_array_equal(p,q)
            assert state==joblib.hash(pipeline)
            fold['models'][name] = {'metrics':evaluate(y,p),'calibration':reliability(y,p),'orientation_max_error':float(np.abs(p-q).max())}
            pooled[name].append((y,p))
        for name,p in [('majority',np.tile(np.eye(3)[priors.argmax()],(len(y),1))),('training_prior',np.tile(priors,(len(y),1)))]:
            fold['models'][name] = {'metrics':evaluate(y,p),'calibration':reliability(y,p)}
            pooled[name].append((y,p))
        print('FOLD',number,'sizes',len(train),len(test),{n:{k:v['metrics'][k] for k in METRICS} for n,v in fold['models'].items()},flush=True)
        folds.append(fold)
    aggregate = {}
    for name,entries in pooled.items():
        y,p = [np.concatenate([e[i] for e in entries]) for i in (0,1)]
        table = pd.DataFrame([{k:f['models'][name]['metrics'][k] for k in METRICS} for f in folds])
        aggregate[name] = {'mean':table.mean().to_dict(),'sample_std':table.std(ddof=1).to_dict(),
                           'pooled':evaluate(y,p),'calibration':reliability(y,p),'pooled_fights':len(y)}
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = {'population':population,'classes':CLASSES,'features':{n:[f+'_diff' for f in fs] for n,fs in groups().items()},
              'folds':folds,'aggregate':aggregate,'runtime_seconds':perf_counter()-start,
              'validation':{'pre2025_v2_reproduction':True,'same_candidate_source_swap':True,'training_only_preprocessing':True,
                            'no_train_test_id_url_overlap':True,'existing_files_unchanged':True},
              'notes':['No augmentation, oversampling, class weights, tuning, calibration fitting, or model artifacts.',
                       'Multiclass lbfgs logistic uses multinomial loss with three classes.',
                       'Brier = mean sum of squared class errors (not divided by number of classes), range 0-2.',
                       'Hard majority log loss uses sklearn numerical clipping; training-prior is the useful probability baseline.',
                       'Missing differences filled with zero unknown relative advantage; scaler fitted only to training.',
                       'Current outcome selects the observed candidate only; it is not a predictor.',
                       'P(method|A wins) and P(method|B wins) are different conditional distributions, not complements.',
                       'Later reversals lack publication dates. Observed UFC history and fighter-name identity limitations persist.',
                       '2025+ counted for population verification only; excluded from all fitting, comparisons, calibration and feature reconstruction.',
                       'Pooled reliability bins are descriptive only; no calibration model fitted.']}
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('POPULATION',population,'COUNTS',{n:len(fs) for n,fs in groups().items()},'RUNTIME',report['runtime_seconds'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
