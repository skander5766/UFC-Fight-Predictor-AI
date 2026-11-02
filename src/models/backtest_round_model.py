"""Experimental finish-round distributions; no production artifacts.

ROUND-METHOD evaluates an oracle conditioning context, not an end-to-end
forecast. Future use must sum P(round|winner,method,schedule)*P(method|winner,
finish) over KO/TKO and Submission; the latter weights exclude Decision and
renormalize. Schedule is provided independently, never inferred from outcome.
"""
from pathlib import Path
import runpy
import json
import hashlib
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
METHOD = runpy.run_path(str(ROOT/'src/models/backtest_method_model.py'))
REPORT = ROOT/'models/round_backtest_metrics.json'
CLASSES = [f'ROUND_{i}' for i in range(1,6)]
METRICS = ['accuracy','balanced_accuracy','macro_f1','log_loss','brier_score','mae_rounds']
ROUND_HISTORY = [x for x in METHOD['METHOD'] if x != 'previous_early_finish_wins'] + [
    'previous_round_1_finish_wins','previous_round_2_finish_wins','previous_round_3_finish_wins','previous_round_4_5_finish_wins']


def groups():
    a = METHOD['CORE']
    d = a + ['prefight_elo'] + METHOD['DEFENSE']
    m = d + ROUND_HISTORY
    return {'ROUND-A':a,'ROUND-D':d,'ROUND-M':m,'ROUND-METHOD':m}


def schedule_mask(probability, schedules):
    p = np.asarray(probability,dtype=float).copy()
    s = np.asarray(schedules)
    if p.shape != (len(s),5) or not np.isin(s,[3,5]).all() or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Invalid probabilities or unsupported schedule')
    p[s==3,3:] = 0
    if (p.sum(axis=1)<=0).any():
        raise ValueError('No probability mass on allowed rounds')
    p /= p.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    assert (p[s==3,3:]==0).all()
    return p


def evaluate(y,p):
    y = np.asarray(y)
    pred = p.argmax(axis=1)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)),
                log_loss=float(log_loss(y,p,labels=list(range(5)))),
                brier_score=float(np.mean(np.sum((p-np.eye(5)[y])**2,axis=1))),
                mae_rounds=float(np.mean(np.abs(y-pred))),
                confusion_matrix=confusion_matrix(y,pred,labels=range(5)).tolist(),
                per_class=classification_report(y,pred,labels=range(5),target_names=CLASSES,output_dict=True,zero_division=0),
                actual_counts=np.bincount(y,minlength=5).tolist(),predicted_counts=np.bincount(pred,minlength=5).tolist())


def main():
    start = perf_counter()
    if REPORT.exists(): raise FileExistsError(REPORT)
    protected = [p for folder in ('data','models','app','src/prediction') for p in (ROOT/folder).rglob('*') if p.is_file()]
    hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    source = pd.read_csv(METHOD['V2']['SOURCE'],low_memory=False)
    data = pd.read_csv(ROOT/'data/processed/prefight_features_v2.csv',low_memory=False)
    binary, filtering = METHOD['BASE']['load_binary_fights']()
    assert source.fight_id.is_unique and data.fight_id.is_unique
    fields = ['fight_id','METHOD','ROUND','TIME FORMAT']
    for _,g in source.groupby('URL'):
        assert not g[fields[1:]].nunique(dropna=False).gt(1).any()
    data = data.merge(source[fields],on='fight_id',validate='one_to_one').copy()
    data = data.set_index('fight_id').loc[binary.fight_id].reset_index().copy()
    data['method'] = data.METHOD.map(METHOD['V2']['AUDIT']['normalize_method'])
    data = data.loc[data.method.isin(['KO/TKO','Submission'])].copy()
    population = {'all_eligible_finishes':len(data),'binary_filtering':filtering}
    data['schedule'] = data['TIME FORMAT'].map({'3 Rnd (5-5-5)':3,'5 Rnd (5-5-5-5-5)':5})
    population['excluded_formats'] = data.loc[data.schedule.isna(),'TIME FORMAT'].value_counts().to_dict()
    population['excluded_other_formats'] = int(data.schedule.isna().sum())
    data = data.loc[data.schedule.notna()].copy()
    population['modern_finishes'] = len(data)
    population['modern_schedule_counts'] = data.schedule.value_counts().to_dict()
    population['modern_round_counts'] = data.ROUND.value_counts().sort_index().to_dict()
    assert len(data)==4325 and population['all_eligible_finishes']==4522
    data['event_date'] = pd.to_datetime(data.event_date)
    data = data.loc[data.event_date.lt('2025-01-01')].copy()
    data['target'] = pd.to_numeric(data.ROUND,errors='raise')-1
    assert data.target.mod(1).eq(0).all() and data.target.ge(0).all() and (data.target<data.schedule).all()
    data['target'] = data.target.astype(int)
    population['primary_finishes'] = len(data)
    population['primary_schedule_counts'] = data.schedule.value_counts().to_dict()
    population['primary_round_counts'] = data.ROUND.value_counts().sort_index().to_dict()
    source['event_date'] = pd.to_datetime(source.event_date)
    rebuilt = METHOD['V2']['build_features'](source.loc[source.event_date.lt('2025-01-01')]).set_index('fight_id').loc[data.fight_id]
    np.testing.assert_allclose(data[rebuilt.columns],rebuilt,atol=1e-10,rtol=1e-10,equal_nan=True)
    side = data.outcome.map({'fighter_1_win':1,'fighter_2_win':2})
    assert data.winner.eq(data.fighter_1.where(side.eq(1),data.fighter_2)).all()
    x = METHOD['candidate_features'](data,groups()['ROUND-M'],side)
    swapped = data.copy()
    for f in groups()['ROUND-M']:
        swapped[f'fighter_1_{f}'],swapped[f'fighter_2_{f}'] = data[f'fighter_2_{f}'],data[f'fighter_1_{f}']
    reverse = METHOD['candidate_features'](swapped,groups()['ROUND-M'],3-side)
    np.testing.assert_allclose(x,reverse,rtol=0,atol=0,equal_nan=True)
    for frame in (x,reverse):
        frame['scheduled_five_rounds'] = data.schedule.eq(5).astype(float)
        frame['conditioning_submission'] = data.method.eq('Submission').astype(float)
    feature_lists = {n:[f+'_diff' for f in fs]+['scheduled_five_rounds']+(['conditioning_submission'] if n=='ROUND-METHOD' else []) for n,fs in groups().items()}
    print('FEATURES',json.dumps(feature_lists),flush=True)
    pooled = {n:[] for n in [*groups(),'majority','training_prior','schedule_prior']}
    folds = []
    seen = set()
    for number,(first,last) in enumerate(METHOD['FOLDS'],1):
        train = data.loc[data.event_date.lt(f'{first}-01-01')]
        test = data.loc[data.event_date.between(f'{first}-01-01',f'{last}-12-31')]
        assert train.event_date.max()<test.event_date.min()
        for key in ['fight_id','URL']: assert set(train[key]).isdisjoint(test[key])
        assert seen.isdisjoint(test.fight_id)
        seen.update(test.fight_id)
        counts = np.bincount(train.target,minlength=5)
        if (counts==0).any(): raise ValueError('A training round class is absent; cannot fit planned five-class model')
        fold = {'fold':number,'train_fights':len(train),'test_fights':len(test),'train_counts':counts.tolist(),
                'test_counts':np.bincount(test.target,minlength=5).tolist(),
                'train_dates':[str(train.event_date.min().date()),str(train.event_date.max().date())],
                'test_dates':[str(test.event_date.min().date()),str(test.event_date.max().date())],'models':{}}
        y = test.target.to_numpy()
        for name,features in feature_lists.items():
            pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                                 ('scaler',StandardScaler(with_mean=False)),('logistic',LogisticRegression(max_iter=2000,solver='lbfgs'))])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                pipeline.fit(x.loc[train.index,features],train.target)
            np.testing.assert_array_equal(pipeline.classes_,np.arange(5))
            assert np.all(pipeline.named_steps['scaler'].n_samples_seen_==len(train))
            np.testing.assert_allclose(pipeline.named_steps['scaler'].var_,x.loc[train.index,features].fillna(0).var(ddof=0),atol=1e-12)
            raw = pipeline.predict_proba(x.loc[test.index,features])
            np.testing.assert_array_equal(raw,pipeline.predict_proba(reverse.loc[test.index,features]))
            p = schedule_mask(raw,test.schedule)
            fold['models'][name] = evaluate(y,p)
            pooled[name].append((y,p))
        priors = counts/counts.sum()
        schedule_prior = np.vstack([np.bincount(train.loc[train.schedule.eq(s),'target'],minlength=5)/train.schedule.eq(s).sum() for s in test.schedule])
        for name,raw in [('majority',np.tile(np.eye(5)[counts.argmax()],(len(test),1))),
                         ('training_prior',np.tile(priors,(len(test),1))),('schedule_prior',schedule_prior)]:
            p = schedule_mask(raw,test.schedule)
            fold['models'][name] = evaluate(y,p)
            pooled[name].append((y,p))
        print('FOLD',number,'sizes',len(train),len(test),'counts',counts.tolist(),fold['test_counts'],
              {n:{m:v[m] for m in METRICS} for n,v in fold['models'].items()},flush=True)
        folds.append(fold)
    aggregate = {}
    for name,entries in pooled.items():
        y,p = [np.concatenate([e[i] for e in entries]) for i in (0,1)]
        table = pd.DataFrame([{m:f['models'][name][m] for m in METRICS} for f in folds])
        aggregate[name] = {'mean':table.mean().to_dict(),'sample_std':table.std(ddof=1).to_dict(),'pooled':evaluate(y,p)}
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = {'population':population,'classes':CLASSES,'features':feature_lists,'folds':folds,'aggregate':aggregate,
              'runtime_seconds':perf_counter()-start,
              'validation':{'candidate_orientation_exact':True,'schedule_mask_all_models_baselines':True,
                            'pre2025_history_reproduction':True,'strict_chronology':True,'existing_files_unchanged':True},
              'notes':['ROUND-METHOD has true method as conditional context; not an end-to-end forecast or fair claim of unconditional improvement.',
                       'Future marginalization weights KO and SUB probabilities conditional on winning AND finishing; do not use true future method.',
                       'MAE uses argmax predicted round. Brier sums five class errors, range 0-2. Macro F1 averages all five classes.',
                       'Global training prior is schedule-masked; schedule prior estimates separate distributions per scheduled format.',
                       'Missing differences -> zero unknown relative advantage. Training-only imputation and uncentered scaling.',
                       'All classes retained; no tuning, oversampling, class weighting, or saved fitted models.',
                       'Rare R4/R5 counts limit reliable separate-class estimation.',
                       '2025+ counted only for population verification, excluded from primary fitting and evaluation.',
                       'Recorded reversals lack publication times; same observed-history limitations as V2.']}
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('POPULATION',population,'RUNTIME',report['runtime_seconds'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
