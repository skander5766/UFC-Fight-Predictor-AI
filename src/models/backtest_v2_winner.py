"""Seven predefined symmetric logistic experiments; no model artifacts or tuning.

FULL adds V2 evidence/denominator differences to M. Redundant counts remain
deliberately visible for this predefined family experiment, not test-selected.
2025+ is excluded before target/feature inspection. Earlier test-period fights
can inform later histories, but each fitted model stays fixed for its fold.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'models/v2_winner_backtest_metrics.json'
BASE = runpy.run_path(str(ROOT/'src/models/train_baseline.py'))
ENRICHED = runpy.run_path(str(ROOT/'src/models/train_enriched_symmetric_logistic.py'))
V2 = runpy.run_path(str(ROOT/'src/features/build_v2_features.py'))
PRIOR = runpy.run_path(str(ROOT/'src/models/backtest_logistic_models.py'))
METRICS = PRIOR['METRICS']
QUALITY = ['average_previous_opponent_elo', 'maximum_previous_opponent_elo',
           'average_opponent_elo_last_3', 'average_opponent_elo_last_5']
DEFENSE = list(V2['DEFENSE']) + ['takedown_defense_rate', 'significant_strikes_absorbed_per_minute']
METHOD = [f'previous_{m}_{result}' for m in ['ko_tko','submission','decision'] for result in ['wins','losses']]
METHOD += [f'{m}_win_rate_per_{denominator}' for m in ['ko_tko','submission','decision'] for denominator in ['win','fight']]
FINISH = ['previous_finish_wins','previous_finish_losses','previous_times_finished',
          'finish_win_rate','finish_loss_rate','previous_round_1_finish_wins',
          'previous_round_2_finish_wins','previous_round_3_finish_wins','previous_round_4_5_finish_wins',
          'previous_early_finish_wins','early_finish_win_rate','average_finish_round',
          'average_fight_duration_seconds','average_finish_duration_seconds']
EVIDENCE = ['method_history_fights','method_history_wins','observed_history_fights',
            'average_finish_round_history_count','average_fight_duration_seconds_history_count',
            'average_finish_duration_seconds_history_count']
EVIDENCE += [name+'_history_count' for name in V2['DEFENSE']]
EVIDENCE += ['takedown_defense_history_count','opponent_td_attempts_history_total','defensive_timed_history_count']


def families():
    groups = {'A': list(BASE['FEATURES'])}
    groups['C'] = groups['A'] + ENRICHED['ACTIVITY'] + ENRICHED['LAYOFF']
    groups['E'] = groups['C'] + ['elo_diff']
    groups['Q'] = groups['E'] + [x+'_diff' for x in QUALITY]
    groups['D'] = groups['Q'] + [x+'_diff' for x in DEFENSE]
    groups['M'] = groups['D'] + [x+'_diff' for x in METHOD+FINISH]
    groups['FULL'] = groups['M'] + [x+'_diff' for x in EVIDENCE]
    for features in groups.values():
        assert len(features) == len(set(features))
        assert not set(features) & BASE['FORBIDDEN']
    return groups


def scores(y, p):
    return BASE['evaluate'](pd.Series(y), p, (p >= .5).astype(int)) | {
        'mean_fighter_1_probability': float(np.mean(p)), 'actual_fighter_1_win_rate': float(np.mean(y))}


def main():
    start = perf_counter()
    if REPORT.exists():
        raise FileExistsError(REPORT)
    protected = [p for directory in ('data','models') for p in (ROOT/directory).rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    groups = families()
    print('EXACT FEATURE LISTS:', json.dumps(groups, indent=2), flush=True)
    data = pd.read_csv(ROOT/'data/processed/prefight_features_v2.csv', low_memory=False)
    data['event_date'] = pd.to_datetime(data.event_date, errors='raise')
    excluded = int(data.event_date.ge('2025-01-01').sum())
    data = data.loc[data.event_date.lt('2025-01-01')].copy()
    # Rebuild only pre-2025 V2 histories and compare every stored V2 quantity.
    source = pd.read_csv(V2['SOURCE'], low_memory=False)
    source['event_date'] = pd.to_datetime(source.event_date, errors='raise')
    rebuilt = V2['build_features'](source.loc[source.event_date.lt('2025-01-01')])
    expected = rebuilt.set_index('fight_id').loc[data.fight_id]
    np.testing.assert_allclose(data[expected.columns], expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    assert data.fight_id.is_unique and data.URL.notna().all()
    assert data.outcome.isin(['fighter_1_win','fighter_2_win','draw','no_contest']).all()
    nonbinary = ~data.outcome.isin(['fighter_1_win','fighter_2_win'])
    filtering = {'excluded_2025_plus_rows':excluded, 'excluded_pre2025_draws_no_contests':int(nonbinary.sum())}
    data = data.loc[~nonbinary].copy()
    data['target'] = data.outcome.map({'fighter_1_win':1,'fighter_2_win':0})
    old = ENRICHED['design'](data, groups['A'])
    for name in ENRICHED['LAYOFF']:
        data[name] = old[name]
    checked = ['event_date','fighter_1','fighter_2','target']+groups['FULL']
    assert not data.groupby('URL')[checked].nunique(dropna=False).gt(1).any().any()
    filtering['removed_pre2025_duplicate_sources'] = int(data.URL.duplicated().sum())
    data = data.sort_values(['event_date','fight_id']).drop_duplicates('URL').reset_index(drop=True)
    filtering['eligible_pre2025_fights'] = len(data)
    prior = json.loads(PRIOR['REPORT'].read_text())
    assert filtering == prior['filtering']
    assert groups['A'] == prior['features']['A'] and groups['C'] == prior['features']['C']
    original = pd.read_csv(BASE['INPUT'], low_memory=False)
    original['event_date'] = pd.to_datetime(original.event_date)
    original = original.loc[original.event_date.lt('2025-01-01') & original.outcome.isin(['fighter_1_win','fighter_2_win'])]
    original = original.sort_values(['event_date','fight_id']).drop_duplicates('URL').reset_index(drop=True)
    pd.testing.assert_frame_equal(data[original.columns], original)
    # Check all V2 differences against opposite-side components, not suffixes alone.
    for name in groups['FULL'][50:]:
        component = 'prefight_elo' if name == 'elo_diff' else name[:-5]
        np.testing.assert_allclose(data[name], data['fighter_1_'+component]-data['fighter_2_'+component], equal_nan=True, atol=1e-10)
    np.testing.assert_allclose(ENRICHED['design'](data, groups['A'], swap=True)[groups['C']], -data[groups['C']], equal_nan=True, atol=1e-10)
    assert not np.isinf(data[groups['FULL']].to_numpy()).any()
    folds, pooled, rows = [], {g:[] for g in groups}, []
    for number, (lower, upper) in enumerate(PRIOR['FOLDS'], 1):
        train = data.loc[data.event_date.lt(f'{lower}-01-01')]
        test = data.loc[data.event_date.between(f'{lower}-01-01',f'{upper}-12-31')]
        assert train.event_date.max() < test.event_date.min()
        for key in ['fight_id','URL']:
            assert set(train[key]).isdisjoint(test[key])
        assert len(train) == prior['folds'][number-1]['train_fights']
        assert len(test) == prior['folds'][number-1]['test_fights']
        fold = {'fold':number, 'train_fights':len(train),'test_fights':len(test),
                'train_dates':[str(train.event_date.min().date()),str(train.event_date.max().date())],
                'test_dates':[str(test.event_date.min().date()),str(test.event_date.max().date())], 'models':{}}
        for name, features in groups.items():
            x = train[features]
            xa = pd.concat([x,-x], ignore_index=True)
            ya = pd.concat([train.target,1-train.target], ignore_index=True)
            assert ya.value_counts().to_dict() == {0:len(train),1:len(train)}
            np.testing.assert_allclose(xa.iloc[len(x):],-xa.iloc[:len(x)],equal_nan=True,atol=0,rtol=0)
            np.testing.assert_array_equal(ya.iloc[len(x):],1-ya.iloc[:len(x)])
            pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                                 ('scaler',StandardScaler(with_mean=False)),('logistic',LogisticRegression(max_iter=2000))])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                pipeline.fit(xa,ya)
            scaler = pipeline.named_steps['scaler']
            assert np.all(scaler.n_samples_seen_ == len(xa))
            np.testing.assert_allclose(scaler.var_, xa.fillna(0).var(ddof=0),atol=1e-12)
            np.testing.assert_array_equal(pipeline.named_steps['imputer'].statistics_, np.zeros(len(features)))
            state = joblib.hash(pipeline)
            p = pipeline.predict_proba(test[features])[:,1]
            q = pipeline.predict_proba(-test[features])[:,1]
            errors = np.abs(p+q-1)
            assert errors.max() < 1e-5 and state == joblib.hash(pipeline)
            coef = pd.Series(pipeline.named_steps['logistic'].coef_[0],index=features).sort_values()
            result = {'metrics':scores(test.target.to_numpy(),p),'symmetry_mean':float(errors.mean()),
                      'symmetry_max':float(errors.max()),'intercept':float(pipeline.named_steps['logistic'].intercept_[0]),
                      'coefficients':coef.to_dict(),'negative_coefficients':coef[coef<0].head(10).to_dict(),
                      'positive_coefficients':coef[coef>0].tail(10).iloc[::-1].to_dict()}
            fold['models'][name] = result
            pooled[name].append((test.target.to_numpy(),p))
            row = {'fold':number,'configuration':name,**{m:result['metrics'][m] for m in METRICS}}
            rows.append(row)
            print(row,flush=True)
        folds.append(fold)
    table = pd.DataFrame(rows)
    aggregate = {}
    for name in groups:
        sample = table.loc[table.configuration.eq(name),METRICS]
        y,p = [np.concatenate([entry[i] for entry in pooled[name]]) for i in (0,1)]
        aggregate[name] = {'fold_mean':sample.mean().to_dict(),'fold_sample_std':sample.std(ddof=1).to_dict(),
                           'pooled':scores(y,p),'pooled_fights':len(y)}
    pairs = [('C','E'),('E','Q'),('Q','D'),('D','M'),('M','FULL')]
    differences = [{'fold':f['fold'],'comparison':b+' minus '+a,
                    **{m:f['models'][b]['metrics'][m]-f['models'][a]['metrics'][m] for m in METRICS}}
                   for f in folds for a,b in pairs]
    pooled_differences = {b+' minus '+a:{m:aggregate[b]['pooled'][m]-aggregate[a]['pooled'][m] for m in METRICS} for a,b in pairs}
    assert hashes == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = {'filtering':filtering,'features':groups,'folds':folds,'aggregate':aggregate,
              'incremental_fold_deltas':differences,'incremental_pooled_deltas':pooled_differences,
              'validation':{'reproduced_pre2025_v2_features':True,'exact_prior_filtering_and_folds':True,
                            'train_only_preprocessing_and_augmentation':True,'no_id_url_overlap':True,
                            'symmetry_passed':True,'existing_data_models_unchanged':True},
              'notes':['No fitted models saved; no hyperparameter tuning or test-selected features.',
                       '2025+ excluded here, but previously inspected; not a pristine holdout.',
                       'Final recorded reversals lack publication timestamps; point-in-time limitation persists.',
                       'Coefficients are standardized associations, not causal effects; redundant counts split weights.',
                       'FULL adds evidence counts; physical and stance excluded from these V2 groups.',
                       'Fold SD uses ddof=1. Pooled AUC compares scores from different fitted models.',
                       'Method/finish eligibility excludes DQ/draw/NC; defense uses complete opponent stats.'],
              'runtime_seconds':perf_counter()-start}
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('FILTERING',filtering,'\nAGGREGATE',json.dumps(aggregate,indent=2),flush=True)
    print('DELTAS',json.dumps(pooled_differences,indent=2),'\nRuntime',report['runtime_seconds'],flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
