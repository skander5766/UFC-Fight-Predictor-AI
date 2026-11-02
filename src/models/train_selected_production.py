"""Fit fixed winner D v2.0.0 and METHOD-M v1.0.0; no fit-quality evaluation."""
import hashlib
import json
from pathlib import Path
import runpy
import platform
import warnings
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = runpy.run_path(str(ROOT/'src/prediction/build_v2_matchup_features.py'))
W = INFERENCE['WINNER']
M = INFERENCE['METHOD']
ARTIFACTS = {'winner': 'ufc_winner_d_v2_0_0', 'method':'ufc_method_m_v1_0_0'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    outputs = [ROOT/'models'/f'{name}{suffix}' for name in ARTIFACTS.values() for suffix in ('.joblib','_metadata.json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Versioned artifacts already exist; refusing to overwrite.')
    protected = [p for folder in ('data','models','app','src/prediction') for p in (ROOT/folder).rglob('*') if p.is_file()]
    hashes = {p:sha(p) for p in protected}
    data = pd.read_csv(ROOT/'data/processed/prefight_features_v2.csv',low_memory=False)
    original = pd.read_csv(W['BASE']['INPUT'],low_memory=False)
    pd.testing.assert_frame_equal(data[original.columns],original)
    binary,filtering = W['BASE']['load_binary_fights']()
    frame = data.set_index('fight_id').loc[binary.fight_id].reset_index().copy()
    frame['event_date'] = binary.event_date.to_numpy()
    wf = list(INFERENCE['WINNER_FEATURES'])
    mf = list(INFERENCE['METHOD_FEATURES'])
    wr = json.loads((ROOT/'models/v2_winner_backtest_metrics.json').read_text())
    mr = json.loads((ROOT/'models/method_backtest_metrics.json').read_text())
    assert wf==wr['features']['D'] and mf==mr['features']['METHOD-M']
    for name in W['ENRICHED']['LAYOFF']:
        key = name[:-5]
        frame[name] = W['ENRICHED']['boolean_values'](frame['fighter_1_'+key])-W['ENRICHED']['boolean_values'](frame['fighter_2_'+key])
    for name in wf:
        key = 'prefight_elo' if name=='elo_diff' else W['ENRICHED']['ALIASES'].get(name[:-5],name[:-5])
        if name not in W['ENRICHED']['LAYOFF']:
            np.testing.assert_allclose(frame[name],frame['fighter_1_'+key]-frame['fighter_2_'+key],atol=1e-10,equal_nan=True)
    source = pd.read_csv(M['V2']['SOURCE'],low_memory=False)
    assert source.fight_id.is_unique
    assert not source.groupby('URL')[['METHOD','ROUND','TIME','TIME FORMAT']].nunique(dropna=False).gt(1).any().any()
    methods = source.set_index('fight_id').loc[frame.fight_id,'METHOD'].map(M['V2']['AUDIT']['normalize_method']).reset_index(drop=True)
    supported = methods.notna()
    candidates = frame.outcome.map({'fighter_1_win':1,'fighter_2_win':2})
    assert frame.winner.eq(frame.fighter_1.where(candidates.eq(1),frame.fighter_2)).all()
    wx = frame[wf]
    mx = M['candidate_features'](frame.loc[supported],M['groups']()['METHOD-M'],candidates.loc[supported])
    my = methods.loc[supported].map({'KO/TKO':0,'Submission':1,'Decision':2})
    assert len(wx)==8547 and len(mx)==8524
    # Verify enriched source aliases also agree, beyond the original baseline checks.
    checked = [c for c in data if c not in ('fight_id','EVENT','BOUT')]
    assert not data.groupby('URL')[checked].nunique(dropna=False).gt(1).any().any()
    builder = INFERENCE['V2MatchupFeatureBuilder']()
    reproduction = INFERENCE['validate_historical'](builder,160)
    trained = {}
    for kind,x,y,features,report,configuration in [
        ('winner',wx,binary.target,wf,wr,'D'),('method',mx,my,mf,mr,'METHOD-M')]:
        assert list(x)==features and not np.isinf(x.to_numpy()).any()
        xa = pd.concat([x,-x],ignore_index=True) if kind=='winner' else x
        ya = pd.concat([y,1-y],ignore_index=True) if kind=='winner' else y
        if kind=='winner':
            np.testing.assert_allclose(xa.iloc[len(x):],-xa.iloc[:len(x)],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(ya.iloc[len(x):],1-ya.iloc[:len(x)])
        pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True)),
                             ('scaler',StandardScaler(with_mean=False)),
                             ('logistic',LogisticRegression(max_iter=2000,solver='lbfgs'))])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            pipeline.fit(xa,ya)
        np.testing.assert_array_equal(pipeline.named_steps['imputer'].statistics_,np.zeros(len(features)))
        np.testing.assert_allclose(pipeline.named_steps['scaler'].var_,xa.fillna(0).var(ddof=0),atol=1e-12)
        assert np.all(pipeline.named_steps['scaler'].n_samples_seen_==len(xa))
        dates = frame.event_date if kind=='winner' else frame.loc[supported,'event_date']
        metadata = {'model_id':ARTIFACTS[kind], 'version':'2.0.0' if kind=='winner' else '1.0.0',
                    'configuration':configuration,'model_type':'LogisticRegression','python_version':platform.python_version(),
                    'sklearn_version':sklearn.__version__,'created_utc':datetime.now(timezone.utc).isoformat(),
                    'training_start':str(dates.min().date()),'training_cutoff':str(dates.max().date()),
                    'original_training_fights':len(x),'fitted_training_rows':len(xa),'symmetric_augmentation':kind=='winner',
                    'filtering':filtering,'excluded_unsupported_methods':int((~supported).sum()) if kind=='method' else 0,
                    'features':features,'feature_count':len(features),'classes':pipeline.classes_.tolist(),
                    'class_labels':{'0':'fighter_b_win','1':'fighter_a_win'} if kind=='winner' else dict(enumerate(M['CLASSES'])),
                    'original_target_counts':{str(k):int(v) for k,v in y.value_counts().items()},
                    'target':'Binary winner, fighter_a minus fighter_b' if kind=='winner' else 'Method conditional on candidate winning a supported-method fight',
                    'preprocessing':'SimpleImputer(constant, fill_value=0, keep_empty_features=True); StandardScaler(with_mean=False)',
                    'missing_policy':'Zero imputed difference means unknown relative advantage, not zero physical or performance measurements.',
                    'parameters':pipeline.named_steps['logistic'].get_params(),
                    'input_sha256':sha(ROOT/'data/processed/prefight_features_v2.csv'),
                    'history_sha256':sha(M['V2']['SOURCE']),
                    'inference_builder_sha256':sha(ROOT/'src/prediction/build_v2_matchup_features.py'),
                    'reference_backtest':{'path':'models/v2_winner_backtest_metrics.json' if kind=='winner' else 'models/method_backtest_metrics.json',
                                          'configuration':configuration,'aggregate':report['aggregate'][configuration]},
                    'compatibility_reproduction':reproduction[kind],
                    'limitations':['Observed dated UFC history only; names may split/conflate identity; missing statistics persist.',
                                   'Same-day histories excluded; Elo updates batched; source aliases count once.',
                                   'Retrospective outcome corrections lack publication timestamps.',
                                   'All-history fit is not an as-of historical model. Historical feature cutoff does not undo later training outcomes.',
                                   'Backtests are exploratory historical evidence, not evaluation of this final fitted artifact; no training accuracy computed.',
                                   'Winner population includes DQs; method population excludes them. Joint composition approximates the supported-outcome population.',
                                   'Winner D retains evaluated collinearity; coefficients are not causal.' if kind=='winner' else 'Low submission recall, Decision-heavy predictions, imperfect calibration; opposing conditional distributions are not complements.']}
        trained[kind]=(pipeline,metadata)
    assert hashes=={p:sha(p) for p in protected}
    for kind,(pipeline,metadata) in trained.items():
        path=ROOT/'models'/f'{ARTIFACTS[kind]}.joblib'
        joblib.dump(pipeline,path)
        restored=joblib.load(path)
        np.testing.assert_array_equal(restored.named_steps['logistic'].coef_,pipeline.named_steps['logistic'].coef_)
        metadata['model_sha256']=sha(path)
        (ROOT/'models'/f'{ARTIFACTS[kind]}_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(kind,metadata['original_training_fights'],metadata['fitted_training_rows'],metadata['training_cutoff'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
