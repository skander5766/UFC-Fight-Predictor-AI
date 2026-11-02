"""Fixed expanding-window evaluation; no production model is saved.

2025+ is excluded from this run, but was evaluated in earlier experiments.
Historical features use rolling strictly prior dates, including earlier fights
within each test period. Current physical-table temporal limitations persist.
"""
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

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "models/logistic_backtest_metrics.json"
METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score"]
FOLDS = [(2017, 2018), (2019, 2020), (2021, 2022), (2023, 2024)]


def main():
    baseline = runpy.run_path(str(Path(__file__).with_name("train_baseline.py")))
    enriched = runpy.run_path(str(Path(__file__).with_name("train_enriched_symmetric_logistic.py")))
    protected = [p for directory in (ROOT / "data", ROOT / "models") for p in directory.rglob("*")
                 if p.is_file() and p != REPORT]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    # Read dates first, then discard 2025+ before filtering outcomes, inspecting
    # feature values, or constructing model matrices for this experiment.
    frame = pd.read_csv(enriched["INPUT"], low_memory=False)
    frame["event_date"] = pd.to_datetime(frame.event_date, format="%Y-%m-%d", errors="raise")
    assert frame.event_date.notna().all()
    excluded = int(frame.event_date.ge("2025-01-01").sum())
    frame = frame.loc[frame.event_date.lt("2025-01-01")].copy()
    historical = baseline["FEATURES"]
    if not frame.outcome.isin(["fighter_1_win", "fighter_2_win", "draw", "no_contest"]).all():
        raise ValueError("Unexpected outcome.")
    nonbinary = ~frame.outcome.isin(["fighter_1_win", "fighter_2_win"])
    excluded_nonbinary = int(nonbinary.sum())
    frame = frame.loc[~nonbinary].copy()
    assert frame.fight_id.is_unique and frame.URL.notna().all()
    frame["target"] = frame.outcome.map({"fighter_1_win": 1, "fighter_2_win": 0})
    all_features = enriched["design"](frame, historical)
    duplicates = frame.URL.duplicated(keep=False)
    checks = pd.concat([frame[["event_date", "target", "fighter_1", "fighter_2"]], all_features], axis=1)
    if checks.loc[duplicates].groupby(frame.loc[duplicates,"URL"]).nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting duplicate source fights.")
    duplicate_count = int(frame.URL.duplicated().sum())
    frame = frame.sort_values(["event_date", "fight_id"]).drop_duplicates("URL").reset_index(drop=True)
    # Verify identical filtering against the original historical input, using
    # only the pre-2025 rows. No reserved-period features or targets are checked.
    original = pd.read_csv(baseline["INPUT"], low_memory=False)
    original["event_date"] = pd.to_datetime(original.event_date)
    original = original.loc[original.event_date.lt("2025-01-01")]
    original = original.loc[original.outcome.isin(["fighter_1_win", "fighter_2_win"])]
    original = original.sort_values(["event_date", "fight_id"]).drop_duplicates("URL").reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[original.columns], original)
    x = enriched["design"](frame, historical)
    swapped = enriched["design"](frame, historical, swap=True)
    np.testing.assert_allclose(swapped, -x, equal_nan=True, rtol=1e-10, atol=1e-10)
    families = {"A": historical, "C": historical+enriched["ACTIVITY"]+enriched["LAYOFF"],
                "D": historical+enriched["PHYSICAL"]+enriched["ACTIVITY"]+enriched["LAYOFF"]+enriched["STANCE"]}
    assert [len(families[k]) for k in ("A","C","D")] == [43,50,57]
    previous = json.loads(enriched["REPORT"].read_text())
    for name, features in families.items():
        assert features == previous["experiments"][name]["features"]
        assert not set(features) & baseline["FORBIDDEN"]
    rows, details, differences = [], [], []
    pooled = {name: [] for name in families}
    test_ids = set()
    for number, (start, end) in enumerate(FOLDS,1):
        lower, upper = pd.Timestamp(start,1,1), pd.Timestamp(end+1,1,1)
        train = frame.loc[frame.event_date.lt(lower)]
        test = frame.loc[frame.event_date.ge(lower)&frame.event_date.lt(upper)]
        assert len(train) and len(test) and train.target.nunique()==test.target.nunique()==2
        assert train.event_date.max() < test.event_date.min()
        assert set(train.fight_id).isdisjoint(test.fight_id) and set(train.URL).isdisjoint(test.URL)
        assert test_ids.isdisjoint(test.fight_id)
        test_ids.update(test.fight_id)
        fold = {"fold":number,"scheduled_test_years":[start,end],"train_fights":len(train),
                "augmented_train_rows":2*len(train),"test_fights":len(test),
                "train_start":str(train.event_date.min().date()),"train_end":str(train.event_date.max().date()),
                "test_start":str(test.event_date.min().date()),"test_end":str(test.event_date.max().date()),"models":{}}
        print("Fold:", {k:v for k,v in fold.items() if k!='models'}, flush=True)
        for name, features in families.items():
            xt = x.loc[train.index, features].reset_index(drop=True)
            xa = pd.concat([xt,-xt],ignore_index=True)
            yt = train.target.reset_index(drop=True)
            ya = pd.concat([yt,1-yt],ignore_index=True)
            np.testing.assert_allclose(xa.iloc[len(train):],-xa.iloc[:len(train)],equal_nan=True,rtol=0,atol=0)
            np.testing.assert_array_equal(ya.iloc[len(train):],1-ya.iloc[:len(train)])
            assert ya.value_counts().to_dict()=={0:len(train),1:len(train)}
            pipeline = Pipeline([("imputer",SimpleImputer(strategy="constant",fill_value=0,keep_empty_features=True)),
                                 ("scaler",StandardScaler(with_mean=False)),
                                 ("logistic",LogisticRegression(max_iter=2000))])
            with warnings.catch_warnings():
                warnings.simplefilter("error",ConvergenceWarning)
                pipeline.fit(xa,ya)
            filled = pipeline.named_steps['imputer'].transform(xa)
            np.testing.assert_array_equal(pipeline.named_steps['imputer'].statistics_,np.zeros(len(features)))
            np.testing.assert_allclose(pipeline.named_steps['scaler'].var_,filled.var(axis=0),atol=1e-12)
            assert np.all(pipeline.named_steps['scaler'].n_samples_seen_==len(xa))
            state=joblib.hash(pipeline)
            p=pipeline.predict_proba(x.loc[test.index,features])[:,1]
            q=pipeline.predict_proba(swapped.loc[test.index,features])[:,1]
            prediction=pipeline.predict(x.loc[test.index,features])
            error=np.abs(p+q-1)
            assert error.max()<1e-5 and joblib.hash(pipeline)==state
            scores=baseline['evaluate'](test.target,p,prediction)
            fold['models'][name]={"metrics":scores,"symmetry_max":float(error.max()),
                                   "symmetry_mean":float(error.mean()),"intercept":float(pipeline.named_steps['logistic'].intercept_[0])}
            row={"fold":number,"configuration":name,"train_fights":len(train),"test_fights":len(test),
                 **{metric:scores[metric] for metric in METRICS},"symmetry_max":float(error.max())}
            rows.append(row)
            pooled[name].append((test.target.to_numpy(),p,prediction))
            print(row,flush=True)
        for name in ('C','D'):
            delta={m:fold['models'][name]['metrics'][m]-fold['models']['A']['metrics'][m] for m in METRICS}
            differences.append({"fold":number,"comparison":name+'-A',**delta})
        details.append(fold)
    table=pd.DataFrame(rows)
    aggregate={}
    for name in families:
        subset=table.loc[table.configuration.eq(name)]
        y=np.concatenate([entry[0] for entry in pooled[name]])
        p=np.concatenate([entry[1] for entry in pooled[name]])
        pred=np.concatenate([entry[2] for entry in pooled[name]])
        aggregate[name]={"fold_mean":subset[METRICS].mean().to_dict(),
                         "fold_sample_std":subset[METRICS].std(ddof=1).to_dict(),
                         "test_count_weighted_fold_mean":{m:float(np.average(subset[m],weights=subset.test_fights)) for m in METRICS},
                         "pooled_out_of_time_metrics":baseline['evaluate'](pd.Series(y),p,pred),
                         "pooled_test_fights":len(y),"maximum_symmetry_error":float(subset.symmetry_max.max())}
    print("\nAggregate results:",json.dumps(aggregate,indent=2),flush=True)
    print("\nPer-fold deltas:\n"+pd.DataFrame(differences).to_string(index=False),flush=True)
    print("2025-2026 excluded from this run, but NOT a pristine holdout: those dates were evaluated in prior experiments.")
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report={"input_sha256":hashes[enriched['INPUT']],"sklearn_version":sklearn.__version__,
            "filtering":{"excluded_2025_plus_rows":excluded,"excluded_pre2025_draws_no_contests":excluded_nonbinary,
                         "removed_pre2025_duplicate_sources":duplicate_count,"eligible_pre2025_fights":len(frame)},
            "features":families,"folds":details,"aggregate":aggregate,"per_fold_differences":differences,
            "validation":{"strict_chronology":True,"identical_filtering":True,"exact_predefined_features":True,
                          "training_only_augmentation_and_preprocessing":True,"no_id_url_overlap":True,
                          "test_periods_disjoint":True,"no_target_or_current_record_predictors":True,
                          "correct_swapping_all_models":True,"existing_files_unchanged":True},
            "holdout":{"excluded_from_this_run":True,"previously_untouched":False,
                       "reason":"2025-2026 already evaluated in baseline, symmetry, HGB and enrichment experiments."},
            "notes":["No model artifacts saved; all twelve models are evaluation-only.",
                     "Population is deduplicated binary fights with dated observed history.",
                     "Unweighted fold SD uses ddof=1; four folds do not establish statistical significance.",
                     "Weighted fold AUC is not pooled AUC; pooled AUC also compares scores from different fitted models.",
                     "Pooled accuracy/log loss/Brier equal test-size-weighted fold values; pooled balanced accuracy uses pooled class recalls.",
                     "Earlier test fights may inform later pre-fight histories; fitted model remains fixed per fold.",
                     "Current-table physical/stance historical-validity and fighter identity limitations persist.",
                     "Several prior experiments examined these periods, so results are exploratory, not an untouched final evaluation."]}
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Saved:",REPORT,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
