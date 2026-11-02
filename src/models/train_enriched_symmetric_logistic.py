"""Four fixed feature-family ablations; save only predefined experiment D.

Stances use signed one-hot contrasts, never ordinal scores. Invariant main
 effects such as same_stance cannot shift the odds toward either fighter in
 an antisymmetric linear logit, so they are deliberately omitted.
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
INPUT = ROOT / "data/processed/prefight_features_activity.csv"
MODEL = ROOT / "models/logistic_enriched_symmetric.joblib"
REPORT = ROOT / "models/logistic_enriched_symmetric_metrics.json"
PHYSICAL = ["height_diff", "reach_diff"]
ACTIVITY = ["fights_last_365_days_diff", "fights_last_730_days_diff",
            "days_since_second_last_fight_diff", "average_days_between_previous_fights_diff"]
LAYOFF = [f"layoff_over_{n}_days_diff" for n in (365, 730, 1095)]
STANCES = ["Orthodox", "Southpaw", "Switch", "Open Stance", "Sideways"]
STANCE = ["stance_" + s.lower().replace(" ", "_") + "_contrast" for s in STANCES]
ALIASES = {"win_rate": "historical_win_rate", "sig_str_accuracy": "historical_sig_str_accuracy",
           "td_accuracy": "historical_td_accuracy"}


def boolean_values(series):
    values = series.astype("string").str.lower()
    if not values.dropna().isin(["true", "false"]).all():
        raise ValueError("Unexpected layoff indicator encoding.")
    return values.map({"true": 1.0, "false": 0.0}).astype(float)


def design(frame, historical, swap=False):
    """Rebuild from fighter components to implement and audit swap semantics."""
    a, b = (2, 1) if swap else (1, 2)
    output = {}
    for column in historical + PHYSICAL + ACTIVITY + LAYOFF:
        base = ALIASES.get(column[:-5], column[:-5])
        first, second = frame[f"fighter_{a}_{base}"], frame[f"fighter_{b}_{base}"]
        if column in LAYOFF:
            first, second = boolean_values(first), boolean_values(second)
        difference = first - second
        if not swap and column in frame:
            np.testing.assert_allclose(difference, frame[column], equal_nan=True, rtol=1e-10, atol=1e-10)
            difference = frame[column]  # Preserve exact original historical numbers.
        output[column] = difference
    first, second = frame[f"fighter_{a}_stance"], frame[f"fighter_{b}_stance"]
    if not pd.concat([first, second]).dropna().isin(STANCES).all():
        raise ValueError("Unknown stance category; review encoding rather than guess.")
    known = first.notna() & second.notna()
    for stance, column in zip(STANCES, STANCE):
        output[column] = (first.eq(stance).astype(float) - second.eq(stance).astype(float)).where(known)
    return pd.DataFrame(output, index=frame.index)


def main():
    baseline = runpy.run_path(str(Path(__file__).with_name("train_baseline.py")))
    protected = [p for directory in (ROOT / "data", ROOT / "models") for p in directory.rglob("*")
                 if p.is_file() and p not in (MODEL, REPORT)]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    old_report = json.loads((ROOT / "models/logistic_symmetric_metrics.json").read_text())
    base, counts = baseline["load_binary_fights"]()
    train, test = baseline["split_by_date"](base)
    if (len(train), len(test)) != (6839, 1708):
        raise ValueError(f"Split counts changed: {len(train)}, {len(test)}; stopping.")
    assert counts == old_report["filtering"]
    assert hashes[baseline["INPUT"]] == old_report["input_sha256"]
    enriched = pd.read_csv(INPUT, low_memory=False)
    original = pd.read_csv(baseline["INPUT"], low_memory=False)
    pd.testing.assert_frame_equal(enriched[original.columns], original)
    assert enriched.fight_id.is_unique
    frame = enriched.set_index("fight_id").loc[base.fight_id].reset_index()
    frame.index = base.index
    historical = baseline["FEATURES"]
    assert historical == old_report["features"] and len(historical) == 43
    x, swapped = design(frame, historical), design(frame, historical, swap=True)
    np.testing.assert_allclose(swapped, -x, equal_nan=True, rtol=1e-10, atol=1e-10)
    # Check new attributes agree across aliases before reusing the old dedup rule.
    all_x = design(enriched, historical)
    duplicate = enriched.URL.duplicated(keep=False)
    if all_x.loc[duplicate].groupby(enriched.loc[duplicate, "URL"]).nunique(dropna=False).gt(1).any().any():
        raise ValueError("Enriched features disagree across duplicate source records.")
    split = {label: {"rows": len(data), "start": str(data.event_date.min().date()), "end": str(data.event_date.max().date())}
             for label, data in (("train", train), ("test", test))}
    assert split == old_report["split"]
    assert train.event_date.max() < test.event_date.min()
    assert set(train.fight_id).isdisjoint(test.fight_id) and set(train.URL).isdisjoint(test.URL)
    assert not set(x) & baseline["FORBIDDEN"]
    families = {"A": historical, "B": historical + PHYSICAL,
                "C": historical + ACTIVITY + LAYOFF,
                "D": historical + PHYSICAL + ACTIVITY + LAYOFF + STANCE}
    print("Split:", split, flush=True)
    print("Stance: signed one-hot category contrasts; unknown pair -> NaN; no ordinal or invariant stance main effects.")
    print("Missing differences -> zero unknown relative advantage, NOT zero physical measurements; no centering.")
    print("Layoff flags -> fighter_1 minus fighter_2, unknown prior history -> NaN.")
    reports = {}
    final_pipeline = None
    old_pipeline = joblib.load(ROOT / "models/logistic_symmetric.joblib")
    old_p = old_pipeline.predict_proba(x.loc[test.index, historical])[:, 1]
    reference = baseline["evaluate"](test.target, old_p, old_pipeline.predict(x.loc[test.index, historical]))
    metrics = ["accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score"]
    test_hash = joblib.hash(x.loc[test.index])
    for label, features in families.items():
        xt = x.loc[train.index, features].reset_index(drop=True)
        xs = swapped.loc[train.index, features].reset_index(drop=True)
        # Use exact negatives after validating component-based swapped semantics.
        np.testing.assert_allclose(xs, -xt, equal_nan=True, rtol=1e-10, atol=1e-10)
        xa = pd.concat([xt, -xt], ignore_index=True)
        yt = train.target.reset_index(drop=True)
        ya = pd.concat([yt, 1-yt], ignore_index=True)
        assert len(xa) == 13678 and ya.value_counts().to_dict() == {0: 6839, 1: 6839}
        np.testing.assert_allclose(xa.iloc[6839:], -xa.iloc[:6839], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(ya.iloc[6839:], 1-ya.iloc[:6839])
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
                             ("scaler", StandardScaler(with_mean=False)),
                             ("logistic", LogisticRegression(max_iter=2000))])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(xa, ya)
        filled = pipeline.named_steps["imputer"].transform(xa)
        np.testing.assert_array_equal(pipeline.named_steps["imputer"].statistics_, np.zeros(len(features)))
        np.testing.assert_allclose(pipeline.named_steps["scaler"].var_, filled.var(axis=0))
        assert np.all(pipeline.named_steps["scaler"].n_samples_seen_ == 13678)
        state = joblib.hash(pipeline)
        test_x = x.loc[test.index, features]
        p = pipeline.predict_proba(test_x)[:, 1]
        q = pipeline.predict_proba(swapped.loc[test.index, features])[:, 1]
        errors = np.abs(p + q - 1)
        assert errors.max() < 1e-5
        scores = baseline["evaluate"](test.target, p, pipeline.predict(test_x))
        scores["mean_predicted_fighter_1_probability"] = float(p.mean())
        if label == "A":
            np.testing.assert_allclose(p, old_p, rtol=0, atol=1e-10)
        assert joblib.hash(pipeline) == state
        coef = pd.Series(pipeline.named_steps["logistic"].coef_[0], index=features)
        reports[label] = {"feature_count": len(features), "features": features, "metrics": scores,
                          "deltas_vs_saved_baseline": {m: scores[m]-reference[m] for m in metrics},
                          "symmetry": {"mean": float(errors.mean()), "median": float(np.median(errors)),
                                       "maximum": float(errors.max()), "percentage_over_0.05": float(100*(errors>.05).mean())},
                          "intercept": float(pipeline.named_steps["logistic"].intercept_[0]),
                          "coefficients": coef.to_dict(),
                          "missing_train": xt.isna().sum().to_dict(), "missing_test": test_x.isna().sum().to_dict()}
        print(f"Experiment {label}: {len(features)} features", json.dumps(scores, indent=2), flush=True)
        print("Deltas:", reports[label]["deltas_vs_saved_baseline"], "Symmetry:", reports[label]["symmetry"], flush=True)
        if label == "D":
            final_pipeline = pipeline
            print("Strongest positive coefficients:\n" + coef[coef>0].nlargest(15).to_string())
            print("Strongest negative coefficients:\n" + coef[coef<0].nsmallest(15).to_string())
    assert joblib.hash(x.loc[test.index]) == test_hash
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    validation = {"exact_split_and_deduplication": True, "training_only_augmentation_and_preprocessing": True,
                  "no_fight_id_source_url_overlap": True, "no_test_data_fit": True,
                  "original_historical_features_and_targets_preserved": True,
                  "allowlisted_features_no_current_records_age_or_postfight_statistics": True,
                  "component_verified_swap_semantics": True, "opposite_targets_exact_negative_pairs": True,
                  "all_four_models_pass_symmetry": True, "A_reproduces_saved_baseline": True,
                  "existing_datasets_and_models_unchanged": True}
    report = {"input_sha256": hashes[INPUT], "sklearn_version": sklearn.__version__, "filtering": counts,
              "split": split, "augmented_training_rows": 13678, "experiments": reports,
              "saved_experiment": "D", "baseline_metrics": reference, "validation": validation,
              "actual_test_fighter_1_win_rate": float(test.target.mean()),
              "notes": ["Four predeclared family ablations; no tuning, D saved regardless of ranking.",
                        "C and D include the three signed layoff contrasts in the activity family.",
                        "Stance uses five signed category indicators; same_stance and switch_in_matchup omitted because invariant main effects cannot favor a side in an antisymmetric linear model.",
                        "Saved pipeline expects engineered features, not raw per-fighter stance columns; design() constructs them.",
                        "Zero imputation denotes unknown relative advantage; never zero physical measurement. No missingness indicator or scaling centering.",
                        "Current-table height/reach are approximately static; stance can change; historical validity is not guaranteed.",
                        "Repeated test-period comparisons are exploratory; coefficient associations are not causal.",
                        "No age, current career records, injury/retirement inference. Existing identity and coverage limitations remain."]}
    joblib.dump(final_pipeline, MODEL)
    np.testing.assert_allclose(joblib.load(MODEL).predict_proba(x.loc[test.index, families["D"]]),
                               final_pipeline.predict_proba(x.loc[test.index, families["D"]]))
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Validation:", validation, flush=True)
    print(f"Saved: {MODEL}\nSaved: {REPORT}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
