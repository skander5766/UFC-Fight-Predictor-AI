"""One chronological logistic-regression baseline; no hyperparameter search.

Test features are rolling pre-fight histories: earlier test-period fights may
inform later test fights, but the fitted pipeline is never updated on test data.
"""

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/prefight_features.csv"
MODEL = ROOT / "models/logistic_baseline.joblib"
REPORT = ROOT / "models/logistic_baseline_metrics.json"

# Explicit allowlist from build_prefight_features.py. Merely being numerical or
# ending in '_diff' does not make an arbitrary input column a safe predictor.
METRICS = ["sig_str_landed", "sig_str_attempted", "total_str_landed", "kd",
           "td_landed", "td_attempted", "sub_att", "ctrl_seconds", "head_landed",
           "body_landed", "leg_landed", "distance_landed", "clinch_landed", "ground_landed"]
HISTORY = ["previous_fights", "previous_wins", "previous_losses", "previous_draws",
           "previous_no_contests", "win_rate", "wins_last_3", "wins_last_5",
           "current_win_streak", "current_loss_streak", "days_since_last_fight",
           "previous_fights_with_stats", "stats_coverage"]
FEATURES = ([name + "_diff" for name in HISTORY]
            + [name + "_per_fight_diff" for name in METRICS]
            + ["sig_str_accuracy_diff", "td_accuracy_diff"]
            + [name + "_history_count_diff" for name in METRICS])
FORBIDDEN = {"winner", "outcome", "fighter_1", "fighter_2", "event_date", "EVENT",
             "BOUT", "fight_id", "URL", "METHOD", "ROUND", "TIME", "source_url_reused"}


def load_binary_fights() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(INPUT, low_memory=False)
    counts = {"input_rows": len(frame)}
    if not frame.outcome.isin(["fighter_1_win", "fighter_2_win", "draw", "no_contest"]).all():
        raise ValueError("Unexpected outcome in input.")
    binary = frame.outcome.isin(["fighter_1_win", "fighter_2_win"])
    counts["excluded_draws_no_contests"] = int((~binary).sum())
    frame = frame.loc[binary].copy()
    frame["event_date"] = pd.to_datetime(frame.event_date, format="%Y-%m-%d", errors="raise")
    if frame.event_date.isna().any() or not frame.fight_id.is_unique or frame.URL.isna().any():
        raise ValueError("Invalid dates or fight identity.")
    frame["target"] = frame.outcome.map({"fighter_1_win": 1, "fighter_2_win": 0})
    selected = frame[FEATURES]  # Fail immediately if any allowlisted feature is absent.
    if set(FEATURES) & FORBIDDEN or len(FEATURES) != len(set(FEATURES)):
        raise AssertionError("Unsafe or duplicated feature name.")
    if not all(pd.api.types.is_numeric_dtype(selected[c]) for c in FEATURES):
        raise ValueError("All selected pre-fight features must be numeric.")
    if np.isinf(selected.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values.")
    for url, group in frame[frame.URL.duplicated(keep=False)].groupby("URL"):
        checked = ["event_date", "target", "fighter_1", "fighter_2", *FEATURES]
        if group[checked].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting duplicate source fight: {url}")
    counts["duplicate_source_rows_removed"] = int(frame.URL.duplicated().sum())
    # Same source bout should not get extra weight merely because it has aliases.
    frame = frame.sort_values(["event_date", "fight_id"]).drop_duplicates("URL").reset_index(drop=True)
    counts["binary_unique_source_fights"] = len(frame)
    return frame, counts


def split_by_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = frame.groupby("event_date", sort=True).size()
    # Choose the whole-date boundary nearest 80%; never split an event date.
    cumulative = counts.cumsum().iloc[:-1]
    if cumulative.empty:
        raise ValueError("At least two distinct dates are required.")
    final_train_date = (cumulative - len(frame) * 0.8).abs().idxmin()
    train = frame.loc[frame.event_date <= final_train_date].copy()
    test = frame.loc[frame.event_date > final_train_date].copy()
    assert train.event_date.max() < test.event_date.min()
    assert set(train.fight_id).isdisjoint(test.fight_id)
    assert set(train.URL).isdisjoint(test.URL)
    if train.target.nunique() != 2 or test.target.nunique() != 2:
        raise ValueError("Both partitions must contain both target classes.")
    return train, test


def evaluate(target: pd.Series, probability: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(target, predictions, labels=[0, 1],
            target_names=["fighter_2_win", "fighter_1_win"], output_dict=True, zero_division=0),
    }


def main() -> None:
    frame, counts = load_binary_fights()
    train, test = split_by_date(frame)
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train.target, test.target
    if x_train.isna().all().any():
        raise ValueError("An allowlisted feature has no training observations for median imputation.")
    print("Input filtering:", counts)
    split = {}
    for name, data in (("train", train), ("test", test)):
        split[name] = {"rows": len(data), "start": data.event_date.min().date().isoformat(),
                       "end": data.event_date.max().date().isoformat(),
                       "fighter_1_win_rate": float(data.target.mean())}
        print(f"{name}: {split[name]}")
    print(f"Final feature list ({len(FEATURES)}):\n" + "\n".join(FEATURES))
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2000)),
    ])
    # This is the only fit call. No test data enters imputation, scaling or fit.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x_train, y_train)
    imputer, scaler = pipeline.named_steps["imputer"], pipeline.named_steps["scaler"]
    np.testing.assert_allclose(imputer.statistics_, x_train.median().to_numpy())
    transformed_train = imputer.transform(x_train)
    np.testing.assert_allclose(scaler.mean_, transformed_train.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, transformed_train.var(axis=0))
    assert np.all(scaler.n_samples_seen_ == len(train))
    before_prediction = joblib.hash(pipeline)
    probability = pipeline.predict_proba(x_test)[:, 1]
    predictions = pipeline.predict(x_test)
    assert before_prediction == joblib.hash(pipeline)
    scores = {"logistic_regression": evaluate(y_test, probability, predictions)}
    majority = int(y_train.value_counts().idxmax())
    majority_predictions = np.full(len(test), majority)
    scores["majority_training_class"] = evaluate(y_test, majority_predictions.astype(float), majority_predictions)
    # At exactly 0.5 the deterministic classification tie rule is class 1.
    # Probability metrics are the meaningful evaluation of the 50/50 baseline.
    scores["fifty_fifty"] = evaluate(y_test, np.full(len(test), 0.5), np.ones(len(test), dtype=int))
    # A training-prior constant gives a less extreme probability comparator
    # than the majority classifier's hard 0/1 probabilities.
    scores["training_prior"] = evaluate(y_test, np.full(len(test), y_train.mean()), majority_predictions)
    comparisons = {}
    for name, values in scores.items():
        print(f"\n{name}:\n" + json.dumps(values, indent=2))
        if name != "logistic_regression":
            comparisons[name] = {metric: bool(scores["logistic_regression"][metric] < values[metric]
                if metric in ("log_loss", "brier_score") else scores["logistic_regression"][metric] > values[metric])
                for metric in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score")}
    observed, predicted = calibration_curve(y_test, probability, n_bins=10, strategy="uniform")
    calibration = {"strategy": "uniform", "n_bins": 10,
                   "mean_predicted_probability": float(probability.mean()),
                   "actual_fighter_1_win_rate": float(y_test.mean()),
                   "nonempty_bins": [{"mean_predicted": float(p), "observed_win_rate": float(o)}
                                     for p, o in zip(predicted, observed)]}
    print("\nCalibration diagnostics:\n" + json.dumps(calibration, indent=2))
    coefficients = pd.Series(pipeline.named_steps["logistic"].coef_[0], index=FEATURES)
    strongest = {"positive": coefficients[coefficients > 0].nlargest(15).to_dict(),
                 "negative": coefficients[coefficients < 0].nsmallest(15).to_dict()}
    print("\nStrongest standardized coefficients:\n" + json.dumps(strongest, indent=2))
    print("Coefficients are conditional model associations, not causal effects; correlated features can make individual signs unstable.")
    validation = {"strict_chronological_split": True, "same_date_not_split": True,
                  "no_fight_id_overlap": True, "no_source_url_overlap": True,
                  "explicit_historical_feature_allowlist": True,
                  "imputer_matches_training_medians": True, "scaler_matches_training_statistics": True,
                  "training_only_fit": True, "prediction_did_not_change_pipeline": True}
    report = {"input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
              "sklearn_version": sklearn.__version__, "input_filtering": counts, "split": split,
              "feature_count": len(FEATURES), "features": FEATURES, "metrics": scores,
              "strict_improvement_over_baselines": comparisons, "calibration": calibration,
              "strongest_coefficients": strongest, "validation": validation,
              "majority_class": majority, "iterations": pipeline.named_steps["logistic"].n_iter_.tolist(),
              "notes": ["No tuning or calibration model; test data used only for evaluation.",
                        "50/50 classification ties resolve to fighter_1; it is not a random classifier.",
                        "Majority baseline assigns hard 0/1 probabilities; training_prior uses training prevalence.",
                        "Histories reflect observed dated records, with known name-identity limitations.",
                        "Test histories may include earlier test fights, representing rolling pre-fight prediction."]}
    joblib.dump(pipeline, MODEL)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.testing.assert_allclose(joblib.load(MODEL).predict_proba(x_test)[:, 1], probability)
    print("\nLeakage validation:", validation)
    print("Strict improvement over each baseline:", comparisons)
    print(f"Saved: {MODEL}\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
