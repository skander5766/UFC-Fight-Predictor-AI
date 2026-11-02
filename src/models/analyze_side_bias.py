"""Read-only ordering and swap diagnostics for the saved logistic baseline.

No fitting, calibration, dataset export or model export occurs here.
"""

import hashlib
import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"win_rate": "historical_win_rate", "sig_str_accuracy": "historical_sig_str_accuracy",
           "td_accuracy": "historical_td_accuracy"}
SIDE_METRICS = ["previous_fights", "previous_wins", "historical_win_rate",
                "wins_last_3", "wins_last_5", "sig_str_landed_per_fight",
                "td_landed_per_fight", "ctrl_seconds_per_fight"]


def normalize(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()


def distribution(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = frame.groupby(keys, dropna=False).target.agg(fights="size", fighter_1_wins="sum", fighter_1_win_rate="mean")
    table["fighter_2_wins"] = table.fights - table.fighter_1_wins
    return table


def outcome_analysis(frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\nOUTCOME DISTRIBUTIONS")
    for label, data in (("overall", frame), ("training", train), ("test", test)):
        print(f"{label}: n={len(data)}, fighter_1 wins={data.target.sum()}, "
              f"fighter_2 wins={(1-data.target).sum()}, fighter_1 win rate={data.target.mean():.6f}, "
              f"dates={data.event_date.min().date()} to {data.event_date.max().date()}")
    print("\nBy year:\n" + distribution(frame.assign(year=frame.event_date.dt.year), ["year"]).to_string())
    early = frame.loc[frame.event_date.dt.year <= 2009]
    print(f"1994-2009: {early.target.sum()}/{len(early)} first-listed fighters won.")
    # Combine standard title/non-title descriptions for the same weight class.
    # Unusual historical tournament descriptions remain separate, not guessed.
    weight = normalize(frame.WEIGHTCLASS).str.replace(r"^ufc\s+", "", regex=True)
    weight = weight.str.replace(r"(?: title)? bout$", "", regex=True)
    print("\nBy weight class (title/bout descriptors removed):\n" + distribution(frame.assign(weight=weight), ["weight"]).to_string())


def side_analysis(train: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = []
    for period, frame in (("overall", pd.concat([train, test])), ("training", train), ("test", test)):
        for metric in SIDE_METRICS:
            first, second = frame[f"fighter_1_{metric}"], frame[f"fighter_2_{metric}"]
            paired = first.notna() & second.notna()
            difference = first[paired] - second[paired]
            rows.append({"period": period, "feature": metric, "paired_n": int(paired.sum()),
                         "fighter_1_mean": first[paired].mean(), "fighter_2_mean": second[paired].mean(),
                         "mean_difference": difference.mean(), "median_difference": difference.median(),
                         "fighter_1_greater_pct": 100 * difference.gt(0).mean(),
                         "tie_pct": 100 * difference.eq(0).mean(),
                         "missing_1": int(first.isna().sum()), "missing_2": int(second.isna().sum())})
    print("\nPAIRED PRE-FIGHT COMPARISONS (higher volume is not necessarily better skill)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def ordering_analysis(frame: pd.DataFrame) -> None:
    raw = pd.read_csv(ROOT / "data/raw/ufc_fight_results.csv")
    for col in ("EVENT", "BOUT", "URL", "OUTCOME"):
        raw[col] = normalize(raw[col])
    current = frame.copy()
    for col in ("EVENT", "BOUT", "URL"):
        current[col] = normalize(current[col])
    joined = current.merge(raw[["EVENT", "BOUT", "URL", "OUTCOME"]],
                           on=["EVENT", "BOUT", "URL"], how="left", validate="one_to_one")
    if joined.OUTCOME.isna().any():
        raise ValueError("Some input fights cannot be verified against raw results.")
    parts = joined.BOUT.str.split(r"\s+vs\.\s+", regex=True)
    first_matches = normalize(joined.fighter_1).eq(parts.str[0])
    second_matches = normalize(joined.fighter_2).eq(parts.str[1])
    expected = joined.OUTCOME.map({"w/l": 1, "l/w": 0})
    print("\nRAW/RESULT ORDERING")
    print(f"fighter_1 equals first BOUT name: {first_matches.sum()}/{len(joined)}")
    print(f"fighter_2 equals second BOUT name: {second_matches.sum()}/{len(joined)}")
    print(f"OUTCOME decoded consistently with target: {expected.eq(joined.target).sum()}/{len(joined)}")
    print("Raw binary outcome counts:\n" + joined.OUTCOME.value_counts().to_string())
    # The independent dates CSV can demonstrate its ordering, but not why its
    # upstream collector chose that ordering. Compare only exact name pairs.
    dates = pd.read_csv(ROOT / "data/raw/fights.csv")
    comparison = frame.merge(dates[["fight_url", "fighter_1", "fighter_2"]],
                             left_on="URL", right_on="fight_url", suffixes=("", "_raw"),
                             how="left", validate="one_to_one")
    a, b = normalize(comparison.fighter_1), normalize(comparison.fighter_2)
    ra, rb = normalize(comparison.fighter_1_raw), normalize(comparison.fighter_2_raw)
    same = a.eq(ra) & b.eq(rb)
    reverse = a.eq(rb) & b.eq(ra)
    comparable = (same | reverse).fillna(False)
    print(f"fights.csv same order={same.sum()}, reversed={reverse.sum()}, "
          f"unresolved names/missing URL={(~comparable).sum()}")
    raw_first_won = normalize(comparison.winner).eq(ra)
    print(f"fights.csv first-listed fighter is the winner: "
          f"{raw_first_won[comparable].sum()}/{comparable.sum()} ({raw_first_won[comparable].mean():.4%})")
    print("Comparable exceptions to winner-first fights.csv ordering:\n" +
          comparison.loc[comparable & ~raw_first_won.fillna(False),
                         ["EVENT", "BOUT", "winner", "fighter_1_raw", "fighter_2_raw", "URL"]].to_string(index=False))
    print("Code evidence: preprocess_fights.py parses BOUT order directly and decodes W/L or L/W relative to it.")
    print("add_fight_dates.py preserves those names/order; it does not replace them with fights.csv ordering.")
    print("Upstream collection rules, ranking, favorite status and corner assignments are not established by these files.")


def swapped_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    swapped = pd.DataFrame(index=frame.index)
    failures = []
    for feature in features:
        # Validate semantics from the component columns, not just the suffix.
        if not feature.endswith("_diff"):
            failures.append((feature, "non-difference feature needs an explicit swap rule"))
            continue
        base = feature[:-5]
        base = ALIASES.get(base, base)
        first, second = f"fighter_1_{base}", f"fighter_2_{base}"
        if first not in frame or second not in frame:
            failures.append((feature, "missing per-fighter components"))
            continue
        expected = frame[first] - frame[second]
        if not np.allclose(frame[feature], expected, equal_nan=True, rtol=1e-10, atol=1e-10):
            failures.append((feature, "stored value is not fighter_1 minus fighter_2"))
            continue
        swapped[feature] = frame[second] - frame[first]
        np.testing.assert_allclose(swapped[feature], -frame[feature], equal_nan=True, rtol=1e-10, atol=1e-10)
    print(f"\nAntisymmetry failures: {failures}")
    if failures:
        raise ValueError("Cannot perform a valid swap without resolving feature semantics.")
    print(f"All {len(features)} selected features verified against original per-fighter components; missing stays missing.")
    return swapped[features]


def model_analysis(pipeline, train, test, swapped, features, evaluate) -> None:
    original = test[features]
    state = joblib.hash(pipeline)
    p = pipeline.predict_proba(original)[:, 1]
    q = pipeline.predict_proba(swapped.loc[test.index])[:, 1]
    sym = (p + 1 - q) / 2
    error = np.abs(p + q - 1)
    print("\nSWAP INVARIANCE")
    print(json.dumps({"mean_symmetry_error": float(error.mean()), "median_symmetry_error": float(np.median(error)),
                      "max_symmetry_error": float(error.max()), "percentage_over_0.05": float(100 * (error > .05).mean())}, indent=2))
    print("\nORIGINAL VS SYMMETRIZED")
    for name, probability in (("original", p), ("symmetrized", sym)):
        scores = evaluate(test.target, probability, (probability > .5).astype(int))
        compact = {key: scores[key] for key in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score")}
        compact["mean_predicted_fighter_1_probability"] = float(probability.mean())
        print(name, json.dumps(compact, indent=2))
    print(f"Actual test fighter_1 win rate: {test.target.mean():.8f}")
    classifier = pipeline.named_steps["logistic"]
    scaler, imputer = pipeline.named_steps["scaler"], pipeline.named_steps["imputer"]
    weights = classifier.coef_[0] / scaler.scale_
    intercept = float(classifier.intercept_[0])
    centered_intercept = intercept - float(weights @ scaler.mean_)
    # For observed difference x: logit = beta*x + b - beta*training_mean.
    # Missing x is filled with the SAME training median in both orientations.
    missing_offsets = original.isna().to_numpy() @ (weights * imputer.statistics_)
    effective = centered_intercept + missing_offsets
    logits = pipeline.decision_function(original)
    swapped_logits = pipeline.decision_function(swapped.loc[test.index])
    np.testing.assert_allclose((logits + swapped_logits) / 2, effective, atol=1e-10)
    print("\nINTERCEPT AND PREPROCESSING")
    print(f"Logistic intercept (standardized coordinates): {intercept:.8f}")
    print(f"Effective intercept for complete raw differences: {centered_intercept:.8f}")
    print(f"Probability at zero raw differences: {1/(1+np.exp(-centered_intercept)):.8f}")
    print(f"Training class-prior log odds: {np.log(train.target.mean()/(1-train.target.mean())):.8f}")
    print(f"Effective row intercept min/mean/max: {effective.min():.8f}/{effective.mean():.8f}/{effective.max():.8f}")
    nonzero = ~np.isclose(imputer.statistics_, 0, atol=1e-12)
    print("Nonzero imputation medians (potential preprocessing asymmetry):",
          dict(zip(np.array(features)[nonzero], imputer.statistics_[nonzero])))
    complete = ~original.isna().any(axis=1).to_numpy()
    for label, mask in (("complete", complete), ("missing", ~complete)):
        print(f"{label} rows={mask.sum()}, mean symmetry error={error[mask].mean() if mask.any() else np.nan:.8f}")
    transformed = scaler.transform(imputer.transform(original))
    transformed_swap = scaler.transform(imputer.transform(swapped.loc[test.index]))
    broken = np.max(np.abs(transformed + transformed_swap), axis=0) > 1e-10
    print("Features no longer antisymmetric AFTER fitted imputation/scaling:", np.array(features)[broken].tolist())
    assert state == joblib.hash(pipeline)
    print("Demonstrated: raw feature differences are antisymmetric; the fitted affine pipeline need not be.")
    print("The logit-pair offset identity above isolates intercept/centering/imputation contributions exactly.")
    print("Class imbalance and unequal historical profiles are descriptive evidence, not proof of the upstream ordering rule.")
    print("Symmetrization is an exploratory test-set diagnostic, not a newly trained or validated production model.")


def main() -> None:
    # run_path loads existing functions without running main or writing pycache.
    baseline = runpy.run_path(str(Path(__file__).with_name("train_baseline.py")))
    paths = [p for folder in (ROOT / "data", ROOT / "models") for p in folder.rglob("*") if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frame, counts = baseline["load_binary_fights"]()
    train, test = baseline["split_by_date"](frame)
    report = json.loads(baseline["REPORT"].read_text(encoding="utf-8"))
    assert report["input_sha256"] == before[baseline["INPUT"]]
    assert counts == report["input_filtering"]
    for name, data in (("train", train), ("test", test)):
        assert len(data) == report["split"][name]["rows"]
        assert str(data.event_date.min().date()) == report["split"][name]["start"]
        assert str(data.event_date.max().date()) == report["split"][name]["end"]
    pipeline = joblib.load(baseline["MODEL"])
    features = list(pipeline.feature_names_in_)
    assert features == baseline["FEATURES"] == report["features"]
    print("Exact baseline filtering and split verified:", counts)
    outcome_analysis(frame, train, test)
    side_analysis(train, test)
    ordering_analysis(frame)
    swapped = swapped_features(frame, features)
    model_analysis(pipeline, train, test, swapped, features, baseline["evaluate"])
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print("\nRead-only validation PASSED: all existing dataset/model/report bytes unchanged; no model fitted.")


if __name__ == "__main__":
    main()
