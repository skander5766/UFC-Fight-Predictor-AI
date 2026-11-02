"""Chronological logistic baseline with training-only orientation augmentation."""

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


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/logistic_symmetric.joblib"
REPORT = ROOT / "models/logistic_symmetric_metrics.json"


def augment_training(train: pd.DataFrame, features: list[str]):
    """Keep source provenance separately; identifiers are never predictors."""
    x = train[features].reset_index(drop=True)
    y = train.target.reset_index(drop=True)
    augmented_x = pd.concat([x, -x], ignore_index=True)
    augmented_y = pd.concat([y, 1 - y], ignore_index=True)
    provenance = pd.concat([train[["fight_id", "URL", "event_date"]]] * 2, ignore_index=True)
    n = len(train)
    np.testing.assert_allclose(augmented_x.iloc[n:], -augmented_x.iloc[:n], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(augmented_y.iloc[n:], 1 - augmented_y.iloc[:n])
    assert augmented_y.value_counts().to_dict() == {0: n, 1: n}
    assert provenance.fight_id.value_counts().eq(2).all()
    return augmented_x, augmented_y, provenance


def main() -> None:
    # Reuse functions without running their main blocks or writing bytecode.
    baseline = runpy.run_path(str(Path(__file__).with_name("train_baseline.py")))
    analysis = runpy.run_path(str(Path(__file__).with_name("analyze_side_bias.py")))
    protected = [p for directory in (ROOT / "data", ROOT / "models")
                 for p in directory.rglob("*") if p.is_file() and p not in (MODEL, REPORT)]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    frame, filtering = baseline["load_binary_fights"]()
    train, test = baseline["split_by_date"](frame)
    features = baseline["FEATURES"]
    previous = json.loads(baseline["REPORT"].read_text(encoding="utf-8"))
    assert hashes[baseline["INPUT"]] == previous["input_sha256"]
    assert filtering == previous["input_filtering"]
    assert features == previous["features"] and len(features) == 43
    swapped = analysis["swapped_features"](frame, features)
    split = {}
    for label, data in (("train", train), ("test", test)):
        split[label] = {"rows": len(data), "start": str(data.event_date.min().date()),
                        "end": str(data.event_date.max().date())}
        for field, value in split[label].items():
            assert value == previous["split"][label][field]
    test_before = joblib.hash(test)
    # Augmentation occurs strictly AFTER the original chronological split.
    x_aug, y_aug, provenance = augment_training(train, features)
    assert set(provenance.fight_id) == set(train.fight_id)
    assert set(provenance.fight_id).isdisjoint(test.fight_id)
    assert set(provenance.URL).isdisjoint(test.URL)
    assert provenance.event_date.max() < test.event_date.min()
    assert not set(features) & baseline["FORBIDDEN"]
    print("Filtering:", filtering)
    print("Chronological split:", split)
    print(f"Original training fights: {len(train)}; augmented rows: {len(x_aug)}; test fights: {len(test)}")
    print("Augmented target distribution:", y_aug.value_counts().to_dict())
    print(f"Selected features ({len(features)}):\n" + "\n".join(features))

    # Zero means an unknown relative advantage, NOT zero fighter performance.
    # NaN remains NaN when swapped, so a shared nonzero fill would violate odd
    # symmetry. Zero fill and division by a fixed scale preserve it exactly.
    # No centering or missingness indicator with even swap semantics is added.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("scaler", StandardScaler(with_mean=False)),
        ("logistic", LogisticRegression(max_iter=2000)),
    ])
    # Leave the intercept trainable: balanced paired rows should make it zero.
    # This is the only fit call. No hyperparameter selection uses test results.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x_aug, y_aug)
    imputer, scaler = pipeline.named_steps["imputer"], pipeline.named_steps["scaler"]
    filled = imputer.transform(x_aug)
    np.testing.assert_array_equal(imputer.statistics_, np.zeros(len(features)))
    np.testing.assert_allclose(scaler.var_, filled.var(axis=0), atol=1e-12)
    assert np.all(scaler.n_samples_seen_ == len(x_aug))
    transformed = scaler.transform(filled)
    np.testing.assert_allclose(transformed[len(train):], -transformed[:len(train)], rtol=0, atol=0)
    x_test = test[features]
    x_swap = swapped.loc[test.index]
    np.testing.assert_allclose(pipeline[:-1].transform(x_swap), -pipeline[:-1].transform(x_test), rtol=0, atol=1e-12)
    fitted_state = joblib.hash(pipeline)
    p = pipeline.predict_proba(x_test)[:, 1]
    q = pipeline.predict_proba(x_swap)[:, 1]
    errors = np.abs(p + q - 1)
    symmetry = {"mean": float(errors.mean()), "median": float(np.median(errors)),
                "maximum": float(errors.max()), "percentage_over_0.05": float(100 * (errors > .05).mean())}
    assert symmetry["maximum"] < 1e-5, "Paired training failed the symmetry tolerance."
    intercept = float(pipeline.named_steps["logistic"].intercept_[0])
    assert abs(intercept) < 1e-5
    assert joblib.hash(test) == test_before and joblib.hash(pipeline) == fitted_state
    old_pipeline = joblib.load(baseline["MODEL"])
    old_p = old_pipeline.predict_proba(x_test)[:, 1]
    majority = int(train.target.value_counts().idxmax())
    probabilities = {"symmetric_logistic": p, "original_logistic": old_p,
                     "majority_training_class": np.full(len(test), float(majority)),
                     "fifty_fifty": np.full(len(test), .5)}
    scores = {}
    for name, probability in probabilities.items():
        if name == "symmetric_logistic":
            predicted = pipeline.predict(x_test)
        elif name == "original_logistic":
            predicted = old_pipeline.predict(x_test)
        else:
            predicted = (probability >= .5).astype(int)
        scores[name] = baseline["evaluate"](test.target, probability, predicted)
        scores[name]["mean_predicted_fighter_1_probability"] = float(probability.mean())
        print(f"\n{name}:\n" + json.dumps(scores[name], indent=2))
    for metric in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score"):
        np.testing.assert_allclose(scores["original_logistic"][metric], previous["metrics"]["logistic_regression"][metric])
    coefficients = pd.Series(pipeline.named_steps["logistic"].coef_[0], index=features)
    strongest = {"positive": coefficients[coefficients > 0].nlargest(15).to_dict(),
                 "negative": coefficients[coefficients < 0].nsmallest(15).to_dict()}
    print("\nSymmetry errors:", symmetry)
    print(f"Trainable intercept after paired augmentation: {intercept:.12g}")
    print("Zero raw differences now imply 50% probability; the artificial orientation intercept is removed.")
    print("This does not force the average probability across unequal test matchups to equal 50%.")
    print("\nStrongest coefficients (associations, not causal effects):\n" + json.dumps(strongest, indent=2))
    validation = {"same_baseline_filter_and_split": True, "strict_date_separation": True,
                  "augmentation_training_only": True, "no_test_ids_in_training": True,
                  "no_source_url_overlap": True, "historical_feature_allowlist": True,
                  "difference_component_validation": True, "opposite_paired_targets": True,
                  "exact_negative_paired_features": True, "balanced_augmented_targets": True,
                  "training_only_preprocessing": True, "antisymmetric_preprocessing": True,
                  "original_test_not_augmented": True, "symmetry_tolerance_passed": True}
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    validation["existing_data_and_baseline_files_unchanged"] = True
    report = {"input_sha256": hashes[baseline["INPUT"]], "sklearn_version": sklearn.__version__,
              "filtering": filtering, "split": split, "augmented_training_rows": len(x_aug),
              "augmented_class_counts": {str(k): int(v) for k, v in y_aug.value_counts().items()},
              "feature_count": len(features), "features": features, "metrics": scores,
              "actual_test_fighter_1_win_rate": float(test.target.mean()), "symmetry": symmetry,
              "intercept": intercept, "strongest_coefficients": strongest, "validation": validation,
              "iterations": pipeline.named_steps["logistic"].n_iter_.tolist(),
              "notes": ["Zero imputes unknown relative advantage, not zero fighter performance.",
                        "StandardScaler(with_mean=False) preserves sign reversal.",
                        "Trainable intercept retained; balanced paired augmentation drives it to zero.",
                        "50/50 baseline classification ties select fighter_1, as in the original experiment.",
                        "Majority probabilities are hard 0/1; their log loss is correspondingly extreme.",
                        "Earlier test-period fights can inform later histories; the model is not refitted.",
                        "Observed-history and name-identity limitations persist; symmetry does not repair source data."]}
    joblib.dump(pipeline, MODEL)
    np.testing.assert_allclose(joblib.load(MODEL).predict_proba(x_test)[:, 1], p)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nValidation checks:", validation)
    print(f"Saved: {MODEL}\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
