"""One fixed nonlinear experiment with chronological, orientation-aware checks.

The saved pipeline produces RAW probabilities. Symmetrized inference requires
two calls: (predict_proba(X)[:, 1] + 1 - predict_proba(-X)[:, 1]) / 2.
"""

import hashlib
import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/hist_gradient_boosting.joblib"
REPORT = ROOT / "models/hist_gradient_boosting_metrics.json"
# Fixed before evaluation. Early stopping is disabled to avoid its automatic
# random validation split, especially with paired orientations of each fight.
CONFIG = dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
              max_depth=4, min_samples_leaf=30, l2_regularization=1.0,
              early_stopping=False, random_state=42)


def symmetry_summary(original: np.ndarray, swapped: np.ndarray) -> dict:
    error = np.abs(original + swapped - 1)
    return {"mean": float(error.mean()), "median": float(np.median(error)),
            "maximum": float(error.max()), "percentage_over_0.05": float(100 * (error > .05).mean())}


def main() -> None:
    baseline = runpy.run_path(str(Path(__file__).with_name("train_baseline.py")))
    symmetric = runpy.run_path(str(Path(__file__).with_name("train_symmetric_baseline.py")))
    analysis = runpy.run_path(str(Path(__file__).with_name("analyze_side_bias.py")))
    protected = [p for folder in (ROOT / "data", ROOT / "models")
                 for p in folder.rglob("*") if p.is_file() and p not in (MODEL, REPORT)]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    previous = json.loads(symmetric["REPORT"].read_text(encoding="utf-8"))
    frame, filtering = baseline["load_binary_fights"]()
    train, test = baseline["split_by_date"](frame)
    features = baseline["FEATURES"]
    assert (len(train), len(test), len(features)) == (6839, 1708, 43)
    assert filtering == previous["filtering"]
    assert features == previous["features"]
    assert hashes[baseline["INPUT"]] == previous["input_sha256"]
    swapped = analysis["swapped_features"](frame, features)
    split = {}
    for name, data in (("train", train), ("test", test)):
        split[name] = {"rows": len(data), "start": str(data.event_date.min().date()),
                       "end": str(data.event_date.max().date())}
    assert split == previous["split"]
    test_hash = joblib.hash(test)
    x_aug, y_aug, provenance = symmetric["augment_training"](train, features)
    assert len(x_aug) == 13678
    assert provenance.event_date.max() < test.event_date.min()
    assert set(provenance.fight_id) == set(train.fight_id)
    assert set(provenance.fight_id).isdisjoint(test.fight_id)
    assert set(provenance.URL).isdisjoint(test.URL)
    assert not set(features) & baseline["FORBIDDEN"]
    print("Filtering:", filtering, flush=True)
    print("Split:", split, flush=True)
    print("Augmented training rows:", len(x_aug), "Class counts:", y_aug.value_counts().to_dict())
    print("Features:\n" + "\n".join(features))
    print("Fixed configuration:", CONFIG, flush=True)
    pipeline = Pipeline([
        # Neutral relative difference, not an assertion of zero performance.
        ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(**CONFIG)),
    ])
    pipeline.fit(x_aug, y_aug)  # Only fit call; never fit on original/swapped test rows.
    imputer = pipeline.named_steps["imputer"]
    filled = imputer.transform(x_aug)
    np.testing.assert_array_equal(imputer.statistics_, np.zeros(len(features)))
    np.testing.assert_allclose(filled[len(train):], -filled[:len(train)], rtol=0, atol=0)
    classifier = pipeline.named_steps["hist_gradient_boosting"]
    assert not classifier.do_early_stopping_ and classifier.n_iter_ == CONFIG["max_iter"]
    x_test, x_swap = test[features], swapped.loc[test.index]
    np.testing.assert_allclose(imputer.transform(x_swap), -imputer.transform(x_test), rtol=0, atol=1e-12)
    state = joblib.hash(pipeline)
    p = pipeline.predict_proba(x_test)[:, 1]
    q = pipeline.predict_proba(x_swap)[:, 1]
    sym_p = (p + 1 - q) / 2
    sym_q = (q + 1 - p) / 2
    old_pipeline = joblib.load(symmetric["MODEL"])
    old_p = old_pipeline.predict_proba(x_test)[:, 1]
    scores = {}
    for name, probability in (("raw_hist_gradient_boosting", p),
                              ("symmetrized_hist_gradient_boosting", sym_p),
                              ("symmetric_logistic", old_p)):
        scores[name] = baseline["evaluate"](test.target, probability, (probability > .5).astype(int))
        scores[name]["mean_predicted_fighter_1_probability"] = float(probability.mean())
        print(f"\n{name}:\n" + json.dumps(scores[name], indent=2), flush=True)
    for metric in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score"):
        np.testing.assert_allclose(scores["symmetric_logistic"][metric], previous["metrics"]["symmetric_logistic"][metric])
    symmetry = {"raw": symmetry_summary(p, q), "symmetrized": symmetry_summary(sym_p, sym_q)}
    assert symmetry["symmetrized"]["maximum"] < 1e-12
    print("\nSwap diagnostics:", symmetry, flush=True)
    print("Augmentation does not guarantee raw tree symmetry; the two-orientation inference formula does.")
    print("Computing exploratory test permutation importance (raw-model ROC-AUC, 5 repeats)...", flush=True)
    importance = permutation_importance(pipeline, x_test, test.target,
                                        scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1)
    importance_table = pd.DataFrame({"feature": features, "mean_auc_decrease": importance.importances_mean,
                                     "std_auc_decrease": importance.importances_std}).sort_values("mean_auc_decrease", ascending=False)
    print(importance_table.head(15).to_string(index=False), flush=True)
    print("Importance is exploratory, not causal. Correlated features can mask or share importance; no parameters are selected from it.")
    assert joblib.hash(test) == test_hash and joblib.hash(pipeline) == state
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    validation = {"exact_filtering_split_features": True, "strict_chronological_separation": True,
                  "training_only_augmentation": True, "no_test_fights_in_training": True,
                  "no_fight_id_or_source_url_overlap": True, "paired_targets_opposite": True,
                  "paired_features_exact_negatives": True, "legitimate_feature_allowlist": True,
                  "antisymmetric_imputation": True, "no_random_early_stopping_split": True,
                  "test_not_augmented_or_modified": True, "evaluation_did_not_mutate_model": True,
                  "existing_datasets_and_models_unchanged": True}
    report = {"input_sha256": hashes[baseline["INPUT"]], "sklearn_version": sklearn.__version__,
              "filtering": filtering, "split": split, "augmented_training_rows": len(x_aug),
              "feature_count": len(features), "features": features, "configuration": CONFIG,
              "fitted_iterations": int(classifier.n_iter_),
              "trees_per_iteration": int(classifier.n_trees_per_iteration_),
              "metrics": scores, "actual_test_fighter_1_win_rate": float(test.target.mean()),
              "symmetry": symmetry, "validation": validation,
              "permutation_importance": {"exploratory": True, "model": "raw_hist_gradient_boosting",
                  "partition": "original_chronological_test", "scoring": "roc_auc", "n_repeats": 5,
                  "random_state": 42, "features": importance_table.to_dict(orient="records")},
              "notes": ["Saved pipeline returns raw probabilities; symmetrized inference requires evaluating X and -X.",
                        "Zero imputation encodes no known relative advantage, not zero fighter performance.",
                        "No tuning or parameter selection; early stopping disabled to avoid random internal splitting.",
                        "Original test fights only; swapped copies are diagnostic/inference evaluations, not extra test outcomes.",
                        "Earlier test fights may inform later historical features; model remains fixed.",
                        "Repeated use of this test period makes comparisons exploratory; source identity limitations remain."]}
    joblib.dump(pipeline, MODEL)
    np.testing.assert_allclose(joblib.load(MODEL).predict_proba(x_test)[:, 1], p)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Validation checks:", validation)
    print(f"Saved: {MODEL}\nSaved: {REPORT}")


if __name__ == "__main__":
    # Limit thread oversubscription without changing the model configuration.
    with threadpool_limits(limits=2):
        main()
