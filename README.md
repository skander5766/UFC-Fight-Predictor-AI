# UFC Fight Predictor AI

A local Streamlit application for exploring UFC matchup probabilities, built around chronological feature engineering, temporal model evaluation, and explicit uncertainty.

The dashboard combines **Winner D**, a symmetric logistic winner model, **METHOD-M**, a conditional method model, and an **empirical finish-round prior**. It distinguishes a fighter's win probability, method conditional on winning, and overall win-by-method probability.

> **Release status:** source and aggregate evaluation reports are prepared for publication. Datasets, fitted models, runtime metadata, and the round-prior artifact are excluded because redistribution clearance is incomplete. A fresh source checkout cannot run the dashboard or the complete suite without matching authorized local files. No public artifact download, hosted application, or deployment is provided.

## Features

- Weight-class-first selection based on observed historical membership, with searchable fighter selectors and swap controls.
- Explicit three- or five-round schedules and a prediction-date cutoff for fighter histories.
- Red/blue fighter cards, fixed 0–100% probability charts, expandable tables, and a responsive dark layout.
- KO/TKO, submission, and decision probabilities conditional on each fighter winning, plus six joint winner/method outcomes.
- Historical finish-round frequencies showing only the scheduled rounds; these are population estimates, not fighter-specific ML forecasts.
- A visible uncertainty disclaimer and expandable scientific/history warnings.
- Artifact checksum, version, feature-order, class-label, and probability-distribution checks.

The catalog represents historical membership, **not a verified current or active roster**. Inactivity is not used to infer retirement.

## Architecture

```text
app/streamlit_app.py                 Presentation, controls and session state
src/data/                           Cleaning, date matching and fighter catalog
src/features/                       Historical records, activity and V2 features
src/models/                         Training and temporal evaluation scripts
src/prediction/predict_fight_v2.py   Production inference orchestration
src/prediction/                      Feature reconstruction and prior validation
models/*_metrics.json                Aggregate reports explicitly allowed by .gitignore
tests/                              Unit, compatibility and Streamlit AppTest tests
.streamlit/config.toml              Publishable theme configuration
```

At runtime, the UI delegates to the V2 engine. The engine reconstructs matchup features from local history, validates and loads two fitted pipelines and the prior, then returns probabilities and warnings. The application does not retrain models. Launch `app/streamlit_app.py` directly.

## Evaluation: actual results and baselines

These figures come from the aggregate reports, not training-set scores or an evaluation of the final all-history fit. Higher accuracy/AUC/F1 is better; lower log loss and Brier score is better.

### Winner prediction

Pooled results across **3,885 fights** in four expanding-window test periods: 2017–2018, 2019–2020, 2021–2022, and 2023–2024.

| Configuration | Accuracy | Balanced accuracy | ROC-AUC | Log loss | Brier score |
|---|---:|---:|---:|---:|---:|
| Symmetric historical baseline A, 43 features | 57.12% | 56.01% | 0.5886 | 0.6789 | 0.2430 |
| **Winner D, 62 features** | **58.56%** | **57.43%** | **0.6125** | **0.6740** | **0.2402** |

Source: [V2 winner backtest](models/v2_winner_backtest_metrics.json), `aggregate.A.pooled` and `aggregate.D.pooled`. Always choosing source fighter 1 would achieve **56.99%** on this pooled population, calculated from the reported fighter-1 win rate. That is an order-dependent descriptive reference, not a deployable skill benchmark.

### Method conditional on winning

Pooled results across **3,874 supported-method fights**, using the same four calendar test windows.

| Configuration | Accuracy | Balanced accuracy | Macro F1 | Log loss | Multiclass Brier |
|---|---:|---:|---:|---:|---:|
| Training-prior probability baseline | 50.31% | 33.33% | 0.2231 | 1.0244 | 0.6184 |
| **METHOD-M, 47 features** | **49.38%** | **38.63%** | **0.3774** | **1.0052** | **0.6096** |

The hard majority baseline also achieves **50.31%** accuracy. METHOD-M improves probability scores over the training-prior baseline but **does not beat its accuracy**. Submission recall is only **13.81%**; predictions are decision-heavy and imperfectly calibrated. Multiclass Brier is the mean sum of squared class errors, range 0–2, and is not directly comparable to the binary winner Brier score.

Source: [method backtest](models/method_backtest_metrics.json), `aggregate.METHOD-M`, `aggregate.training_prior`, and `aggregate.majority`.

### Finish-round estimates

The production round output is an unsmoothed empirical distribution, conditional on a KO/TKO or submission finish and the explicit schedule. The frozen prior uses **3,861 three-round finishes** and **464 five-round finishes** through **2026-05-16**. Three-round schedules have no round-4/5 mass. Decisions receive no finish-round estimate.

The experimental [round-model report](models/round_backtest_metrics.json) is separate from this production prior. Experiments using the true method as conditional context are not end-to-end forecasts. The dashboard makes no learned round-prediction performance claim.

## Methodology and safeguards

1. Clean fight results and per-round statistics, then attach event dates using source URLs or corroborated event/fighter identities. Unmatched dates are not guessed.
2. Build features strictly **before** each fight date. Same-day results are excluded; Elo uses frozen start-of-day ratings. Repeated source URLs count once in histories.
3. Use fighter-A-minus-fighter-B features. Winner D includes historical records/statistics, activity, Elo/opponent quality, and defense. METHOD-M includes historical method/finish summaries.
4. Fit imputation, scaling, and logistic regression on each temporal fold's training population only. Winner training adds swapped features with opposite labels. Method training uses the observed winning candidate once per supported fight, without adding the loser as a winning example.
5. Use constant-zero imputation for missing relative features and uncentered standard scaling. Zero means unknown relative advantage, not a zero physical measurement.
6. Compose joint probabilities as `P(fighter wins) × P(method | fighter wins)`. Opposing conditional method distributions need not be complements. The round prior is reported separately.

The final Winner D and METHOD-M fits use **8,547** and **8,524** original fights respectively, through **2026-05-16**. Winner augmentation produces **17,094** fitted rows. Metadata records Python **3.12.10** and scikit-learn **1.9.1**.

Both production feature schemas were reproduced against 160 historical samples each, with zero mismatches within numerical tolerance. Tests additionally cover temporal exclusion, duplicate handling, fighter swaps, missing data, schemas, probability totals, and UI state transitions.

## Limitations

- **Modest predictive signal:** winner gains are small and method submission recall is low. Probabilities are uncertain estimates, not guarantees or betting advice.
- **Exploratory evaluation:** 2025+ was excluded from primary V2 comparisons, but periods had already been inspected in earlier experiments. They are not pristine holdouts. Pooled metrics combine predictions from different fitted folds.
- **Historical inspection is not historical forecasting:** an earlier selected date limits features, but the frozen models and fixed prior still include training history through May 16, 2026.
- **Incomplete and stale data:** local dated history ends on May 16, 2026. Of 8,936 cleaned records, 210 have no matched date and are excluded from chronological histories. Newer results and fighters may be unavailable.
- **Identity limitations:** whitespace/case normalization does not resolve aliases or shared names. Observed debut does not establish career debut, and historical divisions do not establish a current roster.
- **Retrospective corrections:** final results lack publication timestamps; fight-date-safe features do not establish full point-in-time availability.
- **Outcome-population mismatch:** winner training includes DQs while method training excludes them. The joint output is a supported-outcome approximation; draws, no contests, DQs and overturned outcomes have no explicit probability mass.
- **Limited context:** production predictions do not use age, injuries, betting odds, or external live information. Redundant features remain; coefficients are associations, not causal effects.
- **Round uncertainty:** the prior is neither fighter-specific nor method-specific; five-round and late-round evidence is comparatively sparse.

## Local setup: Python 3.12

The verified platform is **Windows with Python 3.12.10**. A clean temporary environment installed the unchanged `requirements.txt`, passed `pip check`, and passed the complete suite with the existing local data/artifacts. Linux/macOS installation is not verified; requirements include Windows-specific `pywinpty` and notebook tooling. Transitive dependencies are not fully locked.

From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Installation does not supply the restricted files below. Obtain only files you are authorized to possess and use; no public restoration source is provided. A newer dataset or retrained replacement is not the validated snapshot.

### Required local runtime bundle

| File | Role |
|---|---|
| `data/processed/fights_dated.csv` | Exact history snapshot |
| `models/ufc_winner_d_v2_0_0.joblib` | Winner D pipeline |
| `models/ufc_winner_d_v2_0_0_metadata.json` | Winner schema, identity and hashes |
| `models/ufc_method_m_v1_0_0.joblib` | METHOD-M pipeline |
| `models/ufc_method_m_v1_0_0_metadata.json` | Method schema, identity and hashes |
| `models/ufc_round_prior_v1_0_0.json` | Empirical prior |

These six files are deliberately ignored. There is **no approved downloadable bundle**. Restoration currently means placing an authorized copy of the exact frozen files at these paths and verifying against their metadata. Load only trusted joblib files.

Recorded SHA-256 values:

```text
ufc_winner_d_v2_0_0.joblib
5cf97062cb9246638d2e144c706ef28a9ea7488bd2a8ee64a9bf01f2058deed0

ufc_method_m_v1_0_0.joblib
49c873e086ac3e8b84508e24a1bcbd30aa2e21db11f6d52e54b40199875419f0

data/processed/fights_dated.csv
1d515cbbff7de7ea9908ccac5b87ec0bdd0288dad962a4f59229304e655b335e
```

`.gitattributes` disables Git content conversion for CSVs, model files, and checksum-bound `src/prediction/build_v2_matchup_features.py`. It preserves current bytes rather than normalizing them. This is an integrity policy, not permission to redistribute. Editing protected files can still invalidate validation; existing altered checkouts are not repaired automatically.

### Launch

After the authorized runtime bundle is present:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open the local address printed by Streamlit. Select an observed division, two fighters, a date, and the announced three- or five-round schedule. Missing or invalid files produce a friendly initialization error. Input changes clear stale results.

### Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The validated local suite has **55 tests**, with zero failures, errors, or skipped tests in the last clean-install audit. It includes Streamlit AppTest integration tests. Windows must allow temporary-directory writes: `AppTest.from_string()` creates temporary scripts before application execution.

The complete suite needs more than the six-file runtime bundle: legacy V1 artifacts/metadata, intermediate activity/V2 feature CSVs, and evaluation reports are used by compatibility tests. Without authorized local fixtures, a source-only checkout cannot pass the complete suite. Tests do not retrain production models.

Training and preprocessing scripts are included for methodology inspection, not as installation steps. Some refuse to overwrite versioned outputs. No retraining or regeneration is required to run the frozen application with its matching bundle.

## Attribution and publication boundary

| Local raw file | Matching public source |
|---|---|
| `ufc_fight_results.csv` | [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats/blob/main/ufc_fight_results.csv) |
| `ufc_fight_stats.csv` | [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats/blob/main/ufc_fight_stats.csv) |
| `fights.csv` | [larissapavan/ufc-historical-fight-dataset](https://github.com/larissapavan/ufc-historical-fight-dataset/blob/main/fights.csv) |
| `fighters.csv` | [larissapavan/ufc-historical-fight-dataset](https://github.com/larissapavan/ufc-historical-fight-dataset/blob/main/fighters.csv) |

Both repositories identify [UFCStats](http://ufcstats.com/) as the underlying source. Matching filenames, schemas and sampled records support attribution; the exact acquisition revision and full upstream byte identity were not established.

Greco has a repository-level GPL-3.0 license. No explicit redistribution license was located in the inspected Larissapavan repository. Dataset-specific scope and underlying-source rights remain unresolved. Attribution and research-use descriptions do not establish redistribution permission. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for evidence and restrictions.

The publication policy excludes `data/` and defaults to excluding `models/`, with explicit exceptions for eight aggregate evaluation reports. Weights, runtime metadata and the prior remain excluded. Reports describe computed experiments; their inclusion grants no rights to redistribute source data or fitted artifacts.

## License and status

The original code is licensed under the [MIT License](LICENSE), copyright **2026 skander5766**. This license covers only the project's original code. Third-party code and dependencies, source/processed datasets, trained models, runtime metadata, empirical priors, aggregate evaluation artifacts, and third-party names/marks are outside its scope. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their separate status and unresolved permissions. Publication eligibility of an aggregate report does not make it MIT-licensed.

This independent research/portfolio project has no official UFC affiliation. No official branding assets or screenshots are bundled. Source-only publication awaits explicit approval; nothing has been published by this preparation. Publishing the source does not require bundling the excluded datasets or artifacts. A runnable local release still needs an authorized restoration path; a hosted demo separately requires clearance for hosted use and public outputs.
