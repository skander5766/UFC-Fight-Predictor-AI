# Third-party sources and distribution restrictions

Evidence reviewed on **November 2, 2026**. These notices record attribution and known gaps; they do not grant rights or certify that all upstream rights are cleared.

## UFCStats: underlying records

Both identified dataset repositories state that they collect from [UFCStats](http://ufcstats.com/). Local fight and fighter URL columns point to that domain. Local `ufc_fight_stats.csv` has no URL column; preprocessing joins statistics to fight-result identities.

No documented authorization from UFCStats/UFC to redistribute this project's snapshots was located. Public accessibility does not establish an unrestricted dataset license. The applicability of site terms, database rights, and any other underlying rights has not been resolved. No claim of public-domain status is made.

## Greco1899/scrape_ufc_stats

- Repository: <https://github.com/Greco1899/scrape_ufc_stats>
- Results: <https://github.com/Greco1899/scrape_ufc_stats/blob/main/ufc_fight_results.csv>
- Statistics: <https://github.com/Greco1899/scrape_ufc_stats/blob/main/ufc_fight_stats.csv>
- Schema/configuration: <https://github.com/Greco1899/scrape_ufc_stats/blob/main/scrape_ufc_stats_config.yaml>
- Repository license: [GNU GPL version 3](https://github.com/Greco1899/scrape_ufc_stats/blob/main/LICENSE).

Filenames and complete configured schemas match the local files. A sampled result record also matches. Current upstream results contain additional fights, and a full remote statistics-file comparison was unavailable because of browser size limits. The exact acquisition commit and complete byte identity have not been established.

**Verified:** a GPL-3.0 license is present at repository level. The README explicitly offers the scraped CSVs and names UFCStats as the source.

**Unresolved:** the exact dataset-specific license scope and the uploader's authority over underlying-source rights. A repository license cannot establish rights the uploader does not hold. Applicable GPL license, copyright, warranty and modification notices must be preserved for covered material. This document does not relicense Greco's material or claim that using its data automatically licenses this project's code under GPL.

## larissapavan/ufc-historical-fight-dataset

- Repository: <https://github.com/larissapavan/ufc-historical-fight-dataset>
- Fights: <https://github.com/larissapavan/ufc-historical-fight-dataset/blob/main/fights.csv>
- Fighters: <https://github.com/larissapavan/ufc-historical-fight-dataset/blob/main/fighters.csv>
- Source statement: [upstream README](https://github.com/larissapavan/ufc-historical-fight-dataset/blob/main/README.md).

Local fights match the upstream schema, 8,701-record count, and sampled first/last records. Fighters match the schema, 4,494-record count, and sampled first/last records. These are strong matching-source findings, not forensic proof of the original download route or a full hash comparison.

**Verified:** the README attributes collection to UFCStats and describes research, analytics and machine-learning uses.

**Unresolved:** no explicit redistribution license or written grant was found in the inspected repository and README. Listed use cases do not explicitly grant redistribution rights. Public GitHub availability and attribution alone do not settle permission; see [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

## Local derivation and publication policy

| Local material | Derivation / policy |
|---|---|
| `data/raw/ufc_fight_results.csv`, `data/raw/ufc_fight_stats.csv` | Matching Greco source; withheld pending rights clarification |
| `data/raw/fights.csv`, `data/raw/fighters.csv` | Matching Larissapavan source; withheld pending explicit redistribution permission |
| `data/processed/fights_clean.csv` | Results/statistics joined by `src/data/preprocess_fights.py` |
| `data/processed/fights_dated.csv` | Clean records joined to `fights.csv` dates by `src/data/add_fight_dates.py` |
| `data/processed/prefight_features*.csv` | Historical features; enrichment uses `fighters.csv`, followed by activity and V2 stages |
| Fitted models and runtime metadata | Derived from local feature/history snapshots; redistribution clearance not established |
| `models/ufc_round_prior_v1_0_0.json` | Empirical counts/probabilities from local history; distribution held pending review |
| Eight explicitly allowed `models/*_metrics.json` files | Aggregate local evaluation results, not bundled raw fight/statistics records or fitted model binaries |

All source and processed datasets are ignored. Model outputs default to ignored, except the eight explicitly named aggregate reports in `.gitignore`. Derivation does not itself prove permission. Withholding artifacts is a conservative publication decision, not a finding that model redistribution is necessarily forbidden. Byte-preserving `.gitattributes` rules protect integrity; they do not grant redistribution rights.

Before distributing excluded material, identify its exact source revision, establish applicable permissions for the intended distribution, preserve required notices, and document an authorized restoration source with exact paths and SHA-256 hashes. No approved public runtime bundle currently exists. Do not bypass the exclusion policy with force-adds or a GitHub release upload.

## Dependencies, marks and original code

Third-party Python packages in `requirements.txt` retain their own licenses; this repository does not bundle installed package code or relicense it. This is not a complete dependency-license or supply-chain audit. UFC names/marks belong to their respective owners; the project is independent and includes no official branding assets.

The original code of UFC Fight Predictor AI is licensed under the [MIT License](LICENSE), copyright **2026 skander5766**. The standard MIT grant applies only to that original code; the scope notice in `LICENSE` identifies the excluded materials.

The MIT grant does **not** apply to third-party code or dependencies, raw or processed datasets, fitted models, runtime metadata, empirical priors, aggregate evaluation artifacts, or third-party names/marks. These retain their separate applicable rights and obligations; unresolved permissions remain unresolved. The eight aggregate evaluation reports are eligible for the source-only publication package, but publication eligibility is not an MIT license grant for those reports.

In particular, Greco's repository-level GPL-3.0 license is not replaced by this project's MIT license, and no MIT permission is asserted for Larissapavan's datasets or underlying UFCStats material. Merely using, referencing, or deriving outputs from third-party material does not establish an entitlement to relicense it. This notice does not assert that trained-model redistribution is automatically permitted or prohibited.
