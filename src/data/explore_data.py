"""Read-only exploration of the four raw datasets; no preprocessing or training.

Run from any directory with the project's virtual-environment Python interpreter.
"""

from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FILES = ("fights.csv", "fighters.csv", "ufc_fight_stats.csv", "ufc_fight_results.csv")
MISSING_MARKERS = {"", "--", "---", "n/a", "na", "null", "none", "unknown"}


def heading(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def missing_mask(series: pd.Series) -> pd.Series:
    """Detect placeholders for reporting only; never replace source values."""
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin(MISSING_MARKERS)


def describe_dataset(name: str, frame: pd.DataFrame) -> None:
    heading(name)
    print(f"Shape: {frame.shape}")
    print(f"Columns: {frame.columns.tolist()}")
    print("Data types:\n" + frame.dtypes.to_string())
    print(f"Duplicate rows (copies after the first): {frame.duplicated().sum()}")
    print("Missing values per column (pandas NA):\n" + frame.isna().sum().to_string())
    # Literal '--' and '---' can be invisible to pandas' default NA detection.
    placeholders = pd.Series({
        column: (missing_mask(frame[column]) & frame[column].notna()).sum()
        for column in frame.columns
    })
    print("Additional placeholder/blank values per column:\n" + placeholders.to_string())


def describe_results(results: pd.DataFrame) -> None:
    heading("Fight results: outcomes and repeated bout labels")
    print("Exact OUTCOME counts (including missing):")
    print(results["OUTCOME"].value_counts(dropna=False).to_string())
    # These are diagnostic labels, not training targets. Draws and no contests
    # do not establish a winner. Unexpected or missing outcomes need review.
    outcome = results["OUTCOME"].astype("string").str.strip().str.upper()
    categories = {
        "Win/loss": outcome.isin(["W/L", "L/W"]),
        "Draw": outcome.isin(["D/D", "D", "DRAW"]),
        "No contest": outcome.isin(["NC/NC", "NC", "NO CONTEST"]),
    }
    for label, mask in categories.items():
        print(f"{label}: {mask.sum()}")
    unusual = ~pd.concat(categories, axis=1).any(axis=1)
    print(f"Other or missing outcomes: {unusual.sum()}")
    if unusual.any():
        print(results.loc[unusual, "OUTCOME"].value_counts(dropna=False).to_string())
    bout_counts = results["BOUT"].value_counts()
    print(f"Repeated nonmissing BOUT labels: {(bout_counts > 1).sum()}")
    print(f"Extra rows sharing a BOUT label: {(bout_counts - 1).sum()}")
    print(f"Rows with repeated BOUT labels: {bout_counts[bout_counts > 1].sum()}")
    print("Repeated BOUT examples:\n" + bout_counts[bout_counts > 1].head(10).to_string())
    print(f"Duplicate EVENT + BOUT rows after first: {results.duplicated(['EVENT', 'BOUT']).sum()}")
    print("Repeated names may represent rematches; BOUT alone is not a fight ID.")


def describe_stats(stats: pd.DataFrame) -> None:
    heading("Fight statistics: row granularity")
    for column in ("EVENT", "BOUT", "FIGHTER"):
        print(f"Unique {column} values (excluding NA): {stats[column].nunique()}")
    # Each row contains one fighter's statistics for one round, not an entire
    # fight. A typical three-round fight has six rows (two fighters x three).
    # Use EVENT + BOUT to separate rematches sharing the same BOUT label.
    print("Rows represent fighter-round statistics, rather than one row per fight.")
    for keys in (["BOUT"], ["EVENT", "BOUT"]):
        sizes = stats.groupby(keys, dropna=False).size()
        print(f"\nRows per {' + '.join(keys)}: distribution")
        print(sizes.value_counts().sort_index().rename_axis("rows_per_bout").rename("bout_count").to_string())
        print("First 10 bout row counts:\n" + sizes.head(10).to_string())
    key = ["EVENT", "BOUT", "FIGHTER", "ROUND"]
    print(f"Duplicate fighter-round keys after first: {stats.duplicated(key).sum()}")
    fighters_per_bout = stats.groupby(["EVENT", "BOUT"], dropna=False)["FIGHTER"].nunique()
    print(f"Event/bout groups without exactly two fighters: {(fighters_per_bout != 2).sum()}")


def compare_bouts(results: pd.DataFrame, stats: pd.DataFrame) -> None:
    heading("Results/statistics coverage (exact source values, no normalization)")
    # Compare temporary strings only to flag whitespace; do not clean the data.
    for label, frame in (("Results", results), ("Statistics", stats)):
        for column in ("EVENT", "BOUT"):
            values = frame[column].astype("string")
            count = values.ne(values.str.strip()).sum()
            print(f"{label} {column} values with surrounding whitespace: {count}")
    for keys in (["BOUT"], ["EVENT", "BOUT"]):
        result_keys = set(results.dropna(subset=keys)[keys].itertuples(index=False, name=None))
        stat_keys = set(stats.dropna(subset=keys)[keys].itertuples(index=False, name=None))
        common = result_keys & stat_keys
        print(f"\nMatching by {' + '.join(keys)}:")
        print(f"Unique results keys: {len(result_keys)}; unique statistics keys: {len(stat_keys)}")
        print(f"Shared keys: {len(common)}")
        if result_keys:
            print(f"Results key coverage: {len(common) / len(result_keys):.2%}")
        if stat_keys:
            print(f"Statistics key coverage: {len(common) / len(stat_keys):.2%}")
        print(f"Results only: {len(result_keys - stat_keys)}")
        print(f"Statistics only: {len(stat_keys - result_keys)}")
        print(f"Results-only examples (up to 10): {sorted(result_keys - stat_keys)[:10]}")
        print(f"Statistics-only examples (up to 10): {sorted(stat_keys - result_keys)[:10]}")
    print("Missing keys are excluded from these comparisons; see per-column NA counts.")


def describe_fighters(fighters: pd.DataFrame) -> None:
    heading("Fighter attributes and names")
    for column in ("height", "reach", "stance"):
        count = missing_mask(fighters[column]).sum()
        print(f"Missing {column} (NA, blanks, or placeholders): {count}/{len(fighters)}")
    # Combine names only for this report; do not change the source DataFrame.
    names = (fighters["first_name"].fillna("").astype(str).str.strip()
             + " " + fighters["last_name"].fillna("").astype(str).str.strip()).str.strip()
    counts = names[names.ne("")].value_counts()
    repeated = counts[counts > 1]
    print(f"Duplicate fighter names (distinct names): {len(repeated)}")
    print(f"Extra rows with duplicate names: {(repeated - 1).sum()}")
    print(repeated.to_string())
    print("Names are not guaranteed unique identities; investigate fighter URLs before joining.")
    print("Current wins/losses/draws are not historical pre-fight features and may leak future information.")


def main() -> None:
    datasets = {name: pd.read_csv(RAW_DIR / name) for name in FILES}
    for name, frame in datasets.items():
        describe_dataset(name, frame)
    describe_results(datasets["ufc_fight_results.csv"])
    describe_stats(datasets["ufc_fight_stats.csv"])
    compare_bouts(datasets["ufc_fight_results.csv"], datasets["ufc_fight_stats.csv"])
    describe_fighters(datasets["fighters.csv"])


if __name__ == "__main__":
    main()
