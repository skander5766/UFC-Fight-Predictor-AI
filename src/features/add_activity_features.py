"""Observed UFC activity from strictly earlier dates; no medical inference.

Windows include the lower boundary: [current date - N days, current date).
Distinct same-day historical fights count separately. Their zero-day gap is
included in the mean inter-fight interval; no within-day order is inferred.
"""

from bisect import bisect_left
from collections import defaultdict
import hashlib
import io
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/prefight_features_enriched.csv"
SOURCE = ROOT / "data/processed/fights_dated.csv"
OUTPUT = ROOT / "data/processed/prefight_features_activity.csv"
NUMERIC = ["fights_last_365_days", "fights_last_730_days", "days_since_last_fight",
           "days_since_second_last_fight", "average_days_between_previous_fights"]
LAYOFFS = [f"layoff_over_{days}_days" for days in (365, 730, 1095)]


def fighter_key(name: str) -> str:
    # Same conservative identity rule as existing history features. No aliases.
    return " ".join(name.split()).casefold()


def prepare_history(source: pd.DataFrame):
    source = source[["fight_id", "URL", "event_date", "fighter_1", "fighter_2"]].copy()
    dates = pd.to_datetime(source.event_date, format="%Y-%m-%d", errors="coerce")
    if (source.event_date.notna() & dates.isna()).any():
        raise ValueError("Malformed nonmissing source dates.")
    print(f"Undated source rows excluded: {dates.isna().sum()}")
    source = source.loc[dates.notna()].copy()
    source["event_date"] = dates.loc[dates.notna()]
    if source[["URL", "fighter_1", "fighter_2"]].isna().any().any() or not source.fight_id.is_unique:
        raise ValueError("Missing identity or duplicated source fight IDs.")
    duplicates = source.loc[source.URL.duplicated(keep=False)]
    conflicts = duplicates.groupby("URL")[["event_date", "fighter_1", "fighter_2"]].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting duplicated source fights: {conflicts[conflicts].index.tolist()}")
    print(f"Duplicate source aliases counted only once in history: {source.URL.duplicated().sum()}")
    source = source.sort_values(["event_date", "fight_id"]).drop_duplicates("URL")
    histories = defaultdict(list)
    for row in source.itertuples(index=False):
        if fighter_key(row.fighter_1) == fighter_key(row.fighter_2):
            raise ValueError("Indistinguishable fighter identities.")
        for name in (row.fighter_1, row.fighter_2):
            histories[fighter_key(name)].append((row.event_date, row.URL, row.fight_id))
    # Construct the sorted date index ONCE per fighter, not once per appearance.
    return {key: np.array([entry[0] for entry in history], dtype="datetime64[D]").astype(np.int64)
            for key, history in histories.items()}


def query_activity(dates: np.ndarray, current: np.ndarray):
    """Vectorized binary searches for all appearances of one fighter."""
    if np.any(np.diff(dates) < 0):
        raise ValueError("History dates must be sorted.")
    counts = np.searchsorted(dates, current, side="left")
    has_prior, has_two = counts > 0, counts > 1
    last = np.full(len(current), np.nan)
    second = np.full(len(current), np.nan)
    average = np.full(len(current), np.nan)
    previous = np.full(len(current), np.nan)
    previous[has_prior] = dates[counts[has_prior] - 1]
    last[has_prior] = current[has_prior] - previous[has_prior]
    second[has_two] = current[has_two] - dates[counts[has_two] - 2]
    if len(dates):
        average[has_two] = (previous[has_two] - dates[0]) / (counts[has_two] - 1)
    # The maximum included date is strictly earlier. The first excluded date
    # is >= current: together these prove the complete strict-date partition.
    assert np.all(previous[has_prior] < current[has_prior])
    excluded = counts < len(dates)
    assert np.all(dates[counts[excluded]] >= current[excluded])
    values = {
        "fights_last_365_days": counts - np.searchsorted(dates, current - 365, side="left"),
        "fights_last_730_days": counts - np.searchsorted(dates, current - 730, side="left"),
        "days_since_last_fight": last,
        "days_since_second_last_fight": second,
        "average_days_between_previous_fights": average,
    }
    for threshold in (365, 730, 1095):
        values[f"layoff_over_{threshold}_days"] = np.where(has_prior, last > threshold, np.nan)
    return pd.DataFrame(values), counts, previous


def calculate_activity(dates: list[pd.Timestamp], current: pd.Timestamp) -> dict:
    if any(day >= current for day in dates):
        raise AssertionError("Current, same-day or future fight entered history.")
    n = len(dates)
    elapsed = (current - dates[-1]).days if n else np.nan
    result = {
        "fights_last_365_days": n - bisect_left(dates, current - pd.Timedelta(days=365)),
        "fights_last_730_days": n - bisect_left(dates, current - pd.Timedelta(days=730)),
        "days_since_last_fight": elapsed,
        "days_since_second_last_fight": (current - dates[-2]).days if n >= 2 else np.nan,
        # Mean of consecutive historical intervals; excludes the current gap.
        "average_days_between_previous_fights": (dates[-1] - dates[0]).days / (n - 1) if n >= 2 else np.nan,
    }
    for threshold in (365, 730, 1095):
        result[f"layoff_over_{threshold}_days"] = bool(elapsed > threshold) if n else pd.NA
    return result


def add_activity(original: pd.DataFrame, histories):
    current_dates = pd.to_datetime(original.event_date, format="%Y-%m-%d", errors="raise")
    if current_dates.isna().any() or not original.fight_id.is_unique:
        raise ValueError("Output input needs dated, unique fight IDs.")
    result = original.copy()
    audit_rows = []
    day_numbers = current_dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    for side in (1, 2):
        keys = original[f"fighter_{side}"].map(fighter_key)
        # Work on each fighter's appearances together, using positional indices.
        values = np.full((len(original), len(NUMERIC + LAYOFFS)), np.nan)
        previous_dates = np.full(len(original), np.nan)
        history_counts = np.zeros(len(original), dtype=int)
        for key, positions in keys.groupby(keys, sort=False).indices.items():
            dates = histories.get(key, np.array([], dtype=np.int64))
            batch, counts, previous = query_activity(dates, day_numbers[positions])
            values[positions] = batch[NUMERIC + LAYOFFS].to_numpy()
            history_counts[positions] = counts
            previous_dates[positions] = previous
        np.testing.assert_array_equal(history_counts, original[f"fighter_{side}_previous_fights"])
        computed = pd.DataFrame(values, columns=NUMERIC + LAYOFFS, index=original.index)
        for name in ("fights_last_365_days", "fights_last_730_days"):
            computed[name] = computed[name].astype(int)
        audit = computed.copy()
        audit["fighter"] = original[f"fighter_{side}"]
        audit["opponent"] = original[f"fighter_{3-side}"]
        audit["fight_date"] = current_dates
        audit["previous_fight_date"] = pd.to_datetime(previous_dates, unit="D", origin="unix")
        audit_rows.append(audit)
        existing = f"fighter_{side}_days_since_last_fight"
        agrees = np.isclose(original[existing], computed.days_since_last_fight, rtol=0, atol=0, equal_nan=True)
        print(f"Fighter {side} existing days-since-last discrepancies: {(~agrees).sum()}")
        if not agrees.all():
            print(original.loc[~agrees, ["fight_id", "event_date", f"fighter_{side}", existing]].to_string(index=False))
            print(computed.loc[~agrees, ["days_since_last_fight"]].to_string())
            raise ValueError("Existing elapsed-day feature disagrees; no output written.")
        for name in NUMERIC + LAYOFFS:
            column = f"fighter_{side}_{name}"
            if name == "days_since_last_fight":
                continue  # Verified above, preserved rather than overwritten.
            if column in original:
                raise ValueError(f"Enrichment column already exists: {column}")
            result[column] = computed[name].astype("boolean") if name in LAYOFFS else computed[name]
    for name in NUMERIC:
        difference = result[f"fighter_1_{name}"] - result[f"fighter_2_{name}"]
        column = name + "_diff"
        if column in original:
            agrees = np.isclose(original[column], difference, rtol=0, atol=0, equal_nan=True)
            print(f"Existing {column} discrepancies: {(~agrees).sum()}")
            if not agrees.all():
                raise ValueError("Existing difference disagrees; no output written.")
        else:
            result[column] = difference
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.fight_id.is_unique and len(result) == len(original)
    audit = pd.concat(audit_rows, ignore_index=True)
    assert len(audit) == 2 * len(result)
    return result, audit


def boundary_tests() -> None:
    """Compare optimized searches with the independent scalar reference."""
    current = pd.Timestamp("2020-01-01")
    offsets = [[], [365], [366], [730, 365], [731], [1095], [1096], [800, 800], [731, 730, 365, 1, 0, -1]]
    for gaps in offsets:
        dates = sorted(current - pd.Timedelta(days=gap) for gap in gaps)
        earlier = [day for day in dates if day < current]
        expected = calculate_activity(earlier, current)
        values, counts, _ = query_activity(
            np.array(dates, dtype="datetime64[D]").astype(np.int64),
            np.array([current], dtype="datetime64[D]").astype(np.int64))
        assert counts[0] == len(earlier)
        for name, value in expected.items():
            actual = values.loc[0, name]
            assert pd.isna(actual) if pd.isna(value) else actual == value, name
    for invalid in (current, current + pd.Timedelta(days=1)):
        try:
            calculate_activity([invalid], current)
        except AssertionError:
            pass
        else:
            raise AssertionError("Scalar leakage guard failed.")
    print("Boundary tests PASSED: debut, single prior fight, window edges, strict thresholds, same-day exclusion, historical same-day gaps, future exclusion.", flush=True)


def main() -> None:
    started = perf_counter()
    boundary_tests()
    protected = [p for p in (ROOT / "data").rglob("*") if p.is_file() and p != OUTPUT]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    original = pd.read_csv(INPUT, low_memory=False)
    source = pd.read_csv(SOURCE, low_memory=False)
    print(f"Loading/hashing: {perf_counter() - started:.3f}s", flush=True)
    # Verify that the target rows themselves agree with the source calendar.
    source_rows = source.set_index("fight_id").loc[original.fight_id]
    for column in ("URL", "event_date", "fighter_1", "fighter_2"):
        np.testing.assert_array_equal(original[column], source_rows[column])
    history_started = perf_counter()
    histories = prepare_history(source)
    print(f"History preparation: {perf_counter() - history_started:.3f}s", flush=True)
    calculation_started = perf_counter()
    result, audit = add_activity(original, histories)
    print(f"Activity calculation and comparisons: {perf_counter() - calculation_started:.3f}s", flush=True)
    new = [column for column in result if column not in original]
    print(f"\nOutput shape: {result.shape}; new columns ({len(new)}):\n" + "\n".join(new))
    print("\nMissingness of new features:\n" + result[new].isna().sum().to_string())
    for threshold in (365, 730, 1095):
        columns = [f"fighter_{side}_layoff_over_{threshold}_days" for side in (1, 2)]
        print(f"Layoff > {threshold} days: per side={result[columns].sum().to_dict()}, "
              f"fights with at least one={result[columns].fillna(False).any(axis=1).sum()}")
    print("\nManual long-gap examples:")
    examples = audit.loc[
        ((audit.fighter == "Brock Lesnar") & (audit.opponent == "Mark Hunt"))
        | ((audit.fighter == "Georges St-Pierre") & (audit.opponent == "Michael Bisping"))
        | ((audit.fighter == "Dominick Cruz") & (audit.opponent == "Takeya Mizugaki"))
        | ((audit.fighter == "Nick Diaz") & (audit.opponent == "Robbie Lawler"))
    ]
    print(examples[["fighter", "opponent", "fight_date", "previous_fight_date", "days_since_last_fight",
                    "fights_last_365_days", "fights_last_730_days", *LAYOFFS]].to_string(index=False))
    print(f"Strict-date leakage validation PASSED for {len(audit)} fighter histories.")
    print("Counts use [date-N days, date); current/same-date fights excluded. Source aliases counted once.")
    print("Mean historical interval includes zero-day gaps between distinct earlier same-day bouts.")
    print("Elapsed-time flags describe observed UFC gaps only; no injury/retirement inference.")
    print("Observed-history coverage and name-identity limitations remain; debut means no earlier observed UFC fight.")
    # Validate the complete serialized candidate in memory BEFORE writing.
    csv_text = result.to_csv(index=False)
    saved = pd.read_csv(io.StringIO(csv_text), low_memory=False)
    pd.testing.assert_frame_equal(saved[original.columns], original)
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    OUTPUT.write_text(csv_text, encoding="utf-8")
    print("Validation PASSED: original columns, IDs, targets and row count preserved; no duplicate fights introduced; existing datasets unchanged.")
    print(f"Saved: {OUTPUT}")
    print(f"Total execution time: {perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
