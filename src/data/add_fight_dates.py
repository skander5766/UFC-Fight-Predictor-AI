"""Attach source event dates without changing names or building ML features."""

from collections import defaultdict
from datetime import date
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data/processed/fights_clean.csv"
SOURCE = ROOT / "data/raw/fights.csv"
OUTPUT = ROOT / "data/processed/fights_dated.csv"


def text(value) -> str:
    """Temporary matching key only; original values stay untouched."""
    return "" if pd.isna(value) else " ".join(str(value).split()).casefold()


def pair(first, second) -> tuple[str, str]:
    return tuple(sorted((text(first), text(second))))


def method(value) -> str:
    value = text(value)
    decisions = {"u-dec": "decision - unanimous", "s-dec": "decision - split",
                 "m-dec": "decision - majority", "cnc": "could not continue"}
    if value in decisions:
        return decisions[value]
    for prefix, category in (("ko/tko", "ko/tko"), ("tko - doctor", "ko/tko"),
                             ("sub", "submission"), ("overturned", "overturned"), ("dq", "dq")):
        if value.startswith(prefix):
            return category
    return value


def weight(value) -> str:
    value = text(value)
    # Remove title/bout descriptors, retaining sex and weight division.
    value = re.sub(r"^ufc\s+", "", value)
    return re.sub(r"(?: title)? bout$", "", value)


def prepare_source(source: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source["_date"] = pd.to_datetime(source.event_date, format="%B %d, %Y", errors="coerce")
    # A broad plausibility guard, not an inferred date or a date correction.
    source["_plausible"] = source._date.between(pd.Timestamp("1990-01-01"), pd.Timestamp(date.today()))
    print(f"Unparseable source dates: {source._date.isna().sum()}")
    print(f"Implausible/future source dates: {(source._date.notna() & ~source._plausible).sum()}")
    source["_event"] = source.event_name.map(text)
    source["_pair"] = [pair(a, b) for a, b in zip(source.fighter_1, source.fighter_2)]
    source["_url"] = source.fight_url.map(text)
    return source


def corroborates(row: pd.Series, candidate: pd.Series) -> bool:
    """Validate a URL-linked name variation using independent source fields.

    Method/round/time are used only to verify a record's identity, never as
    pre-fight predictors. No aliases are learned or applied to other rows.
    """
    names = set(pair(row.fighter_1, row.fighter_2))
    other_names = set(candidate._pair)
    return (
        text(row.EVENT) == candidate._event
        and len((names & other_names) - {""}) >= 1
        and weight(row.WEIGHTCLASS) == weight(candidate.weight_class)
        and method(row.METHOD) == method(candidate.method)
        and row.ROUND == candidate["round"]
        and text(row.TIME) == text(candidate.time)
    )


def make_indexes(source: pd.DataFrame) -> tuple[dict, dict]:
    urls, event_pairs = defaultdict(list), defaultdict(list)
    for index, row in source.iterrows():
        if row._url:
            urls[row._url].append(index)
        if row._event and all(row._pair):
            event_pairs[(row._event, row._pair)].append(index)
    return urls, event_pairs


def match_one(row: pd.Series, source: pd.DataFrame, urls: dict, event_pairs: dict) -> tuple:
    names = pair(row.fighter_1, row.fighter_2)
    event = text(row.EVENT)
    candidates = urls.get(text(row.URL), [])
    if candidates:
        # Never select the first occurrence of a duplicated source URL.
        if len(candidates) != 1:
            return pd.NaT, "ambiguous_url", None
        index = candidates[0]
        candidate = source.loc[index]
        if names == candidate._pair and all(names):
            status = "matched_url" if event == candidate._event else "matched_url_event_alias"
        elif corroborates(row, candidate):
            status = "matched_url_name_variant"
        else:
            return pd.NaT, "conflicting_url_identity", None
    else:
        # Unordered fighter pairs allow reversed BOUT order. Pair alone is
        # unsafe for rematches, so event agreement is mandatory for fallback.
        candidates = event_pairs.get((event, names), [])
        status = "matched_event_pair"
        if len(candidates) > 1:
            candidates = [i for i in candidates
                          if weight(row.WEIGHTCLASS) == weight(source.loc[i, "weight_class"])
                          and method(row.METHOD) == method(source.loc[i, "method"])]
            status = "matched_event_pair_details"
            if not candidates:
                return pd.NaT, "conflicting_event_pair_details", None
        if len(candidates) > 1:
            return pd.NaT, "ambiguous_event_pair", None
        if not candidates:
            return pd.NaT, "unmatched", None
        index = candidates[0]
        candidate = source.loc[index]
    if pd.isna(candidate._date):
        return pd.NaT, "invalid_source_date", None
    if not candidate._plausible:
        return pd.NaT, "implausible_source_date", None
    return candidate._date, status, index


def add_dates(clean: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not clean.fight_id.is_unique:
        raise ValueError("Clean fight identifiers must be unique.")
    if {"event_date", "date_match_status"} & set(clean.columns):
        raise ValueError("Input already contains date columns.")
    urls, event_pairs = make_indexes(source)
    matches = [match_one(row, source, urls, event_pairs) for _, row in clean.iterrows()]
    dated = clean.copy()
    dated["event_date"] = pd.to_datetime([entry[0] for entry in matches])
    dated["date_match_status"] = [entry[1] for entry in matches]
    source_rows = pd.Series([entry[2] for entry in matches], index=clean.index, dtype="Int64")
    assert len(dated) == len(clean)
    pd.testing.assert_frame_equal(dated[clean.columns], clean)
    # Every assigned date must come directly from its verified source row.
    for index in source_rows.dropna().index:
        assert dated.loc[index, "event_date"] == source.loc[source_rows[index], "_date"]
    return dated, source_rows


def print_report(dated: pd.DataFrame, source_rows: pd.Series, source: pd.DataFrame) -> None:
    present = dated.event_date.notna()
    ambiguous = dated.date_match_status.str.startswith("ambiguous")
    duplicate = source_rows.notna() & source_rows.duplicated(keep=False)
    print(f"\nTotal fights: {len(dated)}")
    print(f"Fights with dates: {present.sum()} ({present.mean():.2%})")
    print(f"Fights without dates: {(~present).sum()}")
    print(f"Ambiguous matches: {ambiguous.sum()}")
    print(f"Duplicated source URL rows: {source._url.duplicated(keep=False).sum()}")
    print(f"Source records matched more than once: {source_rows[duplicate].nunique()}")
    print(f"Clean records involved in duplicate matches: {duplicate.sum()}")
    print(f"Extra matches beyond first: {source_rows.dropna().duplicated().sum()}")
    print(f"Earliest date: {dated.event_date.min()}")
    print(f"Latest date: {dated.event_date.max()}")
    print("\nMatch statuses:\n" + dated.date_match_status.value_counts().to_string())
    print("\nUndated fights by event:\n" + dated.loc[~present, "EVENT"].value_counts().to_string())
    print("\nUndated/ambiguous fight details:")
    print(dated.loc[~present, ["EVENT", "fighter_1", "fighter_2", "date_match_status"]].to_string(index=False))
    special = duplicate | dated.date_match_status.isin(["matched_url_event_alias", "matched_url_name_variant"])
    print("\nDuplicate matches and corroborated naming variations:")
    print(dated.loc[special, ["EVENT", "fighter_1", "fighter_2", "event_date", "date_match_status"]].to_string(index=False))
    print("\nSample matched fights:")
    print(dated.loc[present, ["EVENT", "fighter_1", "fighter_2", "event_date"]].head(8).to_string(index=False))
    print("Dates are source event dates, not inferred from row order, outcomes or later events.")
    print("Repeated source matches remain flagged by this report; no records are deleted.")


def main() -> None:
    clean = pd.read_csv(CLEAN, low_memory=False)
    source = prepare_source(pd.read_csv(SOURCE))
    dated, source_rows = add_dates(clean, source)
    print_report(dated, source_rows, source)
    # Write only the new dated file. Datetimes use ISO dates in CSV format.
    dated.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    print(f"Saved {dated.shape}: {OUTPUT}")


if __name__ == "__main__":
    main()
