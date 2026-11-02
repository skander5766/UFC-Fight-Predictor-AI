"""Create fight-level results and post-fight totals, without ML features.

Run with .venv/Scripts/python.exe src/data/preprocess_fights.py.
Raw files are read only. Ambiguous statistics are never assigned by row order.
"""

import hashlib
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "fights_clean.csv"
KEY = ["EVENT", "BOUT"]
PAIRS = {
    "SIG.STR.": "sig_str", "TOTAL STR.": "total_str", "TD": "td",
    "HEAD": "head", "BODY": "body", "LEG": "leg",
    "DISTANCE": "distance", "CLINCH": "clinch", "GROUND": "ground",
}
COUNTS = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev"}
OUTCOMES = {
    "W/L": "fighter_1_win", "L/W": "fighter_2_win",
    "D/D": "draw", "NC/NC": "no_contest",
}
METRICS = [f"{name}_{kind}" for name in PAIRS.values()
           for kind in ("landed", "attempted")] + list(COUNTS.values()) + ["ctrl_seconds"]


def normalize(values: pd.Series) -> pd.Series:
    """Change whitespace only: preserve spelling, accents, case and punctuation."""
    return values.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def report_rows(label: str, rows: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{label}: {len(rows)}")
    if not rows.empty:
        print(rows[columns].to_string(index=False))


def parse_bout(value: str) -> tuple:
    if pd.isna(value):
        return (pd.NA, pd.NA)
    names = re.split(r"\s+vs\.\s+", value)
    if len(names) != 2 or not all(names) or names[0] == names[1]:
        return (pd.NA, pd.NA)
    return tuple(names)


def make_id(row: pd.Series) -> str:
    # JSON keeps component boundaries unambiguous. URL distinguishes the two
    # same-event bouts; EVENT prevents reused source URLs from collapsing rows.
    identity = json.dumps([row[c] for c in [*KEY, "URL"]], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    for column in [*KEY, "URL", "OUTCOME"]:
        results[column] = normalize(results[column])
    names = results["BOUT"].apply(parse_bout)
    results[["fighter_1", "fighter_2"]] = pd.DataFrame(names.tolist(), index=results.index)
    malformed = results["fighter_1"].isna()
    unexpected = ~results["OUTCOME"].isin(OUTCOMES)
    report_rows("Malformed BOUT values", results[malformed], KEY)
    report_rows("Unexpected OUTCOME values", results[unexpected], [*KEY, "OUTCOME"])
    missing_key = results[[*KEY, "URL"]].isna().any(axis=1) | results[[*KEY, "URL"]].eq("").any(axis=1)
    if malformed.any() or unexpected.any() or missing_key.any():
        raise ValueError("Invalid result identity, BOUT or OUTCOME; no output written.")
    results["outcome"] = results["OUTCOME"].map(OUTCOMES)
    results["winner"] = pd.Series(pd.NA, index=results.index, dtype="string")
    for side, code in ((1, "W/L"), (2, "L/W")):
        mask = results["OUTCOME"].eq(code)
        results.loc[mask, "winner"] = results.loc[mask, f"fighter_{side}"]
    results["identity_collision"] = results.duplicated(KEY, keep=False)
    results["source_url_reused"] = results.duplicated("URL", keep=False)
    report_rows("EVENT + BOUT collision rows", results[results.identity_collision], [*KEY, "URL", "OUTCOME"])
    report_rows("Reused source URL rows (possible cross-event duplicates)",
                results[results.source_url_reused], [*KEY, "URL"])
    results["fight_id"] = results.apply(make_id, axis=1)
    duplicate = results.fight_id.duplicated(keep=False)
    report_rows("Duplicate fight identifiers", results[duplicate], ["fight_id", *KEY, "URL"])
    if duplicate.any():
        raise ValueError("Fight identifiers are not unique; no output written.")
    return results


def source_text(values: pd.Series) -> pd.Series:
    text = normalize(values)
    return text.mask(text.isin(["", "--", "---"]))


def parse_pairs(values: pd.Series) -> pd.DataFrame:
    text = source_text(values)
    parts = text.str.extract(r"^(\d+) of (\d+)$").astype("Int64")
    invalid = text.notna() & parts[0].isna()
    invalid |= parts[0].gt(parts[1]).fillna(False)
    if invalid.any():
        raise ValueError(f"Invalid {values.name} landed/attempted values: {values[invalid].tolist()}")
    return parts


def parse_control(values: pd.Series) -> pd.Series:
    text = source_text(values)
    parts = text.str.extract(r"^(\d+):([0-5]\d)$").astype("Int64")
    if (text.notna() & parts[0].isna()).any():
        raise ValueError("Invalid CTRL time; expected M:SS with seconds below 60.")
    return parts[0] * 60 + parts[1]


def parse_count(values: pd.Series) -> pd.Series:
    text = source_text(values)
    number = pd.to_numeric(text, errors="coerce")
    invalid = text.notna() & (number.isna() | number.lt(0) | number.mod(1).ne(0))
    if invalid.any():
        raise ValueError(f"Invalid nonnegative integer in {values.name}.")
    return number.astype("Int64")


def aggregate_stats(stats: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    for column in [*KEY, "FIGHTER", "ROUND"]:
        stats[column] = normalize(stats[column])
    print(f"\nExact duplicate statistics rows after first: {stats.duplicated().sum()}")
    # Stats have no URL: only non-colliding EVENT + BOUT keys can be mapped.
    identities = results.loc[~results.identity_collision, [*KEY, "fight_id", "fighter_1", "fighter_2"]]
    mapped = stats.merge(identities, on=KEY, how="left", validate="many_to_one")
    report_rows("Statistics keys without an unambiguous result", mapped[mapped.fight_id.isna()].drop_duplicates(KEY), KEY)
    missing_identity = mapped[["FIGHTER", "ROUND"]].isna().any(axis=1)
    print(f"Statistics rows missing fighter or round: {missing_identity.sum()}")
    mapped = mapped.loc[mapped.fight_id.notna() & ~missing_identity].copy()
    wrong_fighter = ~mapped.FIGHTER.eq(mapped.fighter_1) & ~mapped.FIGHTER.eq(mapped.fighter_2)
    report_rows("Statistics rows with unexpected fighter names", mapped[wrong_fighter], [*KEY, "FIGHTER"])
    if wrong_fighter.any():
        raise ValueError("Statistics fighter does not match BOUT; no output written.")
    round_number = mapped.ROUND.str.extract(r"^Round ([1-9]\d*)$", expand=False)
    if round_number.isna().any():
        raise ValueError("Malformed statistics round label.")
    mapped["round_number"] = round_number.astype(int)
    duplicate = mapped.duplicated(["fight_id", "FIGHTER", "round_number"], keep=False)
    report_rows("Unresolved duplicate fighter-round rows", mapped[duplicate], [*KEY, "FIGHTER", "ROUND"])
    if duplicate.any():
        raise ValueError("Repeated fighter-round keys would double count statistics.")
    for source, target in PAIRS.items():
        parts = parse_pairs(mapped[source])
        mapped[f"{target}_landed"] = parts[0]
        mapped[f"{target}_attempted"] = parts[1]
    for source, target in COUNTS.items():
        mapped[target] = parse_count(mapped[source])
    mapped["ctrl_seconds"] = parse_control(mapped.CTRL)
    groups = mapped.groupby(["fight_id", "FIGHTER"], sort=False)
    totals = groups[METRICS].sum(min_count=1)
    # A total with an unknown round value is unknown, not a partial sum or zero.
    totals = totals.mask(groups[METRICS].count().lt(groups.size(), axis=0))
    totals["rounds_recorded"] = groups.size()
    totals["rounds_contiguous"] = groups.round_number.agg(lambda rounds: sorted(rounds) == list(range(1, len(rounds) + 1)))
    # Percentages are not additive; retain landed/attempted counts instead.
    return totals.reset_index()


def join_fights(results: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    cleaned = results.copy()
    for side in (1, 2):
        prefix = f"fighter_{side}_"
        renamed = totals.rename(columns={
            column: (f"fighter_{side}" if column == "FIGHTER" else prefix + column)
            for column in totals.columns if column != "fight_id"
        })
        cleaned = cleaned.merge(renamed, on=["fight_id", f"fighter_{side}"], how="left", validate="one_to_one")
        cleaned[prefix + "stats_present"] = cleaned[prefix + "rounds_recorded"].notna()
        cleaned[prefix + "stats_complete"] = (
            cleaned[prefix + "stats_present"]
            & cleaned[prefix + "rounds_recorded"].eq(cleaned.ROUND)
            & cleaned[prefix + "rounds_contiguous"].fillna(False).astype(bool)
            & cleaned[[prefix + metric for metric in METRICS]].notna().all(axis=1)
        )
    both = cleaned.fighter_1_stats_present & cleaned.fighter_2_stats_present
    either = cleaned.fighter_1_stats_present | cleaned.fighter_2_stats_present
    cleaned["stats_match_status"] = "missing"
    cleaned.loc[either, "stats_match_status"] = "partial"
    cleaned.loc[both, "stats_match_status"] = "matched"
    cleaned.loc[cleaned.identity_collision, "stats_match_status"] = "ambiguous_identity"
    assert len(cleaned) == len(results) and cleaned.fight_id.is_unique
    return cleaned


def main() -> None:
    results = prepare_results(pd.read_csv(RAW / "ufc_fight_results.csv"))
    totals = aggregate_stats(pd.read_csv(RAW / "ufc_fight_stats.csv"), results)
    cleaned = join_fights(results, totals)
    matched = cleaned.stats_match_status.eq("matched")
    print(f"\nFights processed (source result records): {len(cleaned)}")
    print(f"Successfully matched to both fighters' statistics: {matched.sum()} ({matched.mean():.2%})")
    report_rows("Unmatched/ambiguous fights", cleaned[~matched], ["fight_id", *KEY, "stats_match_status"])
    for side in (1, 2):
        print(f"Missing fighter_{side} statistics: {(~cleaned[f'fighter_{side}_stats_present']).sum()}")
        print(f"Incomplete fighter_{side} statistics (including missing): {(~cleaned[f'fighter_{side}_stats_complete']).sum()}")
        print(f"Missing fighter_{side} metric totals:\n" + cleaned[[f"fighter_{side}_{m}" for m in METRICS]].isna().sum().to_string())
    print("Outcome counts:\n" + cleaned.outcome.value_counts().to_string())
    print(f"Dataset shape: {cleaned.shape}")
    print("Columns:\n" + "\n".join(cleaned.columns))
    # This is the only write. No source data, features or models are produced.
    cleaned.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
