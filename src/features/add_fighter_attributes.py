"""Attach approximately static current-table attributes, without model training.

Height/reach are inches. These current measurements and stances are NOT
guaranteed to describe the fighter on each historical event date. Name-only
matching cannot prove identity, even when the normalized name is unique.
"""

import hashlib
from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/prefight_features.csv"
FIGHTERS = ROOT / "data/raw/fighters.csv"
OUTPUT = ROOT / "data/processed/prefight_features_enriched.csv"
ALLOWED = ["first_name", "last_name", "height", "reach", "stance"]
NEW_COLUMNS = [f"fighter_{side}_{attribute}" for side in (1, 2)
               for attribute in ("height", "reach", "stance")] + ["height_diff", "reach_diff", "same_stance"]
MISSING = {"", "--", "---", "n/a", "unknown", "none", "null"}


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    # Unicode canonical normalization, whitespace collapse and case folding.
    # Preserve accents, punctuation, word order, suffixes; no fuzzy aliases.
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).casefold()


def measurement_text(value):
    if pd.isna(value) or str(value).strip().casefold() in MISSING:
        return None
    return str(value).strip()


def parse_height(value):
    text = measurement_text(value)
    if text is None:
        return float("nan")
    match = re.fullmatch(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"", text)
    if not match:
        raise ValueError(f"Unexpected height format: {value!r}")
    feet, inches = map(float, match.groups())
    if not (feet > 0 and 0 <= inches < 12):
        raise ValueError(f"Invalid feet/inches: {value!r}")
    return 12 * feet + inches


def parse_reach(value):
    text = measurement_text(value)
    if text is None:
        return float("nan")
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*"', text)
    if not match or float(match[1]) <= 0:
        raise ValueError(f"Unexpected reach: {value!r}")
    return float(match[1])


def prepare_lookup(fighters: pd.DataFrame):
    fighters = fighters[ALLOWED].copy()
    fighters["name"] = (fighters.first_name.fillna("") + " " + fighters.last_name.fillna("")).str.strip()
    fighters["key"] = fighters.name.map(normalize_name)
    counts = fighters.loc[fighters.key.ne(""), "key"].value_counts()
    ambiguous = set(counts[counts > 1].index)
    print("Normalization: NFC Unicode, case folding, whitespace collapse/trim only; no fuzzy matching.")
    print("Duplicate/ambiguous source names:\n" + fighters.loc[fighters.key.isin(ambiguous), ["name", "key"]].to_string(index=False))
    print(f"Empty source name keys: {fighters.key.eq('').sum()}")
    print('Height source format: feet/inches, e.g. 5\' 11". Reach: inches, e.g. 71.0". Output unit: inches.')
    for column, parser in (("height", parse_height), ("reach", parse_reach)):
        fighters[column] = fighters[column].map(parser)
        print(f"Source {column}: missing={fighters[column].isna().sum()}, range={fighters[column].min()}–{fighters[column].max()} inches")
    stance = fighters.stance.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    fighters["stance"] = stance.mask(stance.str.casefold().isin(MISSING))
    print("Source stance categories:\n" + fighters.stance.value_counts(dropna=False).to_string())
    unique = fighters.loc[~fighters.key.isin(ambiguous) & fighters.key.ne("")]
    lookup = unique.set_index("key")[["height", "reach", "stance"]]
    assert lookup.index.is_unique
    return lookup, ambiguous


def enrich(original: pd.DataFrame, lookup: pd.DataFrame, ambiguous: set) -> pd.DataFrame:
    if set(NEW_COLUMNS) & set(original.columns):
        raise ValueError("Input already contains enrichment columns.")
    if not original.fight_id.is_unique:
        raise ValueError("Input fight IDs are not unique.")
    result = original.copy()
    appearances = []
    for side in (1, 2):
        names = original[f"fighter_{side}"]
        keys = names.map(normalize_name)
        status = pd.Series("unmatched", index=original.index)
        status.loc[keys.isin(ambiguous)] = "ambiguous"
        status.loc[keys.isin(lookup.index)] = "exact"
        print(f"Fighter {side} match counts:\n" + status.value_counts().to_string())
        appearances.append(pd.DataFrame({"name": names, "key": keys, "status": status}))
        for attribute in ("height", "reach", "stance"):
            result[f"fighter_{side}_{attribute}"] = keys.map(lookup[attribute])
            assert result.loc[status.ne("exact"), f"fighter_{side}_{attribute}"].isna().all()
    all_names = pd.concat(appearances, ignore_index=True)
    unique_names = all_names.drop_duplicates("key")
    print("Unique normalized fighter match counts:\n" + unique_names.status.value_counts().to_string())
    print("All fighter-appearance match counts:\n" + all_names.status.value_counts().to_string())
    print("Unmatched/ambiguous names and appearances:\n" +
          all_names.loc[all_names.status.ne("exact")].groupby(["status", "name"], dropna=False).size().to_string())
    result["height_diff"] = result.fighter_1_height - result.fighter_2_height
    result["reach_diff"] = result.fighter_1_reach - result.fighter_2_reach
    both_stances = result.fighter_1_stance.notna() & result.fighter_2_stance.notna()
    result["same_stance"] = result.fighter_1_stance.eq(result.fighter_2_stance).astype("boolean").where(both_stances, pd.NA)
    # Unknown stance is not equivalent to a known different stance.
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert len(result) == len(original) and result.fight_id.is_unique
    assert set(result.columns) - set(original.columns) == set(NEW_COLUMNS)
    return result


def main() -> None:
    protected = [p for p in (ROOT / "data").rglob("*") if p.is_file() and p != OUTPUT]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    original = pd.read_csv(INPUT, low_memory=False)
    # Explicit usecols prevents current career records or belt status entering
    # the lookup. They can encode results occurring after historical fights.
    fighters = pd.read_csv(FIGHTERS, usecols=ALLOWED)
    lookup, ambiguous = prepare_lookup(fighters)
    result = enrich(original, lookup, ambiguous)
    print("\nAttribute coverage (both fighters present):")
    for attribute in ("height", "reach", "stance"):
        columns = [f"fighter_{side}_{attribute}" for side in (1, 2)]
        both = result[columns].notna().all(axis=1)
        print(f"{attribute}: {both.sum()}/{len(result)} ({both.mean():.2%}); per-side missing={result[columns].isna().sum().to_dict()}")
    for side in (1, 2):
        print(f"Fighter {side} stance categories:\n" + result[f"fighter_{side}_stance"].value_counts(dropna=False).to_string())
    print("Same stance:\n" + result.same_stance.value_counts(dropna=False).to_string())
    print("\nExample fights (measurements in inches):\n" +
          result[["fighter_1", "fighter_2", *NEW_COLUMNS]].head(10).to_string(index=False))
    print("Current wins/losses/draws/belt excluded: they can reveal future career information.")
    print("Weight excluded: a current listed weight is not established as stable across historical weight classes.")
    print("TEMPORAL LIMITATION: current height/reach are only approximately static; stance can change.")
    print("These values are not guaranteed to represent the historical state at each fight date.")
    print("Exact name matching is conservative, not identity proof; known duplicate names remain unresolved.")
    result.to_csv(OUTPUT, index=False)
    saved = pd.read_csv(OUTPUT, low_memory=False)
    pd.testing.assert_frame_equal(saved[original.columns], original)
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print(f"Output shape: {result.shape}; new columns: {NEW_COLUMNS}")
    print("Validation PASSED: all original columns/targets/IDs preserved, row count unchanged, no duplicated IDs,")
    print("only allowlisted attributes imported, no current-fight statistics, existing files unchanged.")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
