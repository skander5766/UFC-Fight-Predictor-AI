"""Build strictly earlier-date features, not a training model.

History means observed dated fights in this file, not a complete UFC career.
Prior results are necessary for record/form features. Current winner/outcome
are copied as labels only and never passed to the feature calculation.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/fights_dated.csv"
OUTPUT = ROOT / "data/processed/prefight_features.csv"
METRICS = [
    "sig_str_landed", "sig_str_attempted", "total_str_landed", "kd",
    "td_landed", "td_attempted", "sub_att", "ctrl_seconds",
    "head_landed", "body_landed", "leg_landed", "distance_landed",
    "clinch_landed", "ground_landed",
]
RECORD_FEATURES = [
    "previous_fights", "previous_wins", "previous_losses", "previous_draws",
    "previous_no_contests", "historical_win_rate", "wins_last_3", "wins_last_5",
    "current_win_streak", "current_loss_streak", "days_since_last_fight",
    "previous_fights_with_stats", "stats_coverage", "is_ufc_debut",
]
FEATURES = (RECORD_FEATURES + [m + "_per_fight" for m in METRICS]
            + ["historical_sig_str_accuracy", "historical_td_accuracy"]
            + [m + "_history_count" for m in METRICS])
METADATA = ["fight_id", "URL", "EVENT", "BOUT", "event_date", "fighter_1",
            "fighter_2", "WEIGHTCLASS", "source_url_reused", "winner", "outcome"]


def fighter_key(name: str) -> str:
    # No guessed aliases: the input has names rather than stable fighter IDs.
    return " ".join(name.split()).casefold()


def load_fights(path: Path) -> pd.DataFrame:
    fights = pd.read_csv(path, low_memory=False)
    dates = pd.to_datetime(fights.event_date, format="%Y-%m-%d", errors="coerce")
    invalid = fights.event_date.notna() & dates.isna()
    print(f"Undated rows excluded: {fights.event_date.isna().sum()}")
    print(f"Invalid dates excluded: {invalid.sum()}")
    fights = fights.loc[dates.notna()].copy()
    fights["event_date"] = dates.loc[dates.notna()]
    if fights.empty or not fights.fight_id.is_unique:
        raise ValueError("Expected nonempty dated data with unique fight IDs.")
    if fights[["URL", "fighter_1", "fighter_2"]].isna().any().any():
        raise ValueError("Missing source URL or fighter identity.")
    if not fights.outcome.isin(["fighter_1_win", "fighter_2_win", "draw", "no_contest"]).all():
        raise ValueError("Unexpected outcome: cannot construct historical records.")
    for side in (1, 2):
        won = fights.outcome.eq(f"fighter_{side}_win")
        if not fights.loc[won, "winner"].eq(fights.loc[won, f"fighter_{side}"]).all():
            raise ValueError("Winner and outcome disagree.")
    if fights.loc[fights.outcome.isin(["draw", "no_contest"]), "winner"].notna().any():
        raise ValueError("Draw/no-contest records must not declare a winner.")
    return fights.sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def validate_duplicate_sources(fights: pd.DataFrame) -> None:
    # Preserve all output rows, but count an identical source fight only once
    # in future history. Fail rather than choose between conflicting versions.
    fields = ["event_date", "fighter_1", "fighter_2", "winner", "outcome"]
    fields += [f"fighter_{side}_{metric}" for side in (1, 2) for metric in METRICS]
    repeated = fights[fights.URL.duplicated(keep=False)]
    for url, group in repeated.groupby("URL"):
        if group[fields].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting versions of source fight: {url}")
    print(f"Repeated source rows retained in output, counted once in history: {fights.URL.duplicated().sum()}")


def date_groups(history: list[dict]) -> list[list[dict]]:
    groups = []
    for entry in history:
        if not groups or groups[-1][0]["date"] != entry["date"]:
            groups.append([])
        groups[-1].append(entry)
    return groups


def recent_wins(groups: list[list[dict]], window: int):
    wins = 0
    remaining = window
    for group in reversed(groups):
        results = [entry["result"] == "win" for entry in group]
        if len(group) > remaining:
            # Within-day order is unknown. Do not invent which fights were last.
            return wins + remaining * int(results[0]) if len(set(results)) == 1 else np.nan
        wins += sum(results)
        remaining -= len(group)
        if remaining == 0:
            break
    return wins


def streak(groups: list[list[dict]], result: str):
    count = 0
    for group in reversed(groups):
        flags = [entry["result"] == result for entry in group]
        if all(flags):
            count += len(flags)
        elif any(flags):
            return np.nan  # Same-date mixed outcomes have no known ordering.
        else:
            break  # Draws and no contests break either streak by convention.
    return count


def historical_features(history: list[dict], current_date: pd.Timestamp) -> dict:
    """This function receives no current-fight statistics, outcome or winner."""
    count = len(history)
    records = [entry["result"] for entry in history]
    groups = date_groups(history)
    usable = sum(any(pd.notna(entry[m]) for m in METRICS) for entry in history)
    features = {
        "previous_fights": count,
        "previous_wins": records.count("win"),
        "previous_losses": records.count("loss"),
        "previous_draws": records.count("draw"),
        "previous_no_contests": records.count("no_contest"),
        # Denominator includes draws and no contests; it is stated explicitly.
        "historical_win_rate": records.count("win") / count if count else np.nan,
        "wins_last_3": recent_wins(groups, 3),
        "wins_last_5": recent_wins(groups, 5),
        "current_win_streak": streak(groups, "win"),
        "current_loss_streak": streak(groups, "loss"),
        "days_since_last_fight": (current_date - history[-1]["date"]).days if count else np.nan,
        "previous_fights_with_stats": usable,
        "stats_coverage": usable / count if count else np.nan,
        "is_ufc_debut": count == 0,
    }
    for metric in METRICS:
        values = [entry[metric] for entry in history if pd.notna(entry[metric])]
        features[metric + "_history_count"] = len(values)
        features[metric + "_per_fight"] = sum(values) / len(values) if values else np.nan
    for kind in ("sig_str", "td"):
        # Use the same evidence set for numerator and denominator, not a mean
        # of percentages. Zero attempts means accuracy is undefined.
        evidence = [entry for entry in history
                    if pd.notna(entry[kind + "_landed"]) and pd.notna(entry[kind + "_attempted"])]
        attempted = sum(entry[kind + "_attempted"] for entry in evidence)
        landed = sum(entry[kind + "_landed"] for entry in evidence)
        features[f"historical_{kind}_accuracy"] = landed / attempted if attempted else np.nan
    return features


def history_entry(row: pd.Series, side: int) -> dict:
    result = row.outcome
    if result.endswith("_win"):
        result = "win" if result == f"fighter_{side}_win" else "loss"
    entry = {"date": row.event_date, "fight_id": row.fight_id,
             "url": row.URL, "result": result}
    for metric in METRICS:
        value = row[f"fighter_{side}_{metric}"]
        if pd.notna(value) and (value < 0 or not np.isfinite(value)):
            raise ValueError(f"Invalid historical statistic: {metric}")
        entry[metric] = value
    for kind in ("sig_str", "td"):
        landed, attempted = entry[kind + "_landed"], entry[kind + "_attempted"]
        if pd.notna(landed) and pd.notna(attempted) and landed > attempted:
            raise ValueError("Landed statistics exceed attempts.")
    return entry


def build_features(fights: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    validate_duplicate_sources(fights)
    histories = defaultdict(list)
    seen_urls = set()
    output = []
    audited = 0
    for current_date, batch in fights.groupby("event_date", sort=True):
        # Phase 1: emit ALL rows on this date before updating ANY history.
        for _, row in batch.iterrows():
            generated = {column: row[column] for column in METADATA}
            for side in (1, 2):
                history = histories[fighter_key(row[f"fighter_{side}"])]
                # Audit every contributing record, including those used solely
                # for records/form; this is stronger than checking a shift().
                if any(entry["date"] >= current_date or entry["fight_id"] == row.fight_id
                       or entry["url"] == row.URL for entry in history):
                    raise AssertionError("Leakage: current, same-date or future fight in history.")
                if len({entry["url"] for entry in history}) != len(history):
                    raise AssertionError("Duplicate source fight inflated history.")
                audited += 1
                features = historical_features(history, current_date)
                generated.update({f"fighter_{side}_{name}": value for name, value in features.items()})
            output.append(generated)
        # Phase 2: completed fights become available only to LATER dates.
        for _, row in batch.iterrows():
            if row.URL in seen_urls:
                continue
            if fighter_key(row.fighter_1) == fighter_key(row.fighter_2):
                raise ValueError("Both fighters have the same name key.")
            for side in (1, 2):
                histories[fighter_key(row[f"fighter_{side}"])].append(history_entry(row, side))
            seen_urls.add(row.URL)
    result = pd.DataFrame(output)
    feature_columns = [f"fighter_{side}_{name}" for side in (1, 2) for name in FEATURES]
    aliases = {"historical_win_rate": "win_rate", "historical_sig_str_accuracy": "sig_str_accuracy",
               "historical_td_accuracy": "td_accuracy"}
    differences = {}
    for name in FEATURES:
        if name == "is_ufc_debut":
            continue
        target = aliases.get(name, name) + "_diff"
        differences[target] = result[f"fighter_1_{name}"] - result[f"fighter_2_{name}"]
    result = pd.concat([result, pd.DataFrame(differences)], axis=1)
    feature_columns += list(differences)
    assert len(result) == len(fights) and result.fight_id.is_unique
    assert audited == 2 * len(result)
    return result, feature_columns, audited


def main() -> None:
    fights = load_fights(INPUT)
    result, features, audited = build_features(fights)
    both = result.fighter_1_previous_fights_with_stats.gt(0) & result.fighter_2_previous_fights_with_stats.gt(0)
    either = result.fighter_1_previous_fights_with_stats.gt(0) | result.fighter_2_previous_fights_with_stats.gt(0)
    print(f"Total dated fights: {len(fights)}; rows generated: {len(result)}")
    print(f"Output shape: {result.shape}; generated features: {len(features)}")
    print(f"Fights with historical statistics for both fighters: {both.sum()}")
    print(f"Fights with historical statistics for at least one fighter: {either.sum()}")
    for side in (1, 2):
        print(f"Fighter {side} observed UFC debuts: {result[f'fighter_{side}_is_ufc_debut'].sum()}")
    print("Missing feature counts:\n" + result[features].isna().sum().to_string())
    print(f"Leakage validation PASSED: {audited} fighter histories audited with strict earlier-date checks.")
    print(f"Date range: {result.event_date.min()} to {result.event_date.max()}")
    print("Target distribution:\n" + result.outcome.value_counts().to_string())
    print(f"Duplicate fight IDs: {result.fight_id.duplicated().sum()}")
    print("WARNING: History covers dated source records only. Names lack stable fighter IDs;")
    print("name variants can split careers and shared names can conflate fighters. Debut means no observed prior history.")
    print("Missing metrics stay missing; per-metric history counts expose their denominators.")
    print("Same-date mixed outcomes can leave streaks/recent windows unknown. No within-day order is guessed.")
    print("Winner/outcome are target metadata, not predictors. Reused source URLs remain flagged in the output.")
    result.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
