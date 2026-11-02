"""Configuration C inference features; no model loading, fitting or prediction.

Names use the existing whitespace/casefold rule, without alias resolution.
Observed histories may be incomplete or conflate shared names, as in training.
Dates are calendar dates: intraday times are discarded to exclude same-day bouts.
The default builder caches a local snapshot; clear_builder_cache() reloads it.
"""
from bisect import bisect_left
from collections import defaultdict
import contextlib
from difflib import get_close_matches
from functools import lru_cache
import io
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HISTORY = runpy.run_path(str(ROOT / "src/features/build_prefight_features.py"))
ACTIVITY = runpy.run_path(str(ROOT / "src/features/add_activity_features.py"))
BASELINE = runpy.run_path(str(ROOT / "src/models/train_baseline.py"))
ENRICHED = runpy.run_path(str(ROOT / "src/models/train_enriched_symmetric_logistic.py"))
FEATURES = tuple(BASELINE["FEATURES"] + ENRICHED["ACTIVITY"] + ENRICHED["LAYOFF"])
assert len(FEATURES) == len(set(FEATURES)) == 50


def calendar_date(value):
    if isinstance(value, (int, float)):
        raise ValueError("prediction_date must be a calendar date, not a numeric timestamp.")
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid prediction_date; use YYYY-MM-DD.") from exc
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("prediction_date must be a nonmissing timezone-naive calendar date.")
    return result.normalize()


class MatchupFeatureBuilder:
    """Reusable, indexed local snapshot. Optional source supports isolated tests."""
    def __init__(self, source=None):
        if source is None:
            source = pd.read_csv(ROOT / "data/processed/fights_dated.csv", low_memory=False)
        else:
            source = source.copy()
        self.names = {}
        for name in pd.concat([source.fighter_1, source.fighter_2]).dropna():
            self.names.setdefault(HISTORY["fighter_key"](name), name)
        dates = pd.to_datetime(source.event_date, format="%Y-%m-%d", errors="coerce")
        if (source.event_date.notna() & dates.isna()).any():
            raise ValueError("Malformed source event dates.")
        source = source.loc[dates.notna()].copy()
        source["event_date"] = dates.loc[dates.notna()]
        if not source.fight_id.is_unique or source[["URL","fighter_1","fighter_2"]].isna().any().any():
            raise ValueError("Invalid source fight identity.")
        if not source.outcome.isin(["fighter_1_win","fighter_2_win","draw","no_contest"]).all():
            raise ValueError("Unexpected source outcome.")
        with contextlib.redirect_stdout(io.StringIO()):
            HISTORY["validate_duplicate_sources"](source)
        source = source.sort_values(["event_date","fight_id"]).drop_duplicates("URL")
        self.histories = defaultdict(list)
        for _, row in source.iterrows():
            if HISTORY["fighter_key"](row.fighter_1) == HISTORY["fighter_key"](row.fighter_2):
                raise ValueError("Both sides share the same normalized fighter name.")
            for side in (1,2):
                self.histories[HISTORY["fighter_key"](row[f"fighter_{side}"])].append(HISTORY["history_entry"](row,side))
        self.dates = {name: [entry["date"] for entry in entries] for name,entries in self.histories.items()}

    def resolve(self, name):
        if not isinstance(name,str) or not name.strip():
            raise ValueError("Fighter name must be a nonempty string.")
        key = HISTORY["fighter_key"](name)
        if key not in self.names:
            suggestions = [self.names[k] for k in get_close_matches(key,self.names,n=5,cutoff=.55)]
            raise ValueError(f"Unknown fighter {name!r}. Suggestions only (not selected): {suggestions}")
        return key

    def fighter_features(self, name, prediction_date):
        current = calendar_date(prediction_date)
        key = self.resolve(name)
        dates = self.dates.get(key,[])
        stop = bisect_left(dates,current)
        history = self.histories.get(key,[])[:stop]
        assert all(entry["date"] < current for entry in history)
        values = HISTORY["historical_features"](history,current)
        activity = ACTIVITY["calculate_activity"](dates[:stop],current)
        old,new = values["days_since_last_fight"],activity["days_since_last_fight"]
        assert (pd.isna(old) and pd.isna(new)) or old==new
        values.update(activity)
        return values

    def build_matchup_features(self, fighter_a, fighter_b, prediction_date):
        if self.resolve(fighter_a) == self.resolve(fighter_b):
            raise ValueError("A fighter cannot be matched against the same normalized name.")
        first = self.fighter_features(fighter_a,prediction_date)
        second = self.fighter_features(fighter_b,prediction_date)
        row = {}
        for feature in FEATURES:
            key = ENRICHED["ALIASES"].get(feature[:-5],feature[:-5])
            a,b = first[key],second[key]
            # Nullable layoff flags must not become false for a debutant.
            row[feature] = np.nan if pd.isna(a) or pd.isna(b) else float(a)-float(b)
        return pd.DataFrame([row],columns=FEATURES,dtype=float)


@lru_cache(maxsize=1)
def default_builder():
    return MatchupFeatureBuilder()


def clear_builder_cache():
    default_builder.cache_clear()


def build_matchup_features(fighter_a: str, fighter_b: str, prediction_date):
    """Return one row with exactly configuration C's 50 ordered model columns."""
    return default_builder().build_matchup_features(fighter_a,fighter_b,prediction_date)


def validate_historical(builder, sample_count=160):
    frame = pd.read_csv(ROOT / "data/processed/prefight_features_activity.csv",low_memory=False)
    frame = frame.loc[frame.outcome.isin(["fighter_1_win","fighter_2_win"])].sort_values(["event_date","fight_id"]).drop_duplicates("URL")
    sample = frame.iloc[np.unique(np.linspace(0,len(frame)-1,min(sample_count,len(frame)),dtype=int))]
    expected = ENRICHED["design"](sample,BASELINE["FEATURES"])[list(FEATURES)]
    exact = tolerance = 0
    maximum = swap_max = 0.0
    mismatches = []
    for index,row in sample.iterrows():
        generated = builder.build_matchup_features(row.fighter_1,row.fighter_2,row.event_date).iloc[0].to_numpy()
        stored = expected.loc[index].to_numpy(dtype=float)
        valid = np.isfinite(generated)&np.isfinite(stored)
        if valid.any(): maximum=max(maximum,float(np.max(np.abs(generated[valid]-stored[valid]))))
        identical = (generated==stored)|(np.isnan(generated)&np.isnan(stored))
        close = np.isclose(generated,stored,rtol=1e-10,atol=1e-10,equal_nan=True)
        if identical.all(): exact+=1
        elif close.all(): tolerance+=1
        else:
            mismatches.append({"fight_id":row.fight_id,"features":[FEATURES[i] for i in np.flatnonzero(~close)]})
        reverse = builder.build_matchup_features(row.fighter_2,row.fighter_1,row.event_date).iloc[0].to_numpy()
        np.testing.assert_allclose(reverse,-generated,rtol=0,atol=0,equal_nan=True)
        finite = np.isfinite(generated)
        if finite.any(): swap_max=max(swap_max,float(np.max(np.abs(reverse[finite]+generated[finite]))))
    result={"sample_fights":len(sample),"first_date":sample.event_date.min(),"last_date":sample.event_date.max(),
            "exact_rows":exact,"tolerance_only_rows":tolerance,"mismatches":mismatches,
            "maximum_numerical_difference":maximum,"maximum_swap_error":swap_max}
    if mismatches: raise AssertionError(result)
    return result


def main():
    started=perf_counter()
    builder=default_builder()
    print("Exact ordered features:\n"+"\n".join(FEATURES))
    print("Historical reproduction:",validate_historical(builder),flush=True)
    important=["previous_fights_diff","previous_wins_diff","previous_losses_diff","win_rate_diff",
               "sig_str_landed_per_fight_diff","sig_str_accuracy_diff","days_since_last_fight_diff",
               "fights_last_365_days_diff","fights_last_730_days_diff","layoff_over_1095_days_diff"]
    for a,b,day in [("Brock Lesnar","Mark Hunt","2016-07-09"),("Ronda Rousey","Amanda Nunes","2016-12-30"),
                    ("Khabib Nurmagomedov","Conor McGregor","2018-10-06")]:
        print(a,"vs",b,day)
        print(builder.build_matchup_features(a,b,day)[important].to_string(index=False))
    print(f"Execution time: {perf_counter()-started:.3f}s; no predictions or file writes.")


if __name__=="__main__":
    main()
