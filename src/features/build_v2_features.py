"""V2 observed-UFC histories; no fitting or changes to V1.

Method eligibility: binary winner and supported KO/TKO, Submission, Decision.
Per-win denominators are eligible wins; per-fight and finish/early win/loss
rates use eligible binary method fights. Counts mean observed eligible events.
Finish round/duration averages include wins AND losses. All-fight duration
uses any observed result with valid timing. Defense uses complete opponent
statistics only, including nonbinary fights; missing values are never zeros.
Opponent quality covers all distinct observed dated fights, using opponent
rating before that fight. Elo updates only supported binary method outcomes.
Same-date Elo deltas are summed from frozen start-of-day ratings. Recent
opponent averages are NaN if a window cuts a tied date with unequal ratings.
Final recorded reversals lack publication dates: this is fight-date-safe, not
a claim that retrospectively corrected results were available in real time.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data/processed/prefight_features_activity.csv'
SOURCE = ROOT / 'data/processed/fights_dated.csv'
OUTPUT = ROOT / 'data/processed/prefight_features_v2.csv'
AUDIT = runpy.run_path(str(ROOT / 'src/data/analyze_method_round_targets.py'))
HISTORY = runpy.run_path(str(ROOT / 'src/features/build_prefight_features.py'))
K = 32
METHODS = {'KO/TKO': 'ko_tko', 'Submission': 'submission', 'Decision': 'decision'}
DEFENSE = {
    'significant_strikes_absorbed_per_fight': 'sig_str_landed',
    'takedowns_allowed_per_fight': 'td_landed',
    'opponent_takedown_attempts_per_fight': 'td_attempted',
    'opponent_control_seconds_per_fight': 'ctrl_seconds',
    'opponent_submission_attempts_per_fight': 'sub_att',
}


def ratio(a, b):
    return a / b if b else np.nan


def mean(values):
    return float(np.mean(values)) if values else np.nan


def elo_delta(a, b, score):
    return K * (score - 1 / (1 + 10 ** ((b - a) / 400)))


def duration(row):
    seconds = AUDIT['seconds'](row['TIME'])
    number = pd.to_numeric(row['ROUND'], errors='coerce')
    if seconds is None or pd.isna(number) or number % 1 or number < 1:
        return np.nan
    if row['TIME FORMAT'] == 'No Time Limit' and number == 1:
        return float(seconds)
    lengths = AUDIT['format_minutes'](row['TIME FORMAT'])
    if lengths is None or number > len(lengths) or seconds > lengths[int(number)-1]*60:
        return np.nan
    return float(sum(lengths[:int(number)-1])*60 + seconds)


def recent_opponents(history, window):
    values = []
    for date in sorted({r['date'] for r in history}, reverse=True):
        ratings = [r['opponent_elo'] for r in history if r['date'] == date]
        remaining = window - len(values)
        if len(ratings) > remaining:
            if len(set(ratings)) > 1:
                return np.nan
            ratings = ratings[:remaining]
        values.extend(ratings)
        if len(values) == window:
            break
    return mean(values)


def summarize(history, rating):
    """Receives only prior entries and pre-update Elo, never a current row."""
    eligible = [r for r in history if r['eligible']]
    wins = [r for r in eligible if r['won']]
    finishes = [r for r in eligible if r['method'] != 'Decision']
    fw = [r for r in finishes if r['won']]
    fl = [r for r in finishes if not r['won']]
    out = dict(prefight_elo=rating, method_history_fights=len(eligible),
               method_history_wins=len(wins), observed_history_fights=len(history))
    for method, name in METHODS.items():
        won = sum(r['method'] == method for r in wins)
        out[f'previous_{name}_wins'] = won
        out[f'previous_{name}_losses'] = sum(r['method'] == method and not r['won'] for r in eligible)
        out[f'{name}_win_rate_per_win'] = ratio(won, len(wins))
        out[f'{name}_win_rate_per_fight'] = ratio(won, len(eligible))
    out.update(previous_finish_wins=len(fw), previous_finish_losses=len(fl),
               previous_times_finished=len(fl), finish_win_rate=ratio(len(fw), len(eligible)),
               finish_loss_rate=ratio(len(fl), len(eligible)))
    for name, rounds in [('1', [1]), ('2', [2]), ('3', [3]), ('4_5', [4, 5])]:
        out[f'previous_round_{name}_finish_wins'] = sum(r['round'] in rounds for r in fw)
    early = sum(r['round'] == 1 for r in fw)
    out.update(previous_early_finish_wins=early, early_finish_win_rate=ratio(early, len(eligible)))
    for feature, rows, field in [('average_finish_round', finishes, 'round'),
                                 ('average_fight_duration_seconds', history, 'duration'),
                                 ('average_finish_duration_seconds', finishes, 'duration')]:
        values = [r[field] for r in rows if pd.notna(r[field])]
        out[feature] = mean(values)
        out[feature + '_history_count'] = len(values)
    for feature, metric in DEFENSE.items():
        values = [r['defense'][metric] for r in history if pd.notna(r['defense'][metric])]
        out[feature] = mean(values)
        out[feature + '_history_count'] = len(values)
    paired = [r for r in history if pd.notna(r['defense']['td_landed']) and pd.notna(r['defense']['td_attempted'])]
    attempts = sum(r['defense']['td_attempted'] for r in paired)
    out['takedown_defense_rate'] = 1 - ratio(sum(r['defense']['td_landed'] for r in paired), attempts)
    out['takedown_defense_history_count'] = len(paired)
    out['opponent_td_attempts_history_total'] = attempts
    timed = [r for r in history if pd.notna(r['defense']['sig_str_landed']) and pd.notna(r['duration']) and r['duration'] > 0]
    out['significant_strikes_absorbed_per_minute'] = ratio(sum(r['defense']['sig_str_landed'] for r in timed), sum(r['duration'] for r in timed)/60)
    out['defensive_timed_history_count'] = len(timed)
    opponents = [r['opponent_elo'] for r in history]
    out.update(average_previous_opponent_elo=mean(opponents),
               maximum_previous_opponent_elo=max(opponents) if opponents else np.nan,
               average_opponent_elo_last_3=recent_opponents(history, 3),
               average_opponent_elo_last_5=recent_opponents(history, 5))
    return out


def make_entry(row, side, opponent_elo):
    method = AUDIT['normalize_method'](row['METHOD'])
    opponent = 3-side
    complete = row.get(f'fighter_{opponent}_stats_complete') in (True, 'True')
    defense = {}
    for metric in DEFENSE.values():
        value = pd.to_numeric(row.get(f'fighter_{opponent}_{metric}'), errors='coerce')
        defense[metric] = value if complete and pd.notna(value) and value >= 0 else np.nan
    if defense['td_landed'] > defense['td_attempted']:
        raise ValueError('Opponent takedowns landed exceed attempts.')
    number = pd.to_numeric(row['ROUND'], errors='coerce')
    return dict(date=row['event_date'], source=row['URL'], method=method,
                eligible=method is not None and row['outcome'] in ('fighter_1_win', 'fighter_2_win'),
                won=row['outcome'] == f'fighter_{side}_win',
                round=number if pd.notna(number) and number >= 1 and number % 1 == 0 else np.nan,
                duration=duration(row), defense=defense, opponent_elo=opponent_elo)


def build_features(source):
    source = source.copy()
    source['event_date'] = pd.to_datetime(source.event_date, errors='raise')
    source = source.dropna(subset=['event_date']).sort_values(['event_date', 'fight_id'])
    if not source.fight_id.is_unique or source[['URL', 'fighter_1', 'fighter_2']].isna().any().any():
        raise ValueError('Invalid source identity.')
    HISTORY['validate_duplicate_sources'](source)
    fields = ['METHOD', 'ROUND', 'TIME', 'TIME FORMAT', 'fighter_1_stats_complete', 'fighter_2_stats_complete']
    if source.groupby('URL')[fields].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting duplicate targets or completeness.')
    histories, ratings = defaultdict(list), defaultdict(lambda: 1500.)
    records, audited = [], 0
    for date, batch in source.groupby('event_date', sort=True):
        cache = {}
        # Emit every feature on this date before observing ANY same-day result.
        for row in batch.to_dict('records'):
            keys = [HISTORY['fighter_key'](row[f'fighter_{s}']) for s in (1, 2)]
            if keys[0] == keys[1]:
                raise ValueError('Same fighter on both sides.')
            result = {'fight_id': row['fight_id']}
            values = []
            for side, key in enumerate(keys, 1):
                if key not in cache:
                    history = histories[key]
                    assert all(r['date'] < date and r['source'] != row['URL'] for r in history)
                    assert len({r['source'] for r in history}) == len(history)
                    cache[key] = summarize(history, ratings[key])
                    audited += 1
                values.append(cache[key])
                result.update({f'fighter_{side}_{k}': v for k, v in cache[key].items()})
            for name in values[0]:
                diff_name = 'elo_diff' if name == 'prefight_elo' else name + '_diff'
                result[diff_name] = values[0][name] - values[1][name]
            records.append(result)
        deltas = defaultdict(float)
        for row in batch.drop_duplicates('URL').to_dict('records'):
            a, b = [HISTORY['fighter_key'](row[f'fighter_{s}']) for s in (1, 2)]
            first = make_entry(row, 1, ratings[b])
            second = make_entry(row, 2, ratings[a])
            histories[a].append(first)
            histories[b].append(second)
            if first['eligible']:
                delta = elo_delta(ratings[a], ratings[b], float(first['won']))
                deltas[a] += delta
                deltas[b] -= delta
        for key, delta in deltas.items():
            ratings[key] += delta
    print('Strict-date history audits:', audited, '; duplicate rows counted once:', source.URL.duplicated().sum())
    return pd.DataFrame(records)


def main():
    start = perf_counter()
    if OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUTPUT}')
    protected = [p for folder in ('data', 'models') for p in (ROOT/folder).rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    source = HISTORY['load_fights'](SOURCE)
    old = pd.read_csv(INPUT, low_memory=False)
    generated = build_features(source)
    new_columns = list(generated.columns.drop('fight_id'))
    assert not set(new_columns) & set(old.columns)
    result = old.merge(generated, on='fight_id', how='left', validate='one_to_one', sort=False)
    assert set(old.fight_id) == set(generated.fight_id)
    pd.testing.assert_frame_equal(result[old.columns], old)
    for feature in summarize([], 1500.):
        diff = 'elo_diff' if feature == 'prefight_elo' else feature + '_diff'
        np.testing.assert_allclose(result[diff], -(result[f'fighter_2_{feature}']-result[f'fighter_1_{feature}']), rtol=0, atol=0, equal_nan=True)
    assert not np.isinf(result[new_columns].to_numpy(dtype=float)).any()
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print('Shape:', result.shape, '; new features:', len(new_columns))
    print('Fighter-level features (each has both sides and a difference):', list(summarize([], 1500.)))
    print('Missing new values:\n', result[new_columns].isna().sum().to_string())
    for fighter in ['Brock Lesnar', 'Ronda Rousey', 'Khabib Nurmagomedov']:
        print('\nMANUAL HISTORY:', fighter)
        rows = []
        for row in result.sort_values('event_date').to_dict('records'):
            for side in (1, 2):
                if row[f'fighter_{side}'] == fighter:
                    rows.append({k: row[k] for k in ['event_date', 'fighter_1', 'fighter_2', 'outcome']} | {
                        k: row[f'fighter_{side}_{k}'] for k in ['prefight_elo', 'previous_ko_tko_wins', 'previous_submission_wins', 'previous_finish_wins']})
        print(pd.DataFrame(rows).to_string(index=False))
    result.to_csv(OUTPUT, index=False, mode='x')
    print('PASSED: row/target preservation, strict chronology, source deduplication, antisymmetry, protected hashes.')
    print('Runtime seconds:', round(perf_counter()-start, 3))


if __name__ == '__main__':
    main()
