"""Read-only target feasibility audit. No feature creation, fitting, or file output.

Current METHOD/ROUND/TIME are labels, never predictors. Future historical
features must use dated, distinct sources with event_date < prediction_date.
Final recorded outcomes may include later reversals: their publication dates
are unavailable, so perfect point-in-time outcome validity cannot be claimed.
"""
import hashlib
import json
from pathlib import Path
import re
import runpy
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ['METHOD', 'ROUND', 'TIME', 'TIME FORMAT']


def normalize_method(value):
    """Explicit source-label mapping; unfamiliar/ambiguous labels stay missing."""
    text = ' '.join(str(value).split())
    if text == "TKO - Doctor's Stoppage" or text == 'KO/TKO' or text.startswith('KO/TKO '):
        return 'KO/TKO'
    if text == 'Submission' or text == 'SUB' or text.startswith('SUB '):
        return 'Submission'
    if text in ('Decision - Unanimous', 'Decision - Split', 'Decision - Majority',
                'U-DEC', 'S-DEC', 'M-DEC'):
        return 'Decision'
    return None


def format_minutes(value):
    """Read stated regulation and overtime period lengths, never actual round."""
    match = re.fullmatch(r'(\d+) Rnd(?: \+ (\d*)OT)? \(([\d-]+)\)', str(value))
    if not match:
        return None
    lengths = [int(x) for x in match[3].split('-')]
    extra = (int(match[2] or 1) if 'OT' in str(value) else 0)
    return lengths if len(lengths) == int(match[1]) + extra and min(lengths) > 0 else None


def seconds(value):
    match = re.fullmatch(r'(\d+):([0-5]\d)', str(value))
    return int(match[1]) * 60 + int(match[2]) if match else None


def counts(series):
    return series.fillna('<missing>').value_counts(dropna=False).to_string()


def audit_rounds(frame, round_col, time_col):
    rounds = pd.to_numeric(frame[round_col], errors='coerce')
    elapsed = frame[time_col].map(seconds)
    print('Raw round values:\n' + counts(frame[round_col]))
    print('Round missing/invalid:', int((rounds.isna() | rounds.mod(1).ne(0) | rounds.lt(1)).sum()),
          'outside 1-5:', int((rounds.notna() & ~rounds.between(1, 5)).sum()),
          'TIME missing/invalid:', int(elapsed.isna().sum()),
          'elapsed seconds range:', elapsed.min(), elapsed.max())


def main():
    start = time.perf_counter()
    protected = [p for folder in ('data', 'models') for p in (ROOT / folder).rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print('LOCAL FILE AUDIT (counts are per file, not added across copies)')
    for path in sorted((ROOT / 'data').rglob('*.csv')):
        frame = pd.read_csv(path, low_memory=False)
        relevant = [c for c in frame if c.lower() in ('method', 'round', 'time', 'time format')]
        print('\n', path.relative_to(ROOT), len(frame), 'rows;', relevant)
        for column in relevant:
            if column.lower() in ('method', 'time format'):
                print(column + ':\n' + counts(frame[column]))
        if 'ROUND' in frame and 'TIME' in frame:
            audit_rounds(frame, 'ROUND', 'TIME')
        elif 'round' in frame:
            audit_rounds(frame, 'round', 'time')
        elif 'ROUND' in frame:
            print('Fighter-round statistics, NOT fight ending rounds:\n' + counts(frame.ROUND))

    source = pd.read_csv(ROOT / 'data/processed/fights_dated.csv', low_memory=False)
    ml = pd.read_csv(ROOT / 'data/processed/prefight_features_activity.csv', low_memory=False)
    assert source.fight_id.is_unique and ml.fight_id.is_unique
    identity = ['fight_id', 'URL', 'event_date', 'fighter_1', 'fighter_2', 'outcome', 'winner', 'BOUT']
    joined = ml[identity].merge(source[['fight_id', *TARGETS, 'DETAILS']], on='fight_id', how='left',
                      validate='one_to_one', indicator=True)
    assert joined['_merge'].eq('both').all()
    original = source.set_index('fight_id').loc[ml.fight_id]
    for col in ['URL', 'event_date', 'fighter_1', 'fighter_2', 'outcome', 'winner']:
        pd.testing.assert_series_equal(ml[col].reset_index(drop=True), original[col].reset_index(drop=True))
    print('\nJOIN: matched', len(joined), '/', len(ml), '; ambiguous fight IDs: 0')
    print('Missing joined target fields:', joined[TARGETS].isna().sum().to_dict())
    print('Duplicate source URL excess rows:', int(ml.URL.duplicated().sum()))
    # Independently verify target agreement before applying the established deduplication.
    repeated = joined[joined.URL.duplicated(keep=False)]
    conflicts = repeated.groupby('URL')[TARGETS].nunique(dropna=False).gt(1).any(axis=1)
    print('Conflicting duplicated URL targets:', int(conflicts.sum()))
    assert not conflicts.any()
    raw = pd.read_csv(ROOT / 'data/raw/fights.csv')
    raw_conflicts = raw.groupby('fight_url')[['method', 'round', 'time']].nunique(dropna=False).gt(1).any(axis=1)
    print('Independent raw fights duplicate URL excess:', int(raw.fight_url.duplicated().sum()),
          '; conflicting target URL groups:', int(raw_conflicts.sum()))
    comparable = raw.loc[~raw.fight_url.isin(raw_conflicts[raw_conflicts].index)].drop_duplicates('fight_url')
    comparison = joined.merge(comparable[['fight_url', 'method', 'round', 'time']],
                              left_on='URL', right_on='fight_url', how='left', validate='many_to_one')
    matched = comparison.fight_url.notna()
    print('Independent raw URL join coverage:', int(matched.sum()), '/', len(joined))
    for left, right in [('METHOD', 'method'), ('ROUND', 'round'), ('TIME', 'time')]:
        a, b = comparison[left], comparison[right]
        if left == 'METHOD':
            a, b = a.map(normalize_method), b.map(normalize_method)
            usable = matched & a.notna() & b.notna()
        else:
            usable = matched
        mismatch = usable & a.ne(b)
        print('Independent raw disagreements', left, int(mismatch.sum()), '/', int(usable.sum()))
        if mismatch.any():
            print(comparison.loc[mismatch, ['BOUT', left, right]].to_string(index=False))
    baseline = runpy.run_path(str(ROOT / 'src/models/train_baseline.py'))
    binary, filtering = baseline['load_binary_fights']()
    data = joined.set_index('fight_id').loc[binary.fight_id].reset_index()
    data['event_date'] = pd.to_datetime(data.event_date)
    data['method_target'] = data.METHOD.map(normalize_method)
    data['round_target'] = pd.to_numeric(data.ROUND, errors='coerce')
    data['lengths'] = data['TIME FORMAT'].map(format_minutes)
    data['modern_schedule'] = data['TIME FORMAT'].isin(['3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)'])
    data['elapsed'] = data.TIME.map(seconds)
    def duration(row):
        if pd.isna(row.elapsed) or pd.isna(row.round_target) or row.round_target % 1:
            return None
        r = int(row.round_target)
        if row['TIME FORMAT'] == 'No Time Limit' and r == 1:
            return row.elapsed
        lengths = row.lengths
        if lengths is None or not 1 <= r <= len(lengths) or row.elapsed > lengths[r-1] * 60:
            return None
        return sum(lengths[:r-1]) * 60 + row.elapsed
    data['duration_seconds'] = data.apply(duration, axis=1)
    print('\nESTABLISHED BINARY FILTER:', filtering)
    print('METHOD eligible:', int(data.method_target.notna().sum()))
    table = data.method_target.value_counts().to_frame('count')
    table['percent_of_eligible'] = table['count'] / table['count'].sum() * 100
    print(table.to_string())
    print('Excluded binary methods:\n' + counts(data.loc[data.method_target.isna(), 'METHOD']))
    print('All source outcome/method combinations:\n' + pd.crosstab(source.METHOD, source.OUTCOME).to_string())
    unusual = source.DETAILS.str.contains('technical decision', case=False, na=False)
    print('Technical decision detail rows:\n' + source.loc[unusual, ['BOUT', 'METHOD', 'OUTCOME', 'ROUND', 'TIME', 'DETAILS']].to_string(index=False))
    finishes = data.method_target.isin(['KO/TKO', 'Submission'])
    valid_round = data.round_target.between(1, 5) & data.round_target.mod(1).eq(0)
    print('Finish rounds (decisions excluded):\n' + counts(data.loc[finishes, 'ROUND']))
    print('Eligible finishes with valid round:', int((finishes & valid_round).sum()),
          '; modern 3/5-round subset:', int((finishes & valid_round & data.modern_schedule & data.duration_seconds.notna()).sum()))
    print('Binary schedules:\n' + counts(data['TIME FORMAT']))
    print('Valid duration:', int(data.duration_seconds.notna().sum()), '/', len(data))
    print('Invalid duration rows:\n' + data.loc[data.duration_seconds.isna(), ['BOUT', *TARGETS]].to_string(index=False))
    print('Current decisions before nominal last period:', int(data.apply(
        lambda r: r.method_target == 'Decision' and r.lengths is not None and r.round_target < len(r.lengths), axis=1).sum()))
    print('Early-period decisions:\n' + data.loc[data.apply(
        lambda r: r.method_target == 'Decision' and r.lengths is not None and r.round_target < len(r.lengths),
        axis=1), ['BOUT', *TARGETS]].to_string(index=False))
    print('\nERA DISTRIBUTIONS: unique dated binary fights; percentages within eligible methods / finishes')
    data['era'] = pd.cut(data.event_date.dt.year, [1992, 2000, 2010, 2020, 2024, 2026],
                         labels=['1993-2000', '2001-2010', '2011-2020', '2021-2024', '2025-2026'])
    for era, group in data.groupby('era', observed=True):
        print(era, 'binary n=', len(group))
        print(pd.crosstab(group.era, group.method_target).to_string())
        print((group.method_target.value_counts(normalize=True)*100).round(2).to_dict())
        finish = group[group.method_target.isin(['KO/TKO', 'Submission'])]
        print('Finish rounds:', finish.ROUND.value_counts().sort_index().to_dict(),
              'percent:', (finish.ROUND.value_counts(normalize=True).sort_index()*100).round(2).to_dict())
    print('\nDEFENSIVE EVIDENCE: nonmissing opponent aggregates, not automatically complete rounds')
    for metric in ['sig_str_landed', 'td_landed', 'ctrl_seconds', 'sub_att']:
        columns = [f'fighter_{side}_{metric}' for side in (1, 2)]
        print(metric, 'both sides present:', int(original[columns].notna().all(axis=1).sum()), '/', len(original))
    for side in (1, 2):
        print('side', side, 'complete stats:', counts(original[f'fighter_{side}_stats_complete']))
    metadata = json.loads((ROOT / 'models/ufc_v1_metadata.json').read_text())
    assert not {x.casefold() for x in TARGETS} & {x.casefold() for x in metadata['features']}
    assert not set(TARGETS) & set(ml.columns)
    # Audit temporal eligibility only; do not construct new method features.
    history = source.assign(event_date=pd.to_datetime(source.event_date, errors='coerce')).dropna(subset=['event_date'])
    history = history.sort_values(['event_date', 'fight_id']).drop_duplicates('URL')
    side_rows = pd.concat([history[['event_date', f'fighter_{s}']].rename(columns={f'fighter_{s}': 'fighter'}) for s in (1, 2)])
    side_rows['fighter'] = side_rows.fighter.map(lambda x: ' '.join(x.split()).casefold())
    checked = 0
    for _, group in side_rows.groupby('fighter'):
        dates = group.event_date.sort_values().to_numpy()
        for current in dates:
            prior = dates[:dates.searchsorted(current, side='left')]
            assert (prior < current).all()
            checked += 1
    assert format_minutes('3 Rnd (5-5-5)') == [5, 5, 5]
    assert format_minutes('1 Rnd + 2OT (15-3-3)') == [15, 3, 3]
    assert format_minutes('No Time Limit') is None
    assert seconds('5:60') is None and normalize_method('DQ') is None
    print('Leakage audit PASSED:', checked, 'strict-date fighter histories; same-date/future excluded.')
    print('Frozen feature allowlist excludes current METHOD/ROUND/TIME; no new predictors constructed.')
    print('Caution: reversal publication dates absent; final results are not guaranteed known on fight date.')
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print('Protected data/models unchanged; elapsed seconds:', round(time.perf_counter()-start, 3))


if __name__ == '__main__':
    main()
