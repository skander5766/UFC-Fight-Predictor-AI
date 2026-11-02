"""Read-only inference for evaluated winner D and METHOD-M; no model fitting.

Reuse validated history summaries and Elo updates. Snapshots contain post-date
ratings, accessed only with a strict earlier-date binary search. Same-date
updates use frozen ratings. Clear the cache to load a new local data snapshot.
"""
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import contextlib
import io
import json
import runpy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = runpy.run_path(str(ROOT/'src/prediction/build_matchup_features.py'))
V2 = runpy.run_path(str(ROOT/'src/features/build_v2_features.py'))
WINNER = runpy.run_path(str(ROOT/'src/models/backtest_v2_winner.py'))
METHOD = runpy.run_path(str(ROOT/'src/models/backtest_method_model.py'))
WINNER_FEATURES = tuple(WINNER['families']()['D'])
METHOD_FEATURES = tuple(x+'_diff' for x in METHOD['groups']()['METHOD-M'])
assert len(WINNER_FEATURES)==62 and len(METHOD_FEATURES)==47


class V2MatchupFeatureBuilder:
    def __init__(self, source=None):
        if source is None:
            source = pd.read_csv(V2['SOURCE'],low_memory=False)
        self.base = OLD['MatchupFeatureBuilder'](source)
        source = source.copy()
        source['event_date'] = pd.to_datetime(source.event_date,errors='raise').dt.normalize()
        source = source.dropna(subset=['event_date']).sort_values(['event_date','fight_id'])
        fields = ['METHOD','ROUND','TIME','TIME FORMAT','fighter_1_stats_complete','fighter_2_stats_complete']
        if source.groupby('URL')[fields].nunique(dropna=False).gt(1).any().any():
            raise ValueError('Conflicting duplicate targets or completeness.')
        self.histories = defaultdict(list)
        self.rating_dates = defaultdict(list)
        self.rating_values = defaultdict(list)
        ratings = defaultdict(lambda:1500.)
        for date,batch in source.drop_duplicates('URL').groupby('event_date',sort=True):
            deltas = defaultdict(float)
            for row in batch.to_dict('records'):
                a,b = [OLD['HISTORY']['fighter_key'](row[f'fighter_{s}']) for s in (1,2)]
                first = V2['make_entry'](row,1,ratings[b])
                second = V2['make_entry'](row,2,ratings[a])
                self.histories[a].append(first)
                self.histories[b].append(second)
                if first['eligible']:
                    delta = V2['elo_delta'](ratings[a],ratings[b],float(first['won']))
                    deltas[a] += delta
                    deltas[b] -= delta
            for name,delta in deltas.items():
                ratings[name] += delta
                self.rating_dates[name].append(date)
                self.rating_values[name].append(ratings[name])
        self.dates = {k:[r['date'] for r in v] for k,v in self.histories.items()}

    def fighter_features(self, fighter, prediction_date):
        date = OLD['calendar_date'](prediction_date)
        key = self.base.resolve(fighter)
        stop = bisect_left(self.dates.get(key,[]),date)
        history = self.histories.get(key,[])[:stop]
        rating_stop = bisect_left(self.rating_dates.get(key,[]),date)
        rating = self.rating_values[key][rating_stop-1] if rating_stop else 1500.
        assert all(r['date']<date for r in history)
        result = self.base.fighter_features(fighter,date)
        result.update(V2['summarize'](history,rating))
        return result

    def _build(self, first, second, date, columns):
        if self.base.resolve(first)==self.base.resolve(second):
            raise ValueError('A fighter cannot be matched against the same normalized name.')
        a,b = self.fighter_features(first,date),self.fighter_features(second,date)
        row = {}
        for column in columns:
            key = 'prefight_elo' if column=='elo_diff' else OLD['ENRICHED']['ALIASES'].get(column[:-5],column[:-5])
            row[column] = np.nan if pd.isna(a[key]) or pd.isna(b[key]) else float(a[key])-float(b[key])
        return pd.DataFrame([row],columns=columns,dtype=float)

    def build_winner_features(self, fighter_a, fighter_b, prediction_date):
        return self._build(fighter_a,fighter_b,prediction_date,WINNER_FEATURES)

    def build_method_features(self, candidate, opponent, prediction_date):
        return self._build(candidate,opponent,prediction_date,METHOD_FEATURES)


@lru_cache(maxsize=1)
def default_builder():
    return V2MatchupFeatureBuilder()


def clear_builder_cache():
    default_builder.cache_clear()


def build_winner_features(fighter_a, fighter_b, prediction_date):
    return default_builder().build_winner_features(fighter_a,fighter_b,prediction_date)


def build_method_features(candidate, opponent, prediction_date):
    return default_builder().build_method_features(candidate,opponent,prediction_date)


def validate_historical(builder, sample_count=160):
    winner_report = json.loads((ROOT/'models/v2_winner_backtest_metrics.json').read_text())
    method_report = json.loads((ROOT/'models/method_backtest_metrics.json').read_text())
    assert list(WINNER_FEATURES)==winner_report['features']['D']
    assert list(METHOD_FEATURES)==method_report['features']['METHOD-M']
    frame = pd.read_csv(ROOT/'data/processed/prefight_features_v2.csv',low_memory=False)
    frame = frame.loc[frame.outcome.isin(['fighter_1_win','fighter_2_win'])].sort_values(['event_date','fight_id']).drop_duplicates('URL')
    sample = frame.iloc[np.unique(np.linspace(0,len(frame)-1,min(len(frame),sample_count),dtype=int))].copy()
    for col in OLD['ENRICHED']['LAYOFF']:
        name=col[:-5]
        sample[col]=OLD['ENRICHED']['boolean_values'](sample['fighter_1_'+name])-OLD['ENRICHED']['boolean_values'](sample['fighter_2_'+name])
    method_expected = METHOD['candidate_features'](sample,METHOD['groups']()['METHOD-M'],sample.outcome.map({'fighter_1_win':1,'fighter_2_win':2}))
    results = {}
    for label,columns,fn in [('winner',WINNER_FEATURES,builder.build_winner_features),('method',METHOD_FEATURES,builder.build_method_features)]:
        exact=tolerance=0
        maximum=0.
        for index,row in sample.iterrows():
            a,b=row.fighter_1,row.fighter_2
            expected=row[list(columns)].to_numpy(dtype=float) if label=='winner' else method_expected.loc[index].to_numpy()
            if label=='method' and row.outcome=='fighter_2_win': a,b=b,a
            actual=fn(a,b,row.event_date).iloc[0].to_numpy()
            same=(actual==expected)|(np.isnan(actual)&np.isnan(expected))
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10,equal_nan=True)
            exact+=int(same.all()); tolerance+=int(not same.all())
            finite=np.isfinite(actual)&np.isfinite(expected)
            maximum=max(maximum,float(np.max(np.abs(actual[finite]-expected[finite]))))
            np.testing.assert_allclose(fn(b,a,row.event_date).iloc[0],-actual,rtol=0,atol=0,equal_nan=True)
        results[label]={'sample':len(sample),'exact':exact,'tolerance_only':tolerance,'mismatches':0,'max_difference':maximum,'swap_error':0}
    return results


if __name__=='__main__':
    print(validate_historical(default_builder()))
