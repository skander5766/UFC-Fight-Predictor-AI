"""Build empirical schedule prior through the inclusive production cutoff."""
import hashlib
import json
from pathlib import Path
import runpy
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
API=runpy.run_path(str(ROOT/'src/prediction/round_prior.py'))
CUTOFF='2026-05-16'


def build_prior(source):
    normalize=runpy.run_path(str(ROOT/'src/data/analyze_method_round_targets.py'))['normalize_method']
    source=source.copy()
    source['event_date']=pd.to_datetime(source.event_date,errors='raise')
    source=source.loc[source.event_date.notna() & source.event_date.le(CUTOFF)].copy()
    if source.URL.isna().any(): raise ValueError('Missing source identity')
    fields=['event_date','fighter_1','fighter_2','outcome','METHOD','ROUND','TIME FORMAT']
    if source.groupby('URL')[fields].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting source aliases')
    aliases=int(source.URL.duplicated().sum())
    source=source.sort_values(['event_date','fight_id']).drop_duplicates('URL')
    source=source.loc[source.outcome.isin(['fighter_1_win','fighter_2_win']) & source.METHOD.map(normalize).isin(['KO/TKO','Submission'])].copy()
    finishes=len(source)
    source['schedule']=source['TIME FORMAT'].map({'3 Rnd (5-5-5)':3,'5 Rnd (5-5-5-5-5)':5})
    excluded=int(source.schedule.isna().sum())
    source=source.loc[source.schedule.notna()].copy()
    rounds=pd.to_numeric(source.ROUND,errors='raise')
    if not (rounds.notna() & rounds.mod(1).eq(0) & rounds.ge(1) & rounds.le(source.schedule)).all():
        raise ValueError('Invalid ending round for schedule')
    source['round_number']=rounds.astype(int)
    artifact={'model_id':'ufc_round_prior','version':'1.0.0','type':'empirical_frequency_table',
              'cutoff':CUTOFF,'cutoff_inclusive':True,'round_order':['ROUND_1','ROUND_2','ROUND_3','ROUND_4','ROUND_5'],
              'eligible_finishes':finishes,'excluded_other_formats':excluded,'duplicate_dated_source_rows_removed':aliases,
              'definitions':{'target':'P(round | finish, scheduled_rounds)',
                             'finish':'Known binary winner; KO/TKO including doctor stoppage, or Submission; decisions/DQ/draw/NC/overturned excluded.',
                             'schedule':'Exact TIME FORMAT: 3 Rnd (5-5-5) or 5 Rnd (5-5-5-5-5); not inferred from ending round.',
                             'estimator':'Per-schedule round count / total eligible finishes; no smoothing, fitting, or ML.',
                             'unsupported_schedule':'No estimate (None).'},
              'limitations':['Population prior, not fighter-specific or method-specific.',
                             'Round 4/5 evidence is sparse; historical format and era differences remain.',
                             'Fixed cutoff includes all eligible history through that date; not safe as an earlier as-of prior.',
                             'Only local dated observed UFC history; retrospective corrections lack publication dates.'],
              'schedules':{}}
    for schedule in (3,5):
        subset=source.loc[source.schedule.eq(schedule)]
        counts=subset.round_number.value_counts().reindex(range(1,6),fill_value=0)
        artifact['schedules'][str(schedule)]={'total':len(subset),'counts':counts.tolist(),
                                              'probabilities':(counts/len(subset)).tolist()}
    API['validate_prior'](artifact)
    return artifact


def main():
    output=API['ARTIFACT']
    if output.exists(): raise FileExistsError(output)
    protected=[p for folder in ('data','models','app') for p in (ROOT/folder).rglob('*') if p.is_file()]
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    for stem in ('ufc_winner_d_v2_0_0','ufc_method_m_v1_0_0'):
        assert json.loads((ROOT/'models'/f'{stem}_metadata.json').read_text())['training_cutoff']==CUTOFF
    path=ROOT/'data/processed/fights_dated.csv'
    artifact=build_prior(pd.read_csv(path,low_memory=False))
    artifact['source']='data/processed/fights_dated.csv'
    artifact['source_sha256']=hashes[path]
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    output.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(artifact['schedules'],indent=2))


if __name__=='__main__': main()
