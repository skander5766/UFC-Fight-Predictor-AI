"""Application-only division/status catalog. Never filters training records.

Membership means historically OBSERVED division, not a current roster claim.
Historical labels are retained as named; changing historical weight limits are
not reinterpreted. Undated observations are excluded from as-of queries.
Status is unknown unless explicit verified evidence is supplied separately.
"""
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DIVISIONS=["Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight",
           "Super Heavyweight","Light Heavyweight","Heavyweight","Middleweight","Welterweight",
           "Lightweight","Featherweight","Bantamweight","Flyweight","Open Weight","Catch Weight"]
UNSPECIFIED='Unspecified historical division'


def name_key(name):
    if not isinstance(name,str) or not name.strip():raise ValueError('Expected a nonempty fighter name.')
    return ' '.join(name.split()).casefold()


def normalize_division(label):
    if pd.isna(label):return UNSPECIFIED
    text=' '.join(str(label).split()).replace('TitleBout','Title Bout')
    for division in DIVISIONS:
        # Longest specific phrases precede their substrings (women, light, super).
        if re.search(r'(?<![A-Za-z])'+re.escape(division)+r'(?![A-Za-z])',text,re.I):return division
    return UNSPECIFIED


def cutoff_date(value):
    if value is None:return None
    if isinstance(value,(int,float)):raise ValueError('Expected a calendar date.')
    day=pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None:raise ValueError('Expected a valid timezone-naive date.')
    return day.normalize()


class FighterCatalog:
    def __init__(self, fights=None, verified_status=None):
        if fights is None:fights=pd.read_csv(ROOT/'data/processed/fights_dated.csv',low_memory=False)
        self.raw_counts=fights.WEIGHTCLASS.value_counts(dropna=False)
        dates=pd.to_datetime(fights.event_date,format='%Y-%m-%d',errors='coerce')
        if (fights.event_date.notna() & dates.isna()).any():raise ValueError('Malformed event date.')
        self.names={}
        records=[]
        for side in (1,2):
            for name,label,day in zip(fights[f'fighter_{side}'],fights.WEIGHTCLASS,dates):
                key=name_key(name)
                self.names.setdefault(key,name)
                records.append((key,normalize_division(label),day))
        self.observations=pd.DataFrame(records,columns=['fighter','division','event_date']).drop_duplicates()
        self.status=[]
        # Explicit evidence adapter, never filled from inactivity or age.
        for record in verified_status or []:
            key=name_key(record['fighter'])
            if key not in self.names:raise ValueError('Unknown status identity; resolve explicitly first.')
            if record.get('verified') is not True or not record.get('source'):
                raise ValueError('Status needs verified evidence and a source reference.')
            if record['status'] not in ('retired','active'):raise ValueError('Unsupported explicit status.')
            effective=cutoff_date(record['effective_date'])
            verified=cutoff_date(record['verified_date'])
            if effective is None or verified is None:raise ValueError('Status evidence needs dates.')
            self.status.append(dict(record,key=key,effective=effective,verified_date=verified))

    def _observed(self,as_of_date=None):
        day=cutoff_date(as_of_date)
        if day is None:return self.observations
        return self.observations.loc[self.observations.event_date.notna() & self.observations.event_date.lt(day)]

    def _key(self,fighter):
        key=name_key(fighter)
        if key not in self.names:raise ValueError(f'Unknown fighter: {fighter}')
        return key

    def get_weight_classes(self,as_of_date=None):
        return sorted(self._observed(as_of_date).division.unique(),key=str.casefold)

    def get_fighters_for_weight_class(self,weight_class,as_of_date=None):
        if weight_class not in self.get_weight_classes():raise ValueError(f'Unknown application division: {weight_class}')
        rows=self._observed(as_of_date)
        return sorted((self.names[k] for k in rows.loc[rows.division.eq(weight_class),'fighter'].unique()),key=str.casefold)

    def get_fighter_weight_classes(self,fighter,as_of_date=None):
        rows=self._observed(as_of_date)
        return sorted(rows.loc[rows.fighter.eq(self._key(fighter)),'division'].unique(),key=str.casefold)

    def fighters_share_weight_class(self,fighter_a,fighter_b,as_of_date=None):
        if self._key(fighter_a)==self._key(fighter_b):return False
        shared=set(self.get_fighter_weight_classes(fighter_a,as_of_date)) & set(self.get_fighter_weight_classes(fighter_b,as_of_date))
        return bool(shared-{UNSPECIFIED})

    def is_retired(self,fighter,as_of_date=None):
        key=self._key(fighter)
        day=cutoff_date(as_of_date) if as_of_date is not None else pd.Timestamp.today().normalize()
        known=[r for r in self.status if r['key']==key and r['effective']<=day and r['verified_date']<=day]
        if not known:return None
        latest=max(r['effective'] for r in known)
        values={r['status'] for r in known if r['effective']==latest}
        return (next(iter(values))=='retired') if len(values)==1 else None

    def get_active_fighters_for_weight_class(self,weight_class,as_of_date=None,include_unknown=True):
        """Compatibility API: unknowns remain unless explicitly excluded.

        Returned names are NOT asserted active when include_unknown=True.
        Verified retirement hides only selector entries, not training fights.
        """
        return [name for name in self.get_fighters_for_weight_class(weight_class,as_of_date)
                if self.is_retired(name,as_of_date) is False or (include_unknown and self.is_retired(name,as_of_date) is None)]


@lru_cache(maxsize=1)
def default_catalog():return FighterCatalog()


def get_weight_classes(as_of_date=None):return default_catalog().get_weight_classes(as_of_date)
def get_fighters_for_weight_class(weight_class,as_of_date=None):return default_catalog().get_fighters_for_weight_class(weight_class,as_of_date)
def get_fighter_weight_classes(fighter,as_of_date=None):return default_catalog().get_fighter_weight_classes(fighter,as_of_date)
def fighters_share_weight_class(fighter_a,fighter_b,as_of_date=None):return default_catalog().fighters_share_weight_class(fighter_a,fighter_b,as_of_date)
def is_retired(fighter,as_of_date=None):return default_catalog().is_retired(fighter,as_of_date)
def get_active_fighters_for_weight_class(weight_class,as_of_date=None,include_unknown=True):
    return default_catalog().get_active_fighters_for_weight_class(weight_class,as_of_date,include_unknown)


if __name__=='__main__':
    catalog=default_catalog()
    print('Raw labels and source row counts:\n'+catalog.raw_counts.to_string())
    for division in catalog.get_weight_classes():print(division,len(catalog.get_fighters_for_weight_class(division)))
    for name in ['Conor McGregor','Georges St-Pierre','Amanda Nunes']:
        print(name,catalog.get_fighter_weight_classes(name))
    print('Retirement status: unknown without separate verified evidence. Training data is never filtered.')
