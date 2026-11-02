"""Versioned empirical P(round | finish, scheduled rounds), no ML."""
import json
import math
from numbers import Integral
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2]/'models/ufc_round_prior_v1_0_0.json'


def validate_prior(artifact):
    for schedule in (3,5):
        entry = artifact['schedules'][str(schedule)]
        counts, probabilities = entry['counts'], entry['probabilities']
        if len(counts)!=5 or len(probabilities)!=5 or any(type(c) is not int or c<0 for c in counts):
            raise ValueError('Invalid round counts')
        if sum(counts)!=entry['total'] or entry['total']<=0:
            raise ValueError('Invalid prior denominator')
        if any(not math.isfinite(p) or p<0 or p>1 for p in probabilities):
            raise ValueError('Invalid round probabilities')
        if not math.isclose(sum(probabilities),1,abs_tol=1e-12):
            raise ValueError('Probabilities must sum to one')
        if any(not math.isclose(p,c/entry['total'],abs_tol=1e-12) for c,p in zip(counts,probabilities)):
            raise ValueError('Probabilities disagree with counts')
        if schedule==3 and (counts[3:]!=[0,0] or probabilities[3:]!=[0.,0.]):
            raise ValueError('Three-round schedule contains late-round mass')


def get_round_prior(scheduled_rounds, artifact_path=ARTIFACT):
    """Return R1-R5 probabilities, or None for unknown/unsupported schedules.

Accept explicit integer 3 or 5 only; never infer a schedule from ending round.
This fixed-cutoff population prior is not a historical as-of-date estimate.
"""
    if isinstance(scheduled_rounds,bool) or not isinstance(scheduled_rounds,Integral) or scheduled_rounds not in (3,5):
        return None
    artifact=json.loads(Path(artifact_path).read_text(encoding='utf-8'))
    validate_prior(artifact)
    return list(artifact['schedules'][str(scheduled_rounds)]['probabilities'])
