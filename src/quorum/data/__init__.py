"""Ground-truth data: the shared attribute taxonomy, harmonization, and loaders.

Two independent public sources meet here, which is what keeps the accuracy claim
honest. The population is synthesized from census marginals; the answers it is scored
against come from a survey the synthesis never sees. They only need to agree on one
thing, the attribute taxonomy in :mod:`quorum.data.schema`, and the harmonizers in
:mod:`quorum.data.harmonize` are what force each source onto it.
"""

from quorum.data.schema import ATTRIBUTES, LEVELS, age_band, validate_levels
from quorum.data.targets import MarginalTargets, Question, QuestionBank

__all__ = [
    "ATTRIBUTES",
    "LEVELS",
    "MarginalTargets",
    "Question",
    "QuestionBank",
    "age_band",
    "validate_levels",
]
