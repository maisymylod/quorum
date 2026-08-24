"""Audience generation: turning target marginals into a weighted synthetic population."""

from quorum.synthesis.ipf import RakingResult, rake
from quorum.synthesis.sampler import IndependenceSynthesizer, MicrodataSynthesizer
from quorum.synthesis.validate import FidelityReport, marginal_fidelity

__all__ = [
    "FidelityReport",
    "IndependenceSynthesizer",
    "MicrodataSynthesizer",
    "RakingResult",
    "marginal_fidelity",
    "rake",
]
