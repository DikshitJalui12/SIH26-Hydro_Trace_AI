"""
Quality Gating, Lookalike Discrimination, Wind Gating, and Fay's Spreading Theory.
"""
from .era5_wind_gate import ERA5WindGater, WindGatingResult
from .polarization_features import PolarimetricFeatureExtractor
from .lookalike_classifier import LookalikeClassifier, SlickClassification
from .fays_spreading import FaysSpreadingEstimator, SpreadingRegime

__all__ = [
    "ERA5WindGater",
    "WindGatingResult",
    "PolarimetricFeatureExtractor",
    "LookalikeClassifier",
    "SlickClassification",
    "FaysSpreadingEstimator",
    "SpreadingRegime",
]
