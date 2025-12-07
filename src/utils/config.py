"""
Global configuration values for the phishing detector.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    # Overall phishing score thresholds
    PHISH_HIGH_THRESHOLD: float = 0.70   # clearly phishing
    PHISH_LOW_THRESHOLD: float = 0.35    # clearly benign below this


CONFIG = DetectorConfig()
