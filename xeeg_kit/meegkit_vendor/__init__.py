"""
Minimal vendor of meegkit functions: ASR, STAR, SNS.
Vendored from https://github.com/nschloe/meegkit (MIT License).
"""

from .asr import ASR
from .star import star
from .sns import sns

__all__ = ["ASR", "star", "sns"]
