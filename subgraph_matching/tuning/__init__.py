"""Optional hyperparameter tuning utilities for subgraph_matching.

This package is intentionally isolated so that core training can run without
extra tuner dependencies.
"""

from .base import BaseTuner

__all__ = ["BaseTuner"]
