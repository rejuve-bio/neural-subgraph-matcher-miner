from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Defines which metric to optimize.
    For now, support single-objective optimization.
    this is structured to be extended to multi-objective later.
    """

    name: str = "avg_prec"  # AP , average precision. Other options: "auroc"


def extract_objective(metrics: Dict[str, float], spec: Optional[ObjectiveSpec] = None) -> float:
    """
    Extract a scalar objective from a metrics dict.
    Expected keys:
    - avg_prec (AP)
    - auroc
    Raises KeyError if metric is missing.
    """

    spec = spec or ObjectiveSpec()
    key = spec.name
    return float(metrics[key])
