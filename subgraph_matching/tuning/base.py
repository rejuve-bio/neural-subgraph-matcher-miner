from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class BaseTuner(abc.ABC):
    """Abstract tuner interface.

    Implementations should:
    - propose hyperparameters via `suggest(trial)`
    - evaluate a trial via `objective(trial)`
    - orchestrate optimization via `run()`
    - persist the best config via `save_best_config()`

    This is generic enough to support Bayesian, random/grid search, PBT, etc.
    """

    @abc.abstractmethod
    def suggest(self, trial: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def objective(self, trial: Any) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def save_best_config(self) -> Optional[str]:
        raise NotImplementedError
