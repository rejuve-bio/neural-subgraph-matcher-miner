from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseTuner
from .objectives import ObjectiveSpec, extract_objective


@dataclass
class BayesianTuningConfig:
    n_trials: int = 25
    metric: str = "avg_prec"  # maximize it
    output_dir: str = "tuning_runs"
    seed: Optional[int] = None


class OptunaBayesianTuner(BaseTuner):
    """Optuna-based Bayesian hyperparameter tuner.

    - This tuner calls the existing training entrypoint in `subgraph_matching.train`
      and does NOT rewrite the training loop.
    """

    def __init__(self, base_args: Any, config: Optional[BayesianTuningConfig] = None):
        self.base_args = base_args
        self.config = config or BayesianTuningConfig()
        self._best_trial = None
        self._study = None

        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.config.output_dir, f"bayesian-{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.trials_path = os.path.join(self.run_dir, "trials.jsonl")
        self.best_config_path = os.path.join(self.run_dir, "best_config.json")

    def _require_optuna(self):
        try:
            import optuna  # type: ignore
        except Exception as e:
            raise RuntimeError("Optuna is required for Bayesian tuning. Install with: pip install optuna") from e
        return optuna

    def suggest(self, trial: Any) -> Dict[str, Any]:
        """
        tune a conservative set of existing hyperparameters:
        - lr
        - weight_decay
        - hidden_dim
        - n_layers
        - dropout
        - opt

        not tune model type / embedding type.
        """

        params: Dict[str, Any] = {}

        # learning rate
        if hasattr(self.base_args, "lr"):
            params["lr"] = trial.suggest_float("lr", 1e-5, 3e-3, log=True)

        if hasattr(self.base_args, "weight_decay"):
            params["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 1e-3, log=False)

        if hasattr(self.base_args, "hidden_dim"):
            params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])

        if hasattr(self.base_args, "n_layers"):
            params["n_layers"] = trial.suggest_categorical("n_layers", [4, 8, 12])

        if hasattr(self.base_args, "dropout"):
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

        if hasattr(self.base_args, "opt"):
            # Supported by common.utils.build_optimizer
            params["opt"] = trial.suggest_categorical("opt", ["adam", "sgd", "rmsprop", "adagrad"])

        return params

    def _trial_args(self, trial: Any, suggested: Dict[str, Any]) -> Any:
        args = copy.deepcopy(self.base_args)

        # Per-trial output isolation
        trial_dir = os.path.join(self.run_dir, f"trial-{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        # ensuring don't overwrite the user's default ckpt/model.pt during sweeps
        if hasattr(args, "model_path"):
            args.model_path = os.path.join(trial_dir, "model.pt")

        # Enable best-by-metric checkpointing inside validation()
        args.save_best = True
        args.save_best_metric = self.config.metric

        # Ensure at least one validation happens.
        if hasattr(args, "n_batches") and hasattr(args, "eval_interval"):
            if args.n_batches is not None and args.eval_interval is not None:
                if int(args.n_batches) < int(args.eval_interval):
                    args.eval_interval = max(1, int(args.n_batches))

        # Keep tag readable if present
        if hasattr(args, "tag"):
            base_tag = getattr(args, "tag", "") or ""
            args.tag = (base_tag + "-" if base_tag else "") + f"bayes-t{trial.number:04d}"

        # Apply suggested hyperparameters
        for k, v in suggested.items():
            if hasattr(args, k):
                setattr(args, k, v)

        # Optional seeding (but best so trials are reproducible or consistent). 
        if self.config.seed is not None:
            trial_seed = int(self.config.seed) + int(trial.number)
            if hasattr(args, "seed"):
                args.seed = trial_seed

        return args

    def _set_global_seeds(self, seed: int) -> None:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _evaluate(self, args: Any) -> Dict[str, float]:
        """
        Run training and return best-by-metric validation metrics.
        When `args.save_best` is enabled, `subgraph_matching.test.validation` will
        track `args._best_metrics` at the checkpoint that maximizes the selected
        metric.
        """

        from subgraph_matching import train as train_module
        import torch.multiprocessing as mp

        if getattr(args, "seed", None) is not None:
            self._set_global_seeds(int(args.seed))

        # Mirror `subgraph_matching.train.main` behavior.
        mp.set_start_method("spawn", force=True)

        # Train using existing logic
        train_module.train_loop(args)

        best_metrics = getattr(args, "_best_metrics", None)
        if best_metrics:
            return {
                "auroc": float(best_metrics.get("auroc")),
                "avg_prec": float(best_metrics.get("avg_prec")),
            }

        # Fallback: if best-metric tracking was not enabled, return nans.
        return {"auroc": float("nan"), "avg_prec": float("nan")}

    def objective(self, trial: Any) -> float:
        suggested = self.suggest(trial)
        args = self._trial_args(trial, suggested)

        metrics = self._evaluate(args)
        objective_value = extract_objective(metrics, ObjectiveSpec(name=self.config.metric))

        record = {
            "trial": trial.number,
            "params": suggested,
            "metrics": metrics,
            "objective": objective_value,
            "model_path": getattr(args, "model_path", None),
        }
        with open(self.trials_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return objective_value

    def run(self) -> Any:
        optuna = self._require_optuna()

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._study.optimize(self.objective, n_trials=self.config.n_trials)
        self._best_trial = self._study.best_trial

        self.save_best_config()
        return self._study

    def save_best_config(self) -> Optional[str]:
        if self._best_trial is None:
            return None

        payload = {
            "best_trial": int(self._best_trial.number),
            "best_value": float(self._best_trial.value),
            "best_params": dict(self._best_trial.params),
            "metric": self.config.metric,
            "run_dir": self.run_dir,
        }
        with open(self.best_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Also write a copy/paste-friendly training command.
        cmd_path = os.path.join(self.run_dir, "best_command.txt")
        parts = ["python -m subgraph_matching.train"]
        for k, v in self._best_trial.params.items():
            parts.append(f"--{k} {v}")
        parts.append("--model_path ckpt/model.pt")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(" ".join(parts) + "\n")

        return self.best_config_path
