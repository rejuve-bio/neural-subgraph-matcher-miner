from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class GridTuningConfig:
    metric: str = "avg_prec"
    output_dir: str = "tuning_runs"
    n_trials: Optional[int] = None  # None = exhaustive
    seed: Optional[int] = None


class GridTuner:
    """TestTube-based grid search runner.
    This keeps grid tuning out of `subgraph_matching.train`. so that training won't go for grid-search trials on each run to train a model as the initial implementation.

    Notes:
    - Expects `base_args` to be the object returned by TestTube's parser.
    - Ensures per-trial output isolation and best-by-metric checkpointing.
    """

    def __init__(self, base_args: Any, config: Optional[GridTuningConfig] = None, train_fn: Optional[Callable[[Any], None]] = None):
        self.base_args = base_args
        self.config = config or GridTuningConfig()
        self.train_fn = train_fn

        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.config.output_dir, f"grid-{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.trials_path = os.path.join(self.run_dir, "trials.jsonl")
        self.best_config_path = os.path.join(self.run_dir, "best_config.json")
        self.best_command_path = os.path.join(self.run_dir, "best_command.txt")

    def _trial_args(self, trial_number: int, args: Any) -> Any:
        trial_args = copy.deepcopy(args)

        trial_dir = os.path.join(self.run_dir, f"trial-{trial_number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        if hasattr(trial_args, "model_path"):
            trial_args.model_path = os.path.join(trial_dir, "model.pt")

        # Enable best-by-metric checkpointing inside validation()
        trial_args.save_best = True
        trial_args.save_best_metric = self.config.metric

        # Ensure at least one validation happens.
        if hasattr(trial_args, "n_batches") and hasattr(trial_args, "eval_interval"):
            if trial_args.n_batches is not None and trial_args.eval_interval is not None:
                if int(trial_args.n_batches) < int(trial_args.eval_interval):
                    trial_args.eval_interval = max(1, int(trial_args.n_batches))

        if self.config.seed is not None:
            trial_args.seed = int(self.config.seed) + int(trial_number)

        if hasattr(trial_args, "tag"):
            base_tag = getattr(trial_args, "tag", "") or ""
            trial_args.tag = (base_tag + "-" if base_tag else "") + f"grid-t{trial_number:04d}"

        return trial_args

    def _objective_from_trial(self, trial_args: Any) -> float:
        metrics = getattr(trial_args, "_best_metrics", None)
        if not metrics:
            return float("nan")
        return float(metrics[self.config.metric])

    def _write_best_command(self, best_args: Any) -> None:
        keys = [
            "conv_type",
            "skip",
            "method_type",
            "n_layers",
            "hidden_dim",
            "dropout",
            "margin",
            "lr",
            "weight_decay",
            "opt",
            "opt_scheduler",
            "batch_size",
            "n_batches",
            "eval_interval",
            "val_size",
            "dataset",
            "node_anchored",
        ]
        parts = ["python -m subgraph_matching.train"]
        for key in keys:
            if not hasattr(best_args, key):
                continue
            value = getattr(best_args, key)
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    parts.append(f"--{key}")
                continue
            parts.append(f"--{key} {value}")

        # Do not reuse tuning checkpoint path by default
        parts.append("--model_path ckpt/model.pt")

        with open(self.best_command_path, "w", encoding="utf-8") as f:
            f.write(" ".join(parts) + "\n")

    def run(self) -> Dict[str, Any]:
        if self.train_fn is None:
            from subgraph_matching.train import train_loop

            self.train_fn = train_loop

        best_trial = None
        best_value = None
        best_args = None

        for i, hparam_trial in enumerate(self.base_args.trials(self.config.n_trials)):
            print("Running hyperparameter search trial", i)
            print("-" * 100)
            print(hparam_trial)

            trial_args = self._trial_args(i, hparam_trial)
            self.train_fn(trial_args)

            objective = self._objective_from_trial(trial_args)
            metrics = getattr(trial_args, "_best_metrics", None)

            record = {
                "trial": i,
                "objective": objective,
                "metric": self.config.metric,
                "metrics": metrics,
                "model_path": getattr(trial_args, "model_path", None),
                "params": {k: v for k, v in vars(trial_args).items() if not str(k).startswith("_")},
            }
            with open(self.trials_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

            if objective == objective:  # NaN check
                if best_value is None or objective > best_value:
                    best_value = objective
                    best_trial = i
                    best_args = trial_args

        if best_trial is None or best_args is None:
            payload = {
                "best_trial": None,
                "best_value": None,
                "metric": self.config.metric,
                "run_dir": self.run_dir,
            }
        else:
            payload = {
                "best_trial": int(best_trial),
                "best_value": float(best_value),
                "metric": self.config.metric,
                "run_dir": self.run_dir,
                "best_params": {k: v for k, v in vars(best_args).items() if not str(k).startswith("_")},
                "best_metrics": getattr(best_args, "_best_metrics", None),
            }

        with open(self.best_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        if best_args is not None:
            self._write_best_command(best_args)

        return {
            "best_trial": payload.get("best_trial"),
            "best_value": payload.get("best_value"),
            "best_config_path": self.best_config_path,
            "run_dir": self.run_dir,
        }
