"""Hyperparameter tuning entrypoint for subgraph_matching.

This module intentionally keeps tuning separate from training.
Usage:
  python -m subgraph_matching.tune --tuning_method bayesian
  python -m subgraph_matching.tune --tuning_method grid 
"""

from __future__ import annotations

import argparse
import os
import time

import torch.multiprocessing as mp

from common import utils
from subgraph_matching.config import parse_encoder as parse_train_encoder
from subgraph_matching.train import train_loop


def _peek_tuning_method(argv=None) -> str:
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument(
        "--tuning_method",
        default="bayesian",
        choices=["bayesian", "grid"] )
    
    known, _ = peek.parse_known_args(argv)
    return known.tuning_method


def _add_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tuning_method",
        default="bayesian",
        choices=["bayesian", "grid"],
        help="Hyperparameter tuning method",
        )
    parser.add_argument(
        "--tuning_metric",
        type=str,
        default="avg_prec",
        choices=["avg_prec", "auroc"],
        help="Metric to maximize (used for best checkpoint selection)",
        )
    parser.add_argument(
        "--tuning_output_dir",
        type=str,
        default="tuning_runs",
        help="Directory for tuner outputs (default: tuning_runs)",
        )
    parser.add_argument(
        "--tuning_n_trials",
        type=int,
        default=5,
        help="Number of tuning trials (Bayesian: default 25; Grid: default exhaustive)",
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (best-effort)",
        )

def _parse_args(argv=None):
    method = _peek_tuning_method(argv)

    if method == "grid":
        try:
            from test_tube import HyperOptArgumentParser
            from subgraph_matching.hyp_search import parse_encoder as parse_grid_encoder
        except Exception as e:
            raise RuntimeError("Grid tuning requires test_tube. Install it or use --tuning_method bayesian.") from e

        parser = HyperOptArgumentParser(strategy="grid_search")
        utils.parse_optimizer(parser)
        parse_grid_encoder(parser)
        _add_tuning_args(parser)
        return parser.parse_args(argv)

    else:
        parser = argparse.ArgumentParser(description="subgraph_matching tuning")
        utils.parse_optimizer(parser)
        parse_train_encoder(parser)
        _add_tuning_args(parser)
        return parser.parse_args(argv)

def main(argv=None) -> None:
    mp.set_start_method("spawn", force=True)

    args = _parse_args(argv)

    if args.tuning_method == "bayesian":
        from subgraph_matching.tuning.bayesian import BayesianTuningConfig, OptunaBayesianTuner

        cfg = BayesianTuningConfig(
            n_trials=args.tuning_n_trials,
            metric=args.tuning_metric,
            output_dir=args.tuning_output_dir,
            seed=args.seed )
        
        tuner = OptunaBayesianTuner(base_args=args, config=cfg)
        study = tuner.run()
        print(f"Best trial: {study.best_trial.number} value={study.best_trial.value}")
        print(f"Saved best config: {tuner.best_config_path}")
        return

    # Grid search
    else:
        from subgraph_matching.tuning.grid import GridTuningConfig, GridTuner

        cfg = GridTuningConfig(
            metric=args.tuning_metric,
            output_dir=args.tuning_output_dir,
            n_trials=args.tuning_n_trials,
            seed=args.seed )
        
        tuner = GridTuner(base_args=args, config=cfg, train_fn=train_loop)
        result = tuner.run()
        print(f"Best trial: {result['best_trial']} value={result['best_value']}")
        print(f"Saved best config: {result['best_config_path']}")


if __name__ == "__main__":
    main()
