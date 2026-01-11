# subgraph_matching.tuning

This directory provides hyperparameter tuning utilities for the `subgraph_matching` module.

Key goals:
- **Non-invasive**: core training works without tuner dependencies.
- **Modular**: a generic `BaseTuner` interface supports future tuners (random search, PBT, etc.).
- **No training-loop rewrite**: tuners call the existing training loop in `subgraph_matching/train.py`.

## Bayesian tuning

The Optuna tuner runs multiple training trials and optimizes a scalar objective.
By default it **maximizes Average Precision (AP)** (`avg_prec`), using the same metric computation as the existing validation logic.

### Install

Install Optuna only in order to run the Bayesian tuning:

```bash
pip install optuna # or install from requirements.txt
```

### Run

From the repo root:

```bash
python -m subgraph_matching.tune --tuning_method bayesian --tuning_n_trials 5 --tuning_metric avg_prec --tuning_output_dir tuning_runs 
```

Outputs:
- `tuning_runs/bayesian-*/trials.jsonl`: one JSON record per trial
- `tuning_runs/bayesian-*/best_config.json`: best trial params + score
- `tuning_runs/bayesian-*/best_command.txt`: copy/paste training command

During tuning, checkpoints are saved per trial under `tuning_runs/.../trial-####/model.pt`,
and the saved checkpoint is the **best-so-far** by the selected metric (`avg_prec` or `auroc` in this case).

## How this differs from grid search

- Grid search (TestTube) enumerates a fixed cartesian product of parameter values.
- Bayesian tuning (Optuna TPE) proposes new trials based on prior results to explore promising regions more efficiently.

Both are optional and do not change default training behavior of `subgraph_matching/train.py` overall implementation.

## Grid tuning (TestTube)

```bash
python -m subgraph_matching.tune --tuning_method grid --tuning_metric avg_prec --tuning_output_dir tuning_runs
```

Outputs:
- `tuning_runs/grid-*/trials.jsonl`
- `tuning_runs/grid-*/best_config.json`
- `tuning_runs/grid-*/best_command.txt`
