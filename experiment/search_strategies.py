import os
import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pickle
import networkx as nx

# Configuration
DATASET = 'enzymes' # "data/<custom_dataset>.pkl" if downloaded a custom dataset
STRATEGIES = ["beam", "mcts", "greedy"]
TRIALS_GRID = [3, 5, 10]
NEIGHBORHOODS_GRID = [10, 20, 50, 100]
MIN_SIZE = 3
MAX_SIZE = 10
OUT_BATCH_SIZE = 4
BEAM_WIDTH = 2
RUN_TIMEOUT_SEC = 400
SKIP_ALREADY_DONE = True
SKIP_TIMEOUT_RUNS = True
PRUNE_HARDER_AFTER_TIMEOUT = True
EXPERIMENT_RESULTS_CSV = "results/enzymes_experiment_results.csv"
BEST_SUMMARY_TXT = "results/enzymes_best_summary.txt"
STRATEGY_VS_RUNTIME_PLOT = "plots/enzymes_strategy_vs_runtime.png"
CONFIG_VS_PATTERNS_PLOT = "plots/enzymes_config_vs_patterns.png"


def count_instances_from_all_instances_json(all_instances_json_path):
    """Count discovered pattern instances from *_all_instances.json.

    This gives a more informative signal than representative *.p outputs,
    which are capped by out_batch_size.
    """
    if not os.path.exists(all_instances_json_path):
        return None

    try:
        with open(all_instances_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: could not parse {all_instances_json_path}: {e}")
        return None

    if not isinstance(data, list):
        return None

    return sum(1 for item in data if isinstance(item, dict) and item.get("type") != "graph_context")


def read_existing_results(path):
    if not (SKIP_ALREADY_DONE and os.path.exists(path)):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_result_row(path, result):
    """Append one run result immediately for crash/interrupt safety."""
    row_df = pd.DataFrame([result])
    write_header = not os.path.exists(path)
    row_df.to_csv(path, mode="a", header=write_header, index=False)


def has_existing_run(df, strategy, dataset, n_trials, n_neighborhoods):
    if df.empty:
        return False
    done_statuses = ["success"]
    if SKIP_TIMEOUT_RUNS:
        done_statuses.append("timeout")
    mask = (
        (df["strategy"] == strategy)
        & (df["dataset"] == dataset)
        & (df["n_trials"] == n_trials)
        & (df["n_neighborhoods"] == n_neighborhoods)
        & (df["status"].isin(done_statuses))
    )
    return mask.any()


def has_timeout_on_easier_config(df, strategy, dataset, n_trials, n_neighborhoods):
    """Heuristic pruning: if an easier config timed out, skip harder ones."""
    if not PRUNE_HARDER_AFTER_TIMEOUT or df.empty:
        return False

    mask = (
        (df["strategy"] == strategy)
        & (df["dataset"] == dataset)
        & (df["status"] == "timeout")
        & (df["n_trials"] <= n_trials)
        & (df["n_neighborhoods"] <= n_neighborhoods)
    )
    return mask.any()


def run_experiment(strategy, dataset, n_trials, n_neighborhoods, min_pattern_size, max_pattern_size):
    print(f"Running strategy: {strategy} on {dataset} with {n_trials} trials and {n_neighborhoods} neighborhoods...")
    
    dataset_label = os.path.splitext(os.path.basename(str(dataset)))[0]
    out_stem = f"results/{strategy}_{dataset_label}_t{n_trials}_n{n_neighborhoods}"
    out_path = f"{out_stem}.p"
    all_instances_json_path = f"{out_stem}_all_instances.json"
    
    # decoder run command
    cmd = [
        "python", "-m", "subgraph_mining.decoder",
        f"--dataset={dataset}",
        f"--search_strategy={strategy}",
        f"--n_trials={n_trials}",
        f"--n_neighborhoods={n_neighborhoods}",
        f"--min_pattern_size={min_pattern_size}",
        f"--max_pattern_size={max_pattern_size}",
        f"--out_batch_size={OUT_BATCH_SIZE}",
        f"--out_path={out_path}",
        "--node_anchored"
    ]

    if strategy == "beam":
        cmd.append(f"--beam_width={BEAM_WIDTH}")

    start_time = time.time()
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=RUN_TIMEOUT_SEC,
        )
        runtime = time.time() - start_time
        # Prefer all-instances count for analysis; fallback to .p parsing.
        num_patterns = count_instances_from_all_instances_json(all_instances_json_path)

        return {
            "strategy": strategy,
            "dataset": dataset,
            "n_trials": n_trials,
            "n_neighborhoods": n_neighborhoods,
            "runtime(sec)": runtime,
            "num_patterns": num_patterns,
            "status": "success",
            "out_path": out_path,
        }
    except subprocess.TimeoutExpired:
        print(f"Timeout running {strategy} after {RUN_TIMEOUT_SEC}s")
        return {
            "strategy": strategy,
            "dataset": dataset,
            "n_trials": n_trials,
            "n_neighborhoods": n_neighborhoods,
            "runtime(sec)": time.time() - start_time,
            "num_patterns": 0,
            "status": "timeout",
            "out_path": "N/A",
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running {strategy}: {e.stderr}")
        return {
            "strategy": strategy,
            "dataset": dataset,
            "n_trials": n_trials,
            "n_neighborhoods": n_neighborhoods,
            "runtime(sec)": time.time() - start_time,
            "num_patterns": 0,
            "status": "failed",
            "out_path": "N/A",
        }


def plot_strategy_vs_runtime(df_strategy, dataset):
    if df_strategy.empty:
        print("No successful strategy runs to plot.")
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_strategy, x="strategy", y="avg_runtime", palette="Set2")
    plt.title(f"Search Strategy vs Runtime ({dataset})")
    plt.ylabel("Average Runtime (seconds)")
    plt.xlabel("Search Strategy")
    plt.tight_layout()
    plt.savefig(STRATEGY_VS_RUNTIME_PLOT, dpi=200)
    plt.close()


def plot_config_vs_patterns(df_grid, dataset):
    df_plot = df_grid[df_grid["status"] == "success"].copy()
    if df_plot.empty:
        print("No successful tuning runs to plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x="n_neighborhoods",
        y="num_patterns",
        hue="strategy",
        style="n_trials",
        s=120,
        palette="Set1",
    )
    plt.title(f"Configuration Tuning vs Patterns Found ({dataset})")
    plt.xlabel("Number of Neighborhoods")
    plt.ylabel("Number of Patterns Found")
    ax = plt.gca()
    unique_ns = sorted(df_plot["n_neighborhoods"].dropna().unique().tolist())
    if unique_ns:
        ax.set_xticks(unique_ns)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.tight_layout()
    plt.savefig(CONFIG_VS_PATTERNS_PLOT, dpi=200)
    plt.close()


def build_strategy_summary(df_grid):
    """Aggregate runs by strategy, counting success + timeout."""
    df_eval = df_grid[df_grid["status"].isin(["success", "timeout"])].copy()
    if df_eval.empty:
        return pd.DataFrame()

    return (
        df_eval.groupby("strategy", as_index=False)
        .agg(
            runs=("strategy", "count"),
            avg_runtime=("runtime(sec)", "mean"),
            avg_patterns=("num_patterns", "mean"),
        )
        .sort_values("strategy")
    )


def summarize_best(df_grid):
    successful = df_grid[df_grid["status"] == "success"].copy()
    if successful.empty:
        print("\nNo successful runs found.")
        return None

    best_config = successful.sort_values(
        ["num_patterns", "runtime(sec)"], ascending=[False, True]
    ).iloc[0]

    strategy_summary = build_strategy_summary(df_grid)
    best_algorithm = strategy_summary.sort_values(
        ["avg_patterns", "avg_runtime"], ascending=[False, True]
    ).iloc[0]

    print("\n--- Best Config (max patterns, then min runtime) ---")
    print(best_config[["strategy", "n_trials", "n_neighborhoods", "num_patterns", "runtime(sec)"]])

    print("\n--- Best Algorithm (avg over tuning grid) ---")
    print(best_algorithm[["strategy", "avg_patterns", "avg_runtime"]])

    return {
        "best_config": {
            "strategy": best_config["strategy"],
            "n_trials": int(best_config["n_trials"]),
            "n_neighborhoods": int(best_config["n_neighborhoods"]),
            "num_patterns": int(best_config["num_patterns"]),
            "runtime(sec)": float(best_config["runtime(sec)"]),
        },
        "best_algorithm": {
            "strategy": best_algorithm["strategy"],
            "avg_patterns": float(best_algorithm["avg_patterns"]),
            "avg_runtime": float(best_algorithm["avg_runtime"]),
        },
        "strategy_summary": [
            {
                "strategy": row["strategy"],
                "runs": int(row["runs"]),
                "avg_runtime": float(row["avg_runtime"]),
                "avg_patterns": float(row["avg_patterns"]),
            }
            for _, row in strategy_summary.iterrows()
        ],
    }


def write_best_summary(best_summary):
    if not best_summary:
        return

    best_config = best_summary["best_config"]
    best_algorithm = best_summary["best_algorithm"]
    strategy_summary = best_summary.get("strategy_summary", [])
    lines = [
        "Assignment Best Summary",
        "=======================",
        "",
        "Best Config (max patterns, then min runtime):",
        f"- strategy: {best_config['strategy']}",
        f"- n_trials: {best_config['n_trials']}",
        f"- n_neighborhoods: {best_config['n_neighborhoods']}",
        f"- num_patterns: {best_config['num_patterns']}",
        f"- runtime_seconds: {best_config['runtime(sec)']:.4f}",
        "",
        "Best Algorithm (avg over all successful configs):",
        f"- strategy: {best_algorithm['strategy']}",
        f"- avg_patterns: {best_algorithm['avg_patterns']:.4f}",
        f"- avg_runtime_seconds: {best_algorithm['avg_runtime']:.4f}",
    ]
    if strategy_summary:
        lines.extend([
            "",
            "Strategy Summary (all successful runs):",
        ])
        for row in strategy_summary:
            lines.append(
                f"- {row['strategy']}: runs={row['runs']}, avg_patterns={row['avg_patterns']:.4f}, avg_runtime_seconds={row['avg_runtime']:.4f}"
            )
    with open(BEST_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    sns.set_style("whitegrid")

    existing_grid = read_existing_results(EXPERIMENT_RESULTS_CSV)

    for strategy in STRATEGIES:
        for n_trials in TRIALS_GRID:
            for n_neighborhoods in NEIGHBORHOODS_GRID:
                if has_existing_run(existing_grid, strategy, DATASET, n_trials, n_neighborhoods):
                    print(
                        f"Skipping existing run: {strategy} on {DATASET} with trials={n_trials} neighborhoods={n_neighborhoods}"
                    )
                    continue
                if has_timeout_on_easier_config(existing_grid, strategy, DATASET, n_trials, n_neighborhoods):
                    print(
                        f"Skipping by timeout pruning: {strategy} on {DATASET} with trials={n_trials} neighborhoods={n_neighborhoods}"
                    )
                    pruned_result = {
                        "strategy": strategy,
                        "dataset": DATASET,
                        "n_trials": n_trials,
                        "n_neighborhoods": n_neighborhoods,
                        "runtime(sec)": float(RUN_TIMEOUT_SEC),
                        "num_patterns": 0,
                        "status": "timeout",
                        "out_path": "N/A",
                    }
                    existing_grid = pd.concat([existing_grid, pd.DataFrame([pruned_result])], ignore_index=True)
                    append_result_row(EXPERIMENT_RESULTS_CSV, pruned_result)
                    continue
                result = run_experiment(
                    strategy,
                    DATASET,
                    n_trials,
                    n_neighborhoods,
                    MIN_SIZE,
                    MAX_SIZE,
                )

                # Update in-memory table so timeout pruning works immediately.
                existing_grid = pd.concat([existing_grid, pd.DataFrame([result])], ignore_index=True)
                append_result_row(EXPERIMENT_RESULTS_CSV, result)

    df_grid = existing_grid.copy()

    if not df_grid.empty:
        df_grid = df_grid.sort_values(["strategy", "n_trials", "n_neighborhoods"]).drop_duplicates(
            subset=["strategy", "dataset", "n_trials", "n_neighborhoods"], keep="last"
        )

    strategy_summary = build_strategy_summary(df_grid)
    print("\n--- Strategy Summary (all successful runs) ---")
    if not strategy_summary.empty:
        print(strategy_summary)
    else:
        print("No successful runs yet.")

    print("\n--- Tuning Summary Table ---")
    print(df_grid)
    df_grid.to_csv(EXPERIMENT_RESULTS_CSV, index=False)

    plot_strategy_vs_runtime(strategy_summary, DATASET)
    plot_config_vs_patterns(df_grid, DATASET)
    best_summary = summarize_best(df_grid)
    write_best_summary(best_summary)

    print("\nSaved:")
    print(f"- {EXPERIMENT_RESULTS_CSV}")
    print(f"- {BEST_SUMMARY_TXT}")
    print(f"- {STRATEGY_VS_RUNTIME_PLOT}")
    print(f"- {CONFIG_VS_PATTERNS_PLOT}")

if __name__ == "__main__":
    main()