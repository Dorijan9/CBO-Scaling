"""
Plot scaling experiment results: how CBO performance degrades with graph size.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_scaling_results(results_path: str):
    """Generate scaling experiment plots."""
    with open(results_path) as f:
        results = json.load(f)

    sizes = sorted(results.keys(), key=lambda k: results[k]["n_variables"])
    n_vars = [results[k]["n_variables"] for k in sizes]
    n_edges = [results[k]["n_edges"] for k in sizes]

    correct_rates = [results[k]["correct_rate"] for k in sizes]
    mean_shd = [results[k]["mean_shd"] for k in sizes]
    std_shd = [results[k]["std_shd"] for k in sizes]
    mean_f1 = [results[k]["mean_f1"] for k in sizes]
    std_f1 = [results[k]["std_f1"] for k in sizes]
    mean_iters = [results[k]["mean_iterations"] for k in sizes]
    mean_wrmse = [results[k]["mean_weight_rmse"] for k in sizes]
    mean_wcov = [results[k]["mean_weight_coverage"] for k in sizes]
    mean_map_prob = [results[k]["mean_map_prob"] for k in sizes]
    conv_rates = [results[k]["convergence_rate"] for k in sizes]

    Path("plots").mkdir(exist_ok=True)

    # Figure 1: Main scaling summary (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CBO Recovery Performance vs Graph Size", fontsize=16, fontweight="bold")

    # (a) Correct recovery rate
    ax = axes[0, 0]
    bars = ax.bar(range(len(sizes)), correct_rates, color=["#2ecc71" if r >= 0.8 else "#e74c3c" if r < 0.5 else "#f39c12" for r in correct_rates],
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{n}v\n{e}e" for n, e in zip(n_vars, n_edges)])
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Recovery Accuracy")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(correct_rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")

    # (b) SHD
    ax = axes[0, 1]
    ax.errorbar(n_vars, mean_shd, yerr=std_shd, fmt="o-", color="#3498db",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Structural Hamming Distance")
    ax.set_title("(b) Structural Error (SHD)")
    ax.set_xticks(n_vars)

    # (c) F1 Score
    ax = axes[1, 0]
    ax.errorbar(n_vars, mean_f1, yerr=std_f1, fmt="s-", color="#9b59b6",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Edge F1 Score")
    ax.set_title("(c) Edge Recovery (F1)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(n_vars)

    # (d) Iterations to convergence
    ax = axes[1, 1]
    ax.bar(range(len(sizes)), mean_iters, color="#1abc9c", edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{n}v\n{e}e" for n, e in zip(n_vars, n_edges)])
    ax.set_ylabel("Mean Iterations")
    ax.set_title("(d) Sample Efficiency")
    for i, v in enumerate(mean_iters):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig("plots/scaling_summary.png", dpi=150, bbox_inches="tight")
    print("Saved plots/scaling_summary.png")

    # Figure 2: Weight recovery vs size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Weight Recovery vs Graph Size", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(n_vars, mean_wrmse, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Weight RMSE")
    ax.set_title("(a) Weight Estimation Error")
    ax.set_xticks(n_vars)

    ax = axes[1]
    ax.plot(n_vars, mean_wcov, "s-", color="#2ecc71", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("95% CI Coverage")
    ax.set_title("(b) Weight Credible Interval Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(n_vars)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="Nominal 95%")
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/scaling_weights.png", dpi=150, bbox_inches="tight")
    print("Saved plots/scaling_weights.png")

    # Figure 3: Posterior evolution for each size (example runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 5))
    if len(sizes) == 1:
        axes = [axes]
    fig.suptitle("Posterior Evolution by Graph Size (Example Runs)", fontsize=14, fontweight="bold")

    for idx, key in enumerate(sizes):
        ax = axes[idx]
        example = results[key].get("example_run", {})
        iters_data = example.get("iterations", [])
        if not iters_data:
            continue

        n_cands = results[key]["n_candidates"]
        # Extract MAP probability per iteration
        map_probs = [it["map_prob"] for it in iters_data]
        iter_nums = [it["iteration"] for it in iters_data]

        ax.plot(iter_nums, map_probs, "o-", linewidth=2, markersize=6, color="#3498db")
        ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("P(MAP)")
        ax.set_title(f"{results[key]['n_variables']} vars, {results[key]['n_edges']} edges")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/scaling_posteriors.png", dpi=150, bbox_inches="tight")
    print("Saved plots/scaling_posteriors.png")

    plt.close("all")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_scaling <results_json_path>")
        sys.exit(1)
    plot_scaling_results(sys.argv[1])
