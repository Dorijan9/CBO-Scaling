"""
Plot scaling experiment results: how CBO performance changes with graph size,
including baseline comparisons.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_scaling_results(results_path: str):
    """Generate scaling experiment plots from saved results."""
    with open(results_path) as f:
        all_data = json.load(f)

    # Extract config and size keys
    config = all_data.get("config", {})
    size_keys = [k for k in all_data.keys() if k not in ("timestamp", "config")]
    size_keys = sorted(size_keys, key=lambda k: all_data[k]["eig_llm"]["n_variables"])

    methods = [
        ("eig_llm", "EIG + LLM Prior", "#27AE60"),
        ("uniform_eig", "EIG + Uniform Prior", "#3498DB"),
        ("random_llm", "Random + LLM Prior", "#E74C3C"),
    ]

    n_vars = [all_data[k]["eig_llm"]["n_variables"] for k in size_keys]
    n_edges = [all_data[k]["eig_llm"]["n_edges"] for k in size_keys]
    n_cands = [all_data[k]["eig_llm"]["n_candidates"] for k in size_keys]
    x_labels = [f"{nv}v/{ne}e\nK={nc}" for nv, ne, nc in zip(n_vars, n_edges, n_cands)]

    Path("plots").mkdir(exist_ok=True)

    # =========================================================================
    # Figure 1: Main scaling summary (2x2) with all methods
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CBO Recovery Performance vs Graph Size", fontsize=16, fontweight="bold")

    x = np.arange(len(size_keys))
    width = 0.25

    # (a) Correct recovery rate
    ax = axes[0, 0]
    for i, (mkey, mlabel, mcolor) in enumerate(methods):
        vals = [all_data[k][mkey]["correct_rate"] for k in size_keys]
        ax.bar(x + i * width, vals, width, label=mlabel, color=mcolor,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) Recovery Accuracy")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    # (b) SHD
    ax = axes[0, 1]
    for i, (mkey, mlabel, mcolor) in enumerate(methods):
        means = [all_data[k][mkey]["mean_shd"] for k in size_keys]
        stds = [all_data[k][mkey]["std_shd"] for k in size_keys]
        ax.errorbar([v + i * 0.15 for v in n_vars], means, yerr=stds,
                    fmt="o-", color=mcolor, capsize=4, linewidth=2,
                    markersize=7, label=mlabel)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Structural Hamming Distance")
    ax.set_title("(b) Structural Error (SHD)")
    ax.set_xticks(n_vars)
    ax.legend(fontsize=8)

    # (c) F1 Score
    ax = axes[1, 0]
    for i, (mkey, mlabel, mcolor) in enumerate(methods):
        means = [all_data[k][mkey]["mean_f1"] for k in size_keys]
        stds = [all_data[k][mkey]["std_f1"] for k in size_keys]
        ax.errorbar([v + i * 0.15 for v in n_vars], means, yerr=stds,
                    fmt="s-", color=mcolor, capsize=4, linewidth=2,
                    markersize=7, label=mlabel)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Edge F1 Score")
    ax.set_title("(c) Edge Recovery (F1)")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(n_vars)
    ax.legend(fontsize=8)

    # (d) Iterations to convergence
    ax = axes[1, 1]
    for i, (mkey, mlabel, mcolor) in enumerate(methods):
        means = [all_data[k][mkey]["mean_iterations"] for k in size_keys]
        stds = [all_data[k][mkey]["std_iterations"] for k in size_keys]
        ax.bar(x + i * width, means, width, yerr=stds, label=mlabel,
               color=mcolor, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Mean Iterations")
    ax.set_title("(d) Sample Efficiency")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/scaling_summary.png", dpi=150, bbox_inches="tight")
    print("Saved plots/scaling_summary.png")

    # =========================================================================
    # Figure 2: Weight recovery vs size
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Weight Recovery vs Graph Size", fontsize=14, fontweight="bold")

    ax = axes[0]
    for mkey, mlabel, mcolor in methods:
        means = [all_data[k][mkey]["mean_weight_rmse"] for k in size_keys]
        stds = [all_data[k][mkey]["std_weight_rmse"] for k in size_keys]
        ax.errorbar(n_vars, means, yerr=stds, fmt="o-", color=mcolor,
                    capsize=4, linewidth=2, markersize=7, label=mlabel)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Weight RMSE")
    ax.set_title("(a) Weight Estimation Error")
    ax.set_xticks(n_vars)
    ax.legend(fontsize=8)

    ax = axes[1]
    for mkey, mlabel, mcolor in methods:
        means = [all_data[k][mkey]["mean_weight_coverage"] for k in size_keys]
        stds = [all_data[k][mkey]["std_weight_coverage"] for k in size_keys]
        ax.errorbar(n_vars, means, yerr=stds, fmt="s-", color=mcolor,
                    capsize=4, linewidth=2, markersize=7, label=mlabel)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("95% CI Coverage")
    ax.set_title("(b) Weight Credible Interval Coverage")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(n_vars)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.4, label="Nominal 95%")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/scaling_weights.png", dpi=150, bbox_inches="tight")
    print("Saved plots/scaling_weights.png")

    # =========================================================================
    # Figure 3: Posterior evolution for EIG+LLM (example runs)
    # =========================================================================
    fig, axes = plt.subplots(1, len(size_keys), figsize=(4 * len(size_keys), 5))
    if len(size_keys) == 1:
        axes = [axes]
    fig.suptitle("MAP Posterior Evolution by Graph Size (EIG+LLM, example run)",
                 fontsize=14, fontweight="bold")

    for idx, key in enumerate(size_keys):
        ax = axes[idx]
        ind_runs = all_data[key].get("individual_runs", {}).get("eig_llm", [])
        if not ind_runs:
            continue
        example = ind_runs[0]
        iters_data = example.get("iterations", [])
        if not iters_data:
            continue

        map_probs = [it["map_prob"] for it in iters_data]
        iter_nums = [it["iteration"] for it in iters_data]

        ax.plot(iter_nums, map_probs, "o-", linewidth=2, markersize=6, color="#27AE60")
        threshold = config.get("cbo", {}).get("convergence_threshold", 0.95)
        ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5,
                    label=f"Threshold ({threshold})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("P(MAP)")
        agg = all_data[key]["eig_llm"]
        ax.set_title(f"{agg['n_variables']} vars, {agg['n_edges']} edges, K={agg['n_candidates']}")
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
