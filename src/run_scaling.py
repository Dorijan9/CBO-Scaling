"""
CBO Scaling Experiment: Run validation recovery across graph sizes.

For each graph size (5, 7, 9, 10 variables):
1. Generate ground truth DAG and candidate graphs (K scales with n_edges)
2. Run the CBO loop with EIG-based intervention selection
3. Run baselines (uniform prior CBO, random intervention)
4. Record performance metrics across multiple seeds
5. Compare across sizes

All hyperparameters are loaded from configs/experiment_config.json.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief
from src.acquisition import select_intervention, random_intervention, entropy
from src.metrics import evaluate_graph, evaluate_weights
from src.generate_graphs import generate_json_files
from src.graph_definitions import list_sizes, get_graph


def run_cbo_for_size(size_key: str, config: dict, seed: int = 42,
                     use_eig: bool = True, use_llm_prior: bool = True,
                     verbose: bool = False) -> dict:
    """Run the CBO loop for a given graph size.

    Args:
        use_eig: If False, use random intervention selection.
        use_llm_prior: If False, use uniform prior (tau=0).
    """
    data_dir = "data"
    generate_json_files(size_key, data_dir)

    graph_def = get_graph(size_key)
    true_weights = {}
    for e in graph_def["edges"]:
        true_weights[f"w_{e['source']}{e['target']}"] = e["weight"]

    size_dir = f"{data_dir}/{size_key}"
    gt_path = f"{size_dir}/ground_truth_dag.json"
    cand_path = f"{size_dir}/candidate_graphs.json"

    scm = LinearGaussianSCM(dag_path=gt_path)

    tau = config["graph_prior"]["temperature_tau"] if use_llm_prior else 0.0
    belief = GraphBelief(
        candidates_path=cand_path,
        tau=tau,
        sigma_w2=config["weight_prior"]["variance"],
        sigma_eps2=config["scm"]["observation_noise_variance"],
    )

    rng = np.random.default_rng(seed)
    cbo_cfg = config["cbo"]
    scm_cfg = config["scm"]

    max_iter = cbo_cfg["max_iterations"]
    threshold = cbo_cfg["convergence_threshold"]
    n_sim = cbo_cfg["n_eig_simulations"]
    n_samples = scm_cfg["n_interventional_samples_per_iter"]
    intv_val = scm_cfg["intervention_value"]

    obs_data = scm.sample_observational(scm_cfg["n_observational_samples"], seed=seed)
    all_intv_data = []

    n_vars = len(scm.variables)
    n_edges = len(graph_def["edges"])
    n_candidates = belief.K

    iterations = []
    converged_at = None
    intervention_counts = {}

    if verbose:
        method = "EIG" if use_eig else "Random"
        prior = "LLM" if use_llm_prior else "Uniform"
        print(f"\n{'='*70}")
        print(f"SCALING: {size_key} ({n_vars} vars, {n_edges} edges, K={n_candidates}) "
              f"[{method} + {prior} prior]")
        print(f"{'='*70}")

    for t in range(1, max_iter + 1):
        iter_seed = rng.integers(0, 2**31)

        # Select intervention
        if use_eig:
            target, eig_scores = select_intervention(
                scm, belief, intv_val, n_sim, n_samples, seed=iter_seed,
                intervention_counts=intervention_counts,
            )
        else:
            target = random_intervention(scm, seed=iter_seed)
            eig_scores = {}

        # Track intervention history
        intervention_counts[target] = intervention_counts.get(target, 0) + 1

        # Perform intervention
        intv_data = scm.sample_interventional(
            target, intv_val, n_samples, seed=iter_seed + 1
        )
        all_intv_data.append(intv_data)
        combined_data = np.vstack([obs_data] + all_intv_data)

        # Update
        old_belief = belief.belief.copy()
        new_belief = belief.update(intv_data, target, combined_data)

        # Evaluate
        map_idx = belief.map_estimate()
        map_graph = belief.candidates[map_idx]
        gt_adj = np.array(scm.adj)
        map_adj = np.array(map_graph["adjacency"])
        eval_struct = evaluate_graph(gt_adj, map_adj)

        wp_summary = belief.get_weight_posterior_summary(map_idx)
        eval_wt = evaluate_weights(wp_summary, true_weights)

        iter_result = {
            "iteration": t,
            "target": target,
            "map_graph": map_graph["id"],
            "map_prob": float(new_belief[map_idx]),
            "entropy": float(entropy(new_belief)),
            "shd": eval_struct["shd"],
            "f1": eval_struct["f1"],
            "weight_rmse": eval_wt["weight_rmse"],
            "weight_coverage": eval_wt["weight_coverage"],
        }
        iterations.append(iter_result)

        if verbose:
            print(f"  Iter {t}: do({target}) -> MAP={map_graph['id']} "
                  f"P={new_belief[map_idx]:.4f} SHD={eval_struct['shd']} "
                  f"F1={eval_struct['f1']:.3f} wRMSE={eval_wt['weight_rmse']:.4f}")

        if belief.has_converged(threshold):
            converged_at = t
            if verbose:
                print(f"  *** Converged at iteration {t} ***")
            break

    # Final evaluation
    final_map_idx = belief.map_estimate()
    final_map = belief.candidates[final_map_idx]
    final_eval = evaluate_graph(gt_adj, np.array(final_map["adjacency"]))
    final_wp = belief.get_weight_posterior_summary(final_map_idx)
    final_wt = evaluate_weights(final_wp, true_weights)

    correct = final_map["id"] == "G1"

    return {
        "size_key": size_key,
        "n_variables": n_vars,
        "n_edges": n_edges,
        "n_candidates": n_candidates,
        "seed": seed,
        "iterations": iterations,
        "total_iterations": len(iterations),
        "converged": converged_at is not None,
        "converged_at": converged_at,
        "correct_recovery": correct,
        "final_map_graph": final_map["id"],
        "final_map_prob": float(belief.belief[final_map_idx]),
        "final_entropy": float(entropy(belief.belief)),
        "final_shd": final_eval["shd"],
        "final_f1": final_eval["f1"],
        "final_precision": final_eval["precision"],
        "final_recall": final_eval["recall"],
        "final_weight_rmse": final_wt["weight_rmse"],
        "final_weight_coverage": final_wt["weight_coverage"],
        "entropy_reduction": float(entropy(belief.prior) - entropy(belief.belief)),
    }


def aggregate_runs(size_results: list) -> dict:
    """Compute summary statistics from a list of individual run results."""
    n = len(size_results)
    return {
        "size_key": size_results[0]["size_key"],
        "n_variables": size_results[0]["n_variables"],
        "n_edges": size_results[0]["n_edges"],
        "n_candidates": size_results[0]["n_candidates"],
        "n_repeats": n,
        "correct_rate": float(np.mean([r["correct_recovery"] for r in size_results])),
        "mean_shd": float(np.mean([r["final_shd"] for r in size_results])),
        "std_shd": float(np.std([r["final_shd"] for r in size_results])),
        "mean_f1": float(np.mean([r["final_f1"] for r in size_results])),
        "std_f1": float(np.std([r["final_f1"] for r in size_results])),
        "mean_map_prob": float(np.mean([r["final_map_prob"] for r in size_results])),
        "std_map_prob": float(np.std([r["final_map_prob"] for r in size_results])),
        "mean_iterations": float(np.mean([r["total_iterations"] for r in size_results])),
        "std_iterations": float(np.std([r["total_iterations"] for r in size_results])),
        "mean_weight_rmse": float(np.mean([r["final_weight_rmse"] for r in size_results])),
        "std_weight_rmse": float(np.std([r["final_weight_rmse"] for r in size_results])),
        "mean_weight_coverage": float(np.mean([r["final_weight_coverage"] for r in size_results])),
        "std_weight_coverage": float(np.std([r["final_weight_coverage"] for r in size_results])),
        "mean_entropy_reduction": float(np.mean([r["entropy_reduction"] for r in size_results])),
        "convergence_rate": float(np.mean([r["converged"] for r in size_results])),
    }


def run_scaling_experiment(config: dict, verbose: bool = True):
    """Run the full scaling experiment across all graph sizes with multiple seeds."""
    seed = config["random_seed"]
    n_repeats = config["scaling"]["n_repeats"]
    sizes = config["scaling"]["sizes"]

    all_results = {}

    for size_key in sizes:
        print(f"\n{'#'*70}")
        print(f"# {size_key}: Running {n_repeats} repeats (EIG + LLM prior)")
        print(f"{'#'*70}")

        # --- Main method: EIG + LLM prior ---
        eig_results = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000
            result = run_cbo_for_size(
                size_key, config, seed=rep_seed,
                use_eig=True, use_llm_prior=True,
                verbose=(verbose and rep == 0),
            )
            eig_results.append(result)
            if rep % 10 == 0 and rep > 0:
                print(f"  ... completed {rep}/{n_repeats} repeats")

        eig_agg = aggregate_runs(eig_results)

        # --- Baseline 1: Uniform prior + EIG ---
        print(f"  Running baseline: Uniform prior + EIG ({n_repeats} repeats)...")
        uniform_results = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000
            result = run_cbo_for_size(
                size_key, config, seed=rep_seed,
                use_eig=True, use_llm_prior=False,
                verbose=False,
            )
            uniform_results.append(result)
        uniform_agg = aggregate_runs(uniform_results)

        # --- Baseline 2: Random intervention + LLM prior ---
        print(f"  Running baseline: Random intervention + LLM prior ({n_repeats} repeats)...")
        random_results = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000
            result = run_cbo_for_size(
                size_key, config, seed=rep_seed,
                use_eig=False, use_llm_prior=True,
                verbose=False,
            )
            random_results.append(result)
        random_agg = aggregate_runs(random_results)

        all_results[size_key] = {
            "eig_llm": eig_agg,
            "uniform_eig": uniform_agg,
            "random_llm": random_agg,
            "individual_runs": {
                "eig_llm": eig_results,
                "uniform_eig": uniform_results,
                "random_llm": random_results,
            },
        }

        # Print summary for this size
        print(f"\n  {size_key} SUMMARY ({eig_agg['n_variables']} vars, "
              f"{eig_agg['n_edges']} edges, K={eig_agg['n_candidates']}):")
        for label, agg in [("EIG+LLM", eig_agg), ("Uniform+EIG", uniform_agg),
                           ("Random+LLM", random_agg)]:
            print(f"    {label:>12}: correct={agg['correct_rate']:.0%} "
                  f"SHD={agg['mean_shd']:.2f}+/-{agg['std_shd']:.2f} "
                  f"F1={agg['mean_f1']:.3f}+/-{agg['std_f1']:.3f} "
                  f"iters={agg['mean_iterations']:.1f}+/-{agg['std_iterations']:.1f} "
                  f"wRMSE={agg['mean_weight_rmse']:.4f} "
                  f"wCov={agg['mean_weight_coverage']:.0%}")

    # Save results
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"logs/scaling_results_{timestamp}.json"

    # Save full results including all individual runs
    save_results = {
        "timestamp": timestamp,
        "config": config,
    }
    for key, data in all_results.items():
        save_results[key] = {
            "eig_llm": data["eig_llm"],
            "uniform_eig": data["uniform_eig"],
            "random_llm": data["random_llm"],
            "individual_runs": data["individual_runs"],
        }

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")

    # Print final comparison table
    print(f"\n{'='*90}")
    print(f"SCALING EXPERIMENT SUMMARY (n_repeats={n_repeats}, "
          f"threshold={config['cbo']['convergence_threshold']}, "
          f"n_eig_sims={config['cbo']['n_eig_simulations']})")
    print(f"{'='*90}")
    print(f"{'Size':<8} {'Vars':>4} {'Edges':>5} {'K':>3} | "
          f"{'Method':<12} {'Correct':>8} {'SHD':>10} {'F1':>10} {'Iters':>8} {'wRMSE':>8}")
    print("-" * 90)
    for key in sizes:
        data = all_results[key]
        for i, (label, agg_key) in enumerate([("EIG+LLM", "eig_llm"),
                                               ("Uniform+EIG", "uniform_eig"),
                                               ("Random+LLM", "random_llm")]):
            agg = data[agg_key]
            prefix = f"{key:<8} {agg['n_variables']:>4} {agg['n_edges']:>5} {agg['n_candidates']:>3}" if i == 0 else " " * 22
            print(f"{prefix} | {label:<12} {agg['correct_rate']:>7.0%} "
                  f"{agg['mean_shd']:>5.2f}+/-{agg['std_shd']:<3.1f} "
                  f"{agg['mean_f1']:>5.3f}+/-{agg['std_f1']:<4.2f} "
                  f"{agg['mean_iterations']:>4.1f}+/-{agg['std_iterations']:<3.1f} "
                  f"{agg['mean_weight_rmse']:>8.4f}")
        print("-" * 90)

    return all_results, results_path


def main():
    with open("configs/experiment_config.json") as f:
        config = json.load(f)

    results, path = run_scaling_experiment(config, verbose=True)


if __name__ == "__main__":
    main()
