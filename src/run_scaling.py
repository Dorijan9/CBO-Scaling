"""
CBO Scaling Experiment: Run validation recovery across graph sizes.

For each graph size (5, 7, 9, 10 variables):
1. Generate ground truth DAG and candidate graphs
2. Run the CBO loop
3. Record performance metrics
4. Compare across sizes
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief
from src.acquisition import select_intervention, entropy
from src.metrics import evaluate_graph, evaluate_weights
from src.generate_graphs import generate_json_files
from src.graph_definitions import list_sizes, get_graph


def run_cbo_for_size(size_key: str, config: dict, seed: int = 42,
                     verbose: bool = True) -> dict:
    """Run the CBO loop for a given graph size."""
    # Generate data files
    data_dir = "data"
    generate_json_files(size_key, data_dir)

    graph_def = get_graph(size_key)
    true_weights = {}
    for e in graph_def["edges"]:
        true_weights[f"w_{e['source']}{e['target']}"] = e["weight"]

    # Paths to generated files
    size_dir = f"{data_dir}/{size_key}"
    gt_path = f"{size_dir}/ground_truth_dag.json"
    cand_path = f"{size_dir}/candidate_graphs.json"

    # Init SCM and belief
    scm = LinearGaussianSCM(dag_path=gt_path)
    belief = GraphBelief(
        candidates_path=cand_path,
        tau=config["graph_prior"]["temperature_tau"],
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

    # Observational data
    obs_data = scm.sample_observational(scm_cfg["n_observational_samples"], seed=seed)
    all_intv_data = []

    n_vars = len(scm.variables)
    n_edges = len(graph_def["edges"])
    n_candidates = belief.K

    if verbose:
        print(f"\n{'='*70}")
        print(f"SCALING EXPERIMENT: {size_key} ({n_vars} vars, {n_edges} edges, {n_candidates} candidates)")
        print(f"{'='*70}")
        print(f"Initial belief: {belief.belief.round(4)}")
        print(f"Initial entropy: {entropy(belief.belief):.4f}")

    iterations = []
    converged_at = None

    for t in range(1, max_iter + 1):
        iter_seed = rng.integers(0, 2**31)

        # Select intervention via EIG
        target, eig_scores = select_intervention(
            scm, belief, intv_val, n_sim, n_samples, seed=iter_seed
        )

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
            print(f"  Iter {t}: do({target}) → MAP={map_graph['id']} "
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

    result = {
        "size_key": size_key,
        "n_variables": n_vars,
        "n_edges": n_edges,
        "n_candidates": n_candidates,
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

    if verbose:
        print(f"\n  RESULT: {'CORRECT' if correct else 'INCORRECT'} "
              f"(MAP={final_map['id']}, P={belief.belief[final_map_idx]:.4f}, "
              f"SHD={final_eval['shd']}, F1={final_eval['f1']:.3f})")

    return result


def run_scaling_experiment(n_repeats: int = 10, seed: int = 42, verbose: bool = True):
    """Run the full scaling experiment across all graph sizes with multiple seeds."""
    config = {
        "scm": {
            "observation_noise_variance": 0.3,
            "n_observational_samples": 200,
            "n_interventional_samples_per_iter": 10,
            "intervention_value": 2.0,
        },
        "weight_prior": {"variance": 0.5},
        "graph_prior": {"temperature_tau": 3.0},
        "cbo": {
            "max_iterations": 8,  # More iterations for larger graphs
            "convergence_threshold": 0.90,
            "n_eig_simulations": 100,
        },
    }

    sizes = list_sizes()  # ["5var", "7var", "9var", "10var"]
    all_results = {}

    for size_key in sizes:
        size_results = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000
            result = run_cbo_for_size(
                size_key, config, seed=rep_seed,
                verbose=(verbose and rep == 0)  # Only verbose for first repeat
            )
            size_results.append(result)

            if not verbose and rep == 0:
                n = result["n_variables"]
                print(f"{size_key} ({n} vars): running {n_repeats} repeats...")

        # Aggregate
        agg = {
            "size_key": size_key,
            "n_variables": size_results[0]["n_variables"],
            "n_edges": size_results[0]["n_edges"],
            "n_candidates": size_results[0]["n_candidates"],
            "n_repeats": n_repeats,
            "correct_rate": np.mean([r["correct_recovery"] for r in size_results]),
            "mean_shd": float(np.mean([r["final_shd"] for r in size_results])),
            "std_shd": float(np.std([r["final_shd"] for r in size_results])),
            "mean_f1": float(np.mean([r["final_f1"] for r in size_results])),
            "std_f1": float(np.std([r["final_f1"] for r in size_results])),
            "mean_map_prob": float(np.mean([r["final_map_prob"] for r in size_results])),
            "mean_iterations": float(np.mean([r["total_iterations"] for r in size_results])),
            "mean_weight_rmse": float(np.mean([r["final_weight_rmse"] for r in size_results])),
            "mean_weight_coverage": float(np.mean([r["final_weight_coverage"] for r in size_results])),
            "mean_entropy_reduction": float(np.mean([r["entropy_reduction"] for r in size_results])),
            "convergence_rate": np.mean([r["converged"] for r in size_results]),
            "individual_runs": size_results,
        }
        all_results[size_key] = agg

        print(f"\n  {size_key} SUMMARY: correct={agg['correct_rate']:.0%} "
              f"SHD={agg['mean_shd']:.2f}±{agg['std_shd']:.2f} "
              f"F1={agg['mean_f1']:.3f}±{agg['std_f1']:.3f} "
              f"iters={agg['mean_iterations']:.1f} "
              f"wRMSE={agg['mean_weight_rmse']:.4f} "
              f"wCov={agg['mean_weight_coverage']:.0%}")

    # Save
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"logs/scaling_results_{timestamp}.json"

    # Strip individual runs for the saved summary (they're large)
    save_results = {}
    for key, agg in all_results.items():
        save_copy = {k: v for k, v in agg.items() if k != "individual_runs"}
        # Keep first run's iteration details for plotting
        save_copy["example_run"] = agg["individual_runs"][0]
        save_results[key] = save_copy

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print final comparison table
    print(f"\n{'='*80}")
    print(f"SCALING EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Size':<8} {'Vars':>4} {'Edges':>5} {'K':>3} {'Correct':>8} "
          f"{'SHD':>8} {'F1':>8} {'Iters':>6} {'wRMSE':>8} {'wCov':>6}")
    print("-" * 80)
    for key in sizes:
        a = all_results[key]
        print(f"{key:<8} {a['n_variables']:>4} {a['n_edges']:>5} {a['n_candidates']:>3} "
              f"{a['correct_rate']:>7.0%} "
              f"{a['mean_shd']:>5.2f}±{a['std_shd']:.1f} "
              f"{a['mean_f1']:>5.3f}±{a['std_f1']:.2f} "
              f"{a['mean_iterations']:>5.1f} "
              f"{a['mean_weight_rmse']:>8.4f} "
              f"{a['mean_weight_coverage']:>5.0%}")

    return all_results, results_path


def main():
    results, path = run_scaling_experiment(n_repeats=3, seed=42, verbose=True)


if __name__ == "__main__":
    main()
