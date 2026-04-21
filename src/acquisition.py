"""
Acquisition functions for intervention selection: Expected Information Gain (EIG).

EIG evaluates how much each candidate intervention would reduce entropy over
the graph belief. An exploration bonus ensures diverse variable coverage,
preventing the agent from repeatedly targeting the same variable when
marginal returns diminish.
"""

import numpy as np
from typing import Optional
from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief


def entropy(belief: np.ndarray) -> float:
    b = belief[belief > 0]
    return -np.sum(b * np.log(b))


def expected_information_gain(scm: LinearGaussianSCM, belief: GraphBelief,
                               intervention_value: float = 2.0,
                               n_simulations: int = 500,
                               n_samples_per_sim: int = 50,
                               seed: Optional[int] = None,
                               intervention_counts: dict = None,
                               exploration_weight: float = 0.1) -> dict:
    """Compute EIG for each possible intervention target with exploration bonus.

    For each candidate target:
    1. For each simulation: simulate interventional data from ground-truth SCM,
       compute marginal likelihood under each candidate graph, form simulated
       posterior, compute entropy.
    2. EIG = current_entropy - expected_posterior_entropy
    3. Add exploration bonus: sqrt(2 * ln(total_interventions) / count(target))
       This UCB1-style bonus encourages visiting under-explored variables.
    """
    rng = np.random.default_rng(seed)
    current_ent = entropy(belief.belief)
    eig = {}

    for target in scm.variables:
        post_ent_sum = 0.0
        for _ in range(n_simulations):
            s = rng.integers(0, 2**31)
            sim_data = scm.sample_interventional(target, intervention_value,
                                                  n_samples_per_sim, seed=s)

            log_liks = np.zeros(belief.K)
            for k in range(belief.K):
                log_liks[k] = belief.compute_log_predictive_likelihood(
                    k, sim_data, target)

            # Simulated posterior
            log_p = log_liks + np.log(belief.belief + 1e-300)
            log_p -= log_p.max()
            p = np.exp(log_p)
            p /= p.sum()
            post_ent_sum += entropy(p)

        eig[target] = current_ent - post_ent_sum / n_simulations

    # Add exploration bonus if intervention history is provided
    if intervention_counts is not None:
        total = sum(intervention_counts.values()) + 1  # +1 to avoid log(0)
        for target in scm.variables:
            count = intervention_counts.get(target, 0)
            if count == 0:
                # Unvisited variable gets a strong bonus
                bonus = exploration_weight * np.sqrt(2 * np.log(total + 1))
            else:
                # UCB1-style: sqrt(2 * ln(total) / count)
                bonus = exploration_weight * np.sqrt(2 * np.log(total) / count)
            eig[target] += bonus

    return eig


def select_intervention(scm: LinearGaussianSCM, belief: GraphBelief,
                        intervention_value: float = 2.0,
                        n_simulations: int = 500,
                        n_samples_per_sim: int = 50,
                        seed: Optional[int] = None,
                        intervention_counts: dict = None,
                        exploration_weight: float = 0.1) -> tuple:
    """Select intervention target maximising EIG. Returns (target, scores)."""
    scores = expected_information_gain(scm, belief, intervention_value,
                                       n_simulations, n_samples_per_sim, seed,
                                       intervention_counts, exploration_weight)
    best = max(scores, key=scores.get)
    return best, scores


def random_intervention(scm: LinearGaussianSCM, seed: Optional[int] = None) -> str:
    rng = np.random.default_rng(seed)
    return rng.choice(scm.variables)


if __name__ == "__main__":
    scm = LinearGaussianSCM()
    belief = GraphBelief(tau=3.0, sigma_w2=0.5, sigma_eps2=0.3)
    print(f"Current entropy: {entropy(belief.belief):.4f}")
    best, scores = select_intervention(scm, belief, n_simulations=100, seed=42)
    print("EIG scores:")
    for v, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  do({v}): {s:.4f}")
    print(f"Best: do({best})")
