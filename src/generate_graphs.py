"""
Generate ground-truth DAG and candidate graph JSON files for a given graph size.

Candidate generation strategy:
- G1: correct graph (ground truth)
- G2..G_{K}: each differs from G1 by one structural modification:
    - remove one edge
    - reverse one edge
    - add one spurious edge

K scales with graph complexity: K = min(2 * n_edges, 25), ensuring that
larger graphs face a proportionally harder discrimination problem.
"""

import json
import numpy as np
from pathlib import Path
from src.graph_definitions import get_graph, list_sizes


def build_adjacency(variables: list, edges: list) -> np.ndarray:
    """Build adjacency matrix from edge list. A[i,j]=1 means i->j."""
    var_to_idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    adj = np.zeros((n, n), dtype=int)
    for e in edges:
        i = var_to_idx[e["source"]]
        j = var_to_idx[e["target"]]
        adj[i, j] = 1
    return adj


def has_cycle(adj: np.ndarray) -> bool:
    """Check if adjacency matrix contains a cycle (DFS-based)."""
    n = adj.shape[0]
    WHITE, GREY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(u):
        color[u] = GREY
        for v in range(n):
            if adj[u, v]:
                if color[v] == GREY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(n))


def edges_to_list(variables: list, edges: list) -> list:
    """Convert edge dicts to (source, target) tuples."""
    return [(e["source"], e["target"]) for e in edges]


def generate_candidates(graph_def: dict, max_candidates: int = None) -> list:
    """Generate candidate graphs: G1 = truth, G2+ = structural modifications.

    If max_candidates is None, K is set to min(2 * n_edges, 25) to scale
    difficulty with graph size.

    Modifications:
    1. Edge removals (one at a time)
    2. Edge reversals (one at a time, if acyclic)
    3. Edge additions (one spurious edge, if acyclic)
    """
    variables = list(graph_def["variables"].keys())
    edges = graph_def["edges"]
    n = len(variables)
    n_edges = len(edges)

    if max_candidates is None:
        max_candidates = min(2 * n_edges, 25)

    # G1: ground truth
    adj_true = build_adjacency(variables, edges)
    candidates = [{
        "id": "G1",
        "confidence": 0.70,
        "description": "Ground truth (KEGG baseline)",
        "mod_edge_list": edges_to_list(variables, edges),
        "adjacency": adj_true.tolist(),
    }]

    modifications = []

    # Edge removals
    for idx, e in enumerate(edges):
        mod_edges = [ee for ii, ee in enumerate(edges) if ii != idx]
        adj_mod = build_adjacency(variables, mod_edges)
        desc = f"Remove {e['source']}→{e['target']}"
        modifications.append((desc, adj_mod, mod_edges))

    # Edge reversals
    for idx, e in enumerate(edges):
        rev_edge = {"source": e["target"], "target": e["source"],
                    "weight": e["weight"], "sign": e["sign"], "evidence": "reversed"}
        mod_edges = [ee for ii, ee in enumerate(edges) if ii != idx] + [rev_edge]
        adj_mod = build_adjacency(variables, mod_edges)
        if not has_cycle(adj_mod):
            desc = f"Reverse {e['source']}→{e['target']}"
            modifications.append((desc, adj_mod, mod_edges))

    # Edge additions (one spurious edge)
    for i, vi in enumerate(variables):
        for j, vj in enumerate(variables):
            if i == j:
                continue
            if adj_true[i, j] or adj_true[j, i]:
                continue
            adj_test = adj_true.copy()
            adj_test[i, j] = 1
            if not has_cycle(adj_test):
                desc = f"Add {vi}→{vj}"
                spur_edge = {"source": vi, "target": vj,
                             "weight": 0.5, "sign": "excitatory", "evidence": "spurious"}
                mod_edges = list(edges) + [spur_edge]
                modifications.append((desc, adj_test, mod_edges))

    # Select diverse modifications with balanced type representation
    rng = np.random.default_rng(42)

    removals = [m for m in modifications if m[0].startswith("Remove")]
    reversals = [m for m in modifications if m[0].startswith("Reverse")]
    additions = [m for m in modifications if m[0].startswith("Add")]

    # Shuffle each category
    rng.shuffle(removals)
    rng.shuffle(reversals)
    rng.shuffle(additions)

    # Fill slots with balanced representation: ~40% removals, ~20% reversals, ~40% additions
    n_slots = max_candidates - 1  # G1 already included
    n_removals = min(len(removals), max(1, int(0.4 * n_slots)))
    n_reversals = min(len(reversals), max(1, int(0.2 * n_slots)))
    n_additions = min(len(additions), n_slots - n_removals - n_reversals)

    # If we have leftover slots, fill from additions then removals
    total = n_removals + n_reversals + n_additions
    if total < n_slots:
        extra_add = min(len(additions) - n_additions, n_slots - total)
        n_additions += extra_add
        total = n_removals + n_reversals + n_additions
    if total < n_slots:
        extra_rem = min(len(removals) - n_removals, n_slots - total)
        n_removals += extra_rem

    selected = []
    selected.extend(removals[:n_removals])
    selected.extend(reversals[:n_reversals])
    selected.extend(additions[:n_additions])

    # Shuffle the selected set so types are interleaved
    rng.shuffle(selected)

    # Trim to exact count
    selected = selected[:n_slots]

    # Assign decreasing confidence scores from 0.65 down to 0.30
    for i, (desc, adj, mod_edges) in enumerate(selected):
        adj_arr = np.array(adj) if not isinstance(adj, np.ndarray) else adj
        conf = max(0.30, 0.65 - i * (0.35 / max(1, len(selected) - 1)))
        candidates.append({
            "id": f"G{i + 2}",
            "confidence": round(conf, 3),
            "description": desc,
            "mod_edge_list": edges_to_list(variables, mod_edges),
            "adjacency": adj_arr.tolist(),
        })

    return candidates


def generate_json_files(size_key: str, output_dir: str = "data",
                        max_candidates: int = None) -> tuple:
    """Generate ground_truth_dag.json and candidate_graphs.json for a graph size."""
    graph_def = get_graph(size_key)
    variables = graph_def["variables"]
    var_list = list(variables.keys())
    topo = graph_def["topological_order"]
    edges = graph_def["edges"]

    # Ground truth DAG JSON
    adj = build_adjacency(var_list, edges)

    true_weights = {}
    for e in edges:
        key = f"w_{e['source']}{e['target']}"
        true_weights[key] = e["weight"]

    gt_json = {
        "description": graph_def["description"],
        "variables": variables,
        "topological_order": topo,
        "edges": edges,
        "adjacency_matrix": {
            "order": var_list,
            "matrix": adj.tolist(),
        },
        "scm_parameters": {
            "noise_variance": 0.3,
            "weights": true_weights,
        },
    }

    out = Path(output_dir) / size_key
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "ground_truth_dag.json", "w") as f:
        json.dump(gt_json, f, indent=2)

    # Candidate graphs JSON
    candidates = generate_candidates(graph_def, max_candidates=max_candidates)
    cand_json = {
        "description": f"Candidate graphs for {size_key}",
        "variable_order": var_list,
        "candidates": [],
    }
    for c in candidates:
        cand_entry = {
            "id": c["id"],
            "confidence": c["confidence"],
            "description": c["description"],
            "edges": [],
            "adjacency_matrix": c["adjacency"],
            "rationale": c["description"],
        }
        cand_json["candidates"].append(cand_entry)

    with open(out / "candidate_graphs.json", "w") as f:
        json.dump(cand_json, f, indent=2)

    print(f"Generated {size_key}: {len(var_list)} vars, {len(edges)} edges, "
          f"{len(candidates)} candidates (K={len(candidates)}) -> {out}/")

    return gt_json, cand_json


if __name__ == "__main__":
    for key in list_sizes():
        generate_json_files(key)
