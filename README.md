# CBO Scaling Experiment

Tests CBO graph recovery across progressively larger KEGG-grounded graphs.

## Design

The number of candidate graphs (K) scales with graph complexity via
`K = min(2 * n_edges, 25)`, ensuring that larger graphs face a proportionally
harder discrimination problem. Each candidate differs from the ground truth by
exactly one structural modification (edge removal, reversal, or addition).

## Graph Sizes

| Size  | Variables | Edges | K (candidates) | New Variables                                    |
|-------|-----------|-------|-----------------|--------------------------------------------------|
| 5var  | 5         | 5     | 10              | A (Glycolysis), B (TCA), C (OXPHOS), D (HIF-1), E (PI3K-Akt) |
| 7var  | 7         | 8     | 16              | + F (AMPK), G (mTOR)                             |
| 9var  | 9         | 11    | 22              | + H (PPP), I (FAO)                               |
| 10var | 10        | 13    | 25              | + J (p53)                                        |

## Methods Compared

| Method          | Description                                        |
|-----------------|----------------------------------------------------|
| EIG + LLM Prior | Full CBRA: EIG-based intervention + LLM prior (τ=3)|
| Uniform + EIG   | EIG-based intervention, uniform graph prior (τ=0)  |
| Random + LLM    | Random intervention selection, LLM prior (τ=3)     |

## Configuration

All hyperparameters are in `configs/experiment_config.json`:
- Convergence threshold: 0.95 (matching core experiment)
- EIG simulations: 200
- Repeats per size: 30
- Max iterations: 10

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data files
python -m src.generate_graphs

# Run scaling experiment (all sizes x all methods x 30 seeds)
python -m src.run_scaling

# Plot results
python -m src.plot_scaling logs/scaling_results_<timestamp>.json
```

## Architecture

* `configs/experiment_config.json` — All hyperparameters
* `src/graph_definitions.py` — KEGG-grounded graph definitions for all sizes
* `src/generate_graphs.py` — Generates ground truth + candidate graph JSON files (K scales with n_edges)
* `src/run_scaling.py` — Main experiment: sweeps across sizes with multiple seeds and baselines
* `src/plot_scaling.py` — Visualisation of scaling results with baseline comparisons
* Core modules (`scm.py`, `graph_belief.py`, `acquisition.py`, `metrics.py`) are identical to the base CBO experiment
