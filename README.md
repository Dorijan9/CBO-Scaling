# CBO Scaling Experiment

Tests CBO validation recovery across progressively larger KEGG-grounded graphs.

## Graph Sizes

| Size | Variables | Edges | New Variables |
|------|-----------|-------|---------------|
| 5var | 5 | 5 | A (Glycolysis), B (TCA), C (OXPHOS), D (HIF-1), E (PI3K-Akt) |
| 7var | 7 | 8 | + F (AMPK), G (mTOR) |
| 9var | 9 | 11 | + H (PPP), I (FAO) |
| 10var | 10 | 13 | + J (p53) |

## Running

```bash
# Generate data files
python -m src.generate_graphs

# Run scaling experiment
python -m src.run_scaling

# Plot results
python -m src.plot_scaling logs/scaling_results_<timestamp>.json
```

## Architecture

- `src/graph_definitions.py` — KEGG-grounded graph definitions for all sizes
- `src/generate_graphs.py` — Generates ground truth + candidate graph JSON files
- `src/run_scaling.py` — Main experiment: sweeps across sizes with multiple seeds
- `src/plot_scaling.py` — Visualisation of scaling results
- Core modules (`scm.py`, `graph_belief.py`, `acquisition.py`, `metrics.py`) are unchanged from the base CBO experiment
