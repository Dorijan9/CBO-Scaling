"""
KEGG-grounded graph definitions for the scaling experiment.

Four progressively larger DAGs, each extending the previous by adding
biologically grounded variables with documented KEGG cross-references.

Base (5 vars):  E→D→A→B→C, D→C
7 vars:         + F (AMPK), G (mTOR)
9 vars:         + H (PPP), I (FAO)
10 vars:        + J (p53)
"""

GRAPHS = {}

# =============================================================================
# 5-variable base graph (same as original validation experiment)
# =============================================================================
GRAPHS["5var"] = {
    "description": "Base KEGG graph: PI3K-Akt → HIF-1 → Glycolysis → TCA → OXPHOS",
    "variables": {
        "A": {"name": "Glycolytic flux", "kegg_id": "hsa00010"},
        "B": {"name": "TCA cycle activity", "kegg_id": "hsa00020"},
        "C": {"name": "Oxidative phosphorylation", "kegg_id": "hsa00190"},
        "D": {"name": "HIF-1 signalling", "kegg_id": "hsa04066"},
        "E": {"name": "PI3K-Akt signalling", "kegg_id": "hsa04151"},
    },
    "topological_order": ["E", "D", "A", "B", "C"],
    "edges": [
        {"source": "E", "target": "D", "weight": 0.75, "sign": "excitatory",
         "evidence": "PI3K-Akt activates HIF-1α via mTOR translation"},
        {"source": "D", "target": "A", "weight": 0.80, "sign": "excitatory",
         "evidence": "HIF-1 upregulates glycolytic enzymes HK2, PFK, LDHA"},
        {"source": "D", "target": "C", "weight": -0.50, "sign": "inhibitory",
         "evidence": "HIF-1 suppresses OXPHOS via PDK1, BNIP3 mitophagy"},
        {"source": "A", "target": "B", "weight": 0.70, "sign": "excitatory",
         "evidence": "Glycolysis produces pyruvate/acetyl-CoA feeding TCA"},
        {"source": "B", "target": "C", "weight": 0.60, "sign": "excitatory",
         "evidence": "TCA generates NADH/FADH2 for OXPHOS electron transport"},
    ],
}

# =============================================================================
# 7-variable graph: add AMPK (F) and mTOR (G)
# =============================================================================
GRAPHS["7var"] = {
    "description": "Extended: + AMPK signalling (F), mTOR signalling (G)",
    "variables": {
        **GRAPHS["5var"]["variables"],
        "F": {"name": "AMPK signalling", "kegg_id": "hsa04152"},
        "G": {"name": "mTOR signalling", "kegg_id": "hsa04150"},
    },
    "topological_order": ["F", "E", "G", "D", "A", "B", "C"],
    "edges": [
        # Original 5-var edges (E→D replaced by E→G→D)
        {"source": "D", "target": "A", "weight": 0.80, "sign": "excitatory",
         "evidence": "HIF-1 upregulates glycolytic enzymes"},
        {"source": "D", "target": "C", "weight": -0.50, "sign": "inhibitory",
         "evidence": "HIF-1 suppresses OXPHOS via PDK1"},
        {"source": "A", "target": "B", "weight": 0.70, "sign": "excitatory",
         "evidence": "Glycolysis feeds pyruvate to TCA"},
        {"source": "B", "target": "C", "weight": 0.60, "sign": "excitatory",
         "evidence": "TCA NADH/FADH2 drives OXPHOS"},
        # New edges involving F (AMPK) and G (mTOR)
        {"source": "F", "target": "E", "weight": -0.45, "sign": "inhibitory",
         "evidence": "AMPK inhibits PI3K-Akt via TSC2 phosphorylation (KEGG hsa04152→hsa04151)"},
        {"source": "F", "target": "A", "weight": 0.50, "sign": "excitatory",
         "evidence": "AMPK activates glycolysis via PFK2 phosphorylation (KEGG hsa04152→hsa00010)"},
        {"source": "E", "target": "G", "weight": 0.70, "sign": "excitatory",
         "evidence": "PI3K-Akt activates mTORC1 via TSC2 inhibition (KEGG hsa04151→hsa04150)"},
        {"source": "G", "target": "D", "weight": 0.65, "sign": "excitatory",
         "evidence": "mTOR promotes HIF-1α translation (KEGG hsa04150→hsa04066)"},
    ],
}

# =============================================================================
# 9-variable graph: add PPP (H) and FAO (I)
# =============================================================================
GRAPHS["9var"] = {
    "description": "Extended: + Pentose phosphate pathway (H), Fatty acid oxidation (I)",
    "variables": {
        **GRAPHS["7var"]["variables"],
        "H": {"name": "Pentose phosphate pathway", "kegg_id": "hsa00030"},
        "I": {"name": "Fatty acid oxidation", "kegg_id": "hsa00071"},
    },
    "topological_order": ["F", "E", "G", "D", "I", "A", "H", "B", "C"],
    "edges": [
        # All 7-var edges
        *GRAPHS["7var"]["edges"],
        # New edges involving H (PPP) and I (FAO)
        {"source": "A", "target": "H", "weight": 0.55, "sign": "excitatory",
         "evidence": "G6P from glycolysis feeds PPP (KEGG hsa00010→hsa00030)"},
        {"source": "F", "target": "I", "weight": 0.60, "sign": "excitatory",
         "evidence": "AMPK activates fatty acid oxidation via ACC phosphorylation (KEGG hsa04152→hsa00071)"},
        {"source": "I", "target": "B", "weight": 0.45, "sign": "excitatory",
         "evidence": "FAO produces acetyl-CoA feeding TCA cycle (KEGG hsa00071→hsa00020)"},
    ],
}

# =============================================================================
# 10-variable graph: add p53 (J)
# =============================================================================
GRAPHS["10var"] = {
    "description": "Extended: + p53 signalling (J)",
    "variables": {
        **GRAPHS["9var"]["variables"],
        "J": {"name": "p53 signalling", "kegg_id": "hsa04115"},
    },
    "topological_order": ["F", "J", "E", "G", "D", "I", "A", "H", "B", "C"],
    "edges": [
        # All 9-var edges
        *GRAPHS["9var"]["edges"],
        # New edges involving J (p53)
        {"source": "J", "target": "A", "weight": -0.40, "sign": "inhibitory",
         "evidence": "p53 suppresses glycolysis via TIGAR (KEGG hsa04115→hsa00010)"},
        {"source": "J", "target": "C", "weight": 0.45, "sign": "excitatory",
         "evidence": "p53 promotes OXPHOS via SCO2 expression (KEGG hsa04115→hsa00190)"},
    ],
}


def get_graph(size_key: str) -> dict:
    """Return graph definition by size key ('5var', '7var', '9var', '10var')."""
    return GRAPHS[size_key]


def list_sizes() -> list:
    return list(GRAPHS.keys())


if __name__ == "__main__":
    for key in GRAPHS:
        g = GRAPHS[key]
        n_vars = len(g["variables"])
        n_edges = len(g["edges"])
        print(f"{key}: {n_vars} variables, {n_edges} edges")
        print(f"  Topo order: {g['topological_order']}")
        for e in g["edges"]:
            print(f"  {e['source']}→{e['target']} (w={e['weight']:+.2f}, {e['sign']})")
        print()
