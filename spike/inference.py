"""Realized inference paths and the data-grounded misprediction estimate.

Path lengths are *data-dependent* (spike-prereg.md §3.1), so we run real
inference on a fixed sample of real rows and record, per row per tree, the
actual root->leaf path length. From the per-node branch direction frequencies
we derive `p_mis` empirically (§3.3) instead of assuming it — this is what
kills / confirms rival hypothesis R-A ("tree branches are predictable").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ensemble import Ensemble, Tree


@dataclass
class InferenceStats:
    path_len: np.ndarray      # (n_rows, n_trees) int32; realized root->leaf depth+1
    p_mis_data: float         # data-grounded per-node misprediction probability
    mean_W: float             # mean total node-evals per inference (sum over trees)
    mean_path_max: float      # mean over rows of max path length across trees (for L*)


def _traverse_tree(tree: Tree, X: np.ndarray):
    """Vectorised traversal of one tree for all rows.

    Returns (path_len[n_rows], yes_visits[n_nodes], total_visits[n_nodes]).
    """
    n_rows = X.shape[0]
    cur = np.zeros(n_rows, dtype=np.int64)
    length = np.zeros(n_rows, dtype=np.int32)
    active = tree.feature[cur] >= 0

    yes_visits = np.zeros(tree.n_nodes, dtype=np.int64)
    total_visits = np.zeros(tree.n_nodes, dtype=np.int64)

    # Guard against pathological cycles; a tree can be at most n_nodes deep.
    for _ in range(tree.n_nodes + 1):
        if not active.any():
            break
        idx = np.where(active)[0]
        nodes = cur[idx]
        feats = tree.feature[nodes]
        vals = X[idx, feats]
        thr = tree.threshold[nodes]

        missing = np.isnan(vals)
        go_yes = vals < thr  # xgboost: x < split_condition -> yes child

        np.add.at(total_visits, nodes, 1)
        np.add.at(yes_visits, nodes, go_yes.astype(np.int64))

        child = np.where(go_yes, tree.yes[nodes], tree.no[nodes])
        child = np.where(missing, tree.missing[nodes], child)

        cur[idx] = child
        length[idx] += 1
        active = tree.feature[cur] >= 0

    return length + 1, yes_visits, total_visits  # +1 counts the leaf node-eval


def run_inference(ens: Ensemble, X: np.ndarray) -> InferenceStats:
    """Compute realized path lengths and the data-grounded p_mis over `X`."""
    n_rows = X.shape[0]
    path_len = np.zeros((n_rows, ens.n_trees), dtype=np.int32)

    weighted_pmis_num = 0.0
    weight_den = 0.0

    for t, tree in enumerate(ens.trees):
        length, yes_v, tot_v = _traverse_tree(tree, X)
        path_len[:, t] = length

        internal = tree.feature >= 0
        tv = tot_v[internal].astype(np.float64)
        yv = yes_v[internal].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(tv > 0, yv / tv, 0.0)
        # A 2-bit/bias predictor learns the majority direction; the achievable
        # mispredict rate at a node is min(f, 1-f). Weight by visit count so
        # hot nodes dominate (path-visit-weighted mean, §3.3).
        node_pmis = np.minimum(f, 1.0 - f)
        weighted_pmis_num += float((node_pmis * tv).sum())
        weight_den += float(tv.sum())

    p_mis = weighted_pmis_num / weight_den if weight_den > 0 else 0.0
    mean_W = float(path_len.sum(axis=1).mean())
    mean_path_max = float(path_len.max(axis=1).mean())
    return InferenceStats(
        path_len=path_len,
        p_mis_data=p_mis,
        mean_W=mean_W,
        mean_path_max=mean_path_max,
    )
