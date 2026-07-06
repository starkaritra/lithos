"""Parse an XGBoost booster into flat, per-tree arrays (spike-prereg.md §3.1).

The cost model is engine-agnostic and dataset-agnostic: everything downstream
consumes this `Ensemble` structure, never xgboost directly. Node ids are dense
per tree (0..n-1); leaves have feature == -1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

_FEAT_RE = re.compile(r"[fF]?(\d+)")


@dataclass
class Tree:
    """One decision tree as parallel arrays indexed by dense node id."""

    feature: np.ndarray      # int32; split feature index, -1 for a leaf
    threshold: np.ndarray    # float32; split condition (x < threshold -> yes)
    yes: np.ndarray          # int32; child id if x < threshold (or missing dir)
    no: np.ndarray           # int32; child id otherwise
    missing: np.ndarray      # int32; child id when the feature is missing
    leaf_value: np.ndarray   # float32; only meaningful for leaves
    depth: np.ndarray        # int32; node depth (root = 0)

    @property
    def n_nodes(self) -> int:
        return int(self.feature.shape[0])

    @property
    def n_internal(self) -> int:
        return int(np.count_nonzero(self.feature >= 0))

    @property
    def max_depth(self) -> int:
        return int(self.depth.max()) if self.n_nodes else 0


@dataclass
class Ensemble:
    """A parsed tree ensemble plus the feature count it expects."""

    trees: list[Tree]
    n_features: int

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    @property
    def total_nodes(self) -> int:
        return sum(t.n_nodes for t in self.trees)

    @property
    def total_internal(self) -> int:
        return sum(t.n_internal for t in self.trees)


def _feature_index(split) -> int:
    """XGBoost json dump gives 'f12' (or an int) for the split feature."""
    if isinstance(split, (int, np.integer)):
        return int(split)
    m = _FEAT_RE.fullmatch(str(split).strip())
    if not m:
        raise ValueError(f"cannot parse split feature {split!r}")
    return int(m.group(1))


def _parse_tree(node: dict) -> Tree:
    """Walk one tree's nested json dump into dense arrays.

    We assign a compact contiguous id to every node in traversal order so the
    arrays are gap-free regardless of xgboost's internal nodeids.
    """
    # First pass: assign dense ids in a stable DFS order.
    order: list[dict] = []
    remap: dict[int, int] = {}

    def _collect(n: dict, depth: int) -> None:
        remap[n["nodeid"]] = len(order)
        n["_depth"] = depth
        order.append(n)
        if "children" in n:
            for c in n["children"]:
                _collect(c, depth + 1)

    _collect(node, 0)

    m = len(order)
    feature = np.full(m, -1, dtype=np.int32)
    threshold = np.zeros(m, dtype=np.float32)
    yes = np.full(m, -1, dtype=np.int32)
    no = np.full(m, -1, dtype=np.int32)
    missing = np.full(m, -1, dtype=np.int32)
    leaf_value = np.zeros(m, dtype=np.float32)
    depth = np.zeros(m, dtype=np.int32)

    for i, n in enumerate(order):
        depth[i] = n["_depth"]
        if "leaf" in n:
            leaf_value[i] = float(n["leaf"])
            continue
        feature[i] = _feature_index(n["split"])
        threshold[i] = float(n.get("split_condition", 0.0))
        yes[i] = remap[n["yes"]]
        no[i] = remap[n["no"]]
        missing[i] = remap[n.get("missing", n["yes"])]

    return Tree(feature, threshold, yes, no, missing, leaf_value, depth)


def parse_booster(booster, n_features: int) -> Ensemble:
    """Build an :class:`Ensemble` from a trained xgboost Booster."""
    dumps = booster.get_dump(with_stats=False, dump_format="json")
    trees = [_parse_tree(json.loads(d)) for d in dumps]
    return Ensemble(trees=trees, n_features=int(n_features))
