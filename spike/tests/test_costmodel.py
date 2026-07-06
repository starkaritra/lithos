"""Behavioural / contract tests for the cost model (pre-reg §8: any claimed
number must be backed by a runnable test). These assert *correctness and
reproducibility*, never a pass/fail metric threshold (thresholds are inputs)."""
from __future__ import annotations

import numpy as np

from spike.costmodel import (IDEAL_OVERHEADS, EdgeOverheads, edge_rate,
                             footprints, scalar_branchless_rates,
                             scalar_branchy_rate, schedule_makespan)
from spike.ensemble import Ensemble, Tree
from spike.inference import run_inference


def _stump(feat=0, thr=0.5):
    """A 3-node tree: split on `feat`; yes->leaf +1, no->leaf -1."""
    return Tree(
        feature=np.array([feat, -1, -1], dtype=np.int32),
        threshold=np.array([thr, 0, 0], dtype=np.float32),
        yes=np.array([1, -1, -1], dtype=np.int32),
        no=np.array([2, -1, -1], dtype=np.int32),
        missing=np.array([1, -1, -1], dtype=np.int32),
        leaf_value=np.array([0, 1.0, -1.0], dtype=np.float32),
        depth=np.array([0, 1, 1], dtype=np.int32),
    )


# ---- scheduler ------------------------------------------------------------ #
def test_single_chain_makespan_equals_length():
    assert schedule_makespan(np.array([5]), N=4, c_edge=1.0, R=float("inf")) == 5.0


def test_independent_chains_pack_onto_units():
    # two length-2 chains on 2 units, unit-cost -> both finish in 2 cycles
    assert schedule_makespan(np.array([2, 2]), N=2, c_edge=1.0, R=float("inf")) == 2.0


def test_one_unit_serializes():
    # 2 chains x 2 nodes on a single unit -> 4 cycles of occupancy
    assert schedule_makespan(np.array([2, 2]), N=1, c_edge=1.0, R=float("inf")) == 4.0


def test_c_edge_scales_makespan():
    base = schedule_makespan(np.array([3]), N=1, c_edge=1.0, R=float("inf"))
    doubled = schedule_makespan(np.array([3]), N=1, c_edge=2.0, R=float("inf"))
    assert doubled == 2 * base


def test_empty_schedule():
    assert schedule_makespan(np.array([0, 0]), N=4, c_edge=1.0, R=float("inf")) == 0.0


# ---- scalar model --------------------------------------------------------- #
def test_branchy_rate_perfect_prediction_hits_issue_ceiling():
    assert scalar_branchy_rate(N=16, p_mis=0.0, B=12, batch=1) == 16.0


def test_branchy_rate_mispredict_throttles():
    assert scalar_branchy_rate(16, 0.5, 12, 1) < 16.0


def test_batch_simd_credit_recovers_rate():
    lo = scalar_branchy_rate(16, 0.5, 12, batch=1)
    hi = scalar_branchy_rate(16, 0.5, 12, batch=16)
    assert hi > lo  # SIMD across rows shrinks EDGE's edge at larger batch


def test_branchless_rate_contract():
    # optimistic = N/pred_factor; pessimistic = N * useful / internal_nodes.
    ens = Ensemble(trees=[_stump(), _stump()], n_features=1)  # 2 internal nodes
    opt, pess = scalar_branchless_rates(ens, mean_W=4.0, N=16, pred_factor=2.0)
    assert opt == 16.0 / 2.0
    assert pess == 16.0 * 4.0 / 2.0


def test_branchless_more_internal_nodes_lowers_pess_rate():
    small = Ensemble(trees=[_stump()], n_features=1)              # 1 internal
    big = Ensemble(trees=[_stump() for _ in range(8)], n_features=1)  # 8 internal
    _, pess_small = scalar_branchless_rates(small, 4.0, 16, 2.0)
    _, pess_big = scalar_branchless_rates(big, 4.0, 16, 2.0)
    assert pess_big < pess_small  # more nodes to evaluate -> lower useful rate


# ---- inference & p_mis ---------------------------------------------------- #
def test_realized_path_length():
    ens = Ensemble(trees=[_stump(feat=0, thr=0.5)], n_features=1)
    X = np.array([[0.0], [1.0]], dtype=np.float32)  # one yes, one no
    stats = run_inference(ens, X)
    assert stats.path_len.shape == (2, 1)
    assert (stats.path_len == 2).all()  # root + leaf


def test_pmis_in_valid_range_and_balanced_split():
    ens = Ensemble(trees=[_stump(feat=0, thr=0.5)], n_features=1)
    X = np.array([[0.0], [1.0]], dtype=np.float32)  # 50/50 split
    stats = run_inference(ens, X)
    assert 0.0 <= stats.p_mis_data <= 0.5
    assert abs(stats.p_mis_data - 0.5) < 1e-6  # unpredictable node


def test_pmis_zero_for_predictable_split():
    ens = Ensemble(trees=[_stump(feat=0, thr=0.5)], n_features=1)
    X = np.array([[0.0], [0.1], [0.2]], dtype=np.float32)  # always yes
    stats = run_inference(ens, X)
    assert stats.p_mis_data < 1e-6


# ---- footprints ----------------------------------------------------------- #
def test_footprint_formula():
    ens = Ensemble(trees=[_stump(), _stump()], n_features=4)
    X = np.zeros((3, 4), dtype=np.float32)
    stats = run_inference(ens, X)
    resident, touched = footprints(ens, stats, bytes_per_node=16)
    # 6 nodes * 16 + 4 feats * 4 + 4096 buffer
    assert resident == 6 * 16 + 4 * 4 + 4096


# ---- overheads reduce EDGE rate (G3 direction) ---------------------------- #
def test_overheads_reduce_edge_rate():
    ens = Ensemble(trees=[_stump() for _ in range(8)], n_features=1)
    X = np.zeros((10, 1), dtype=np.float32)
    stats = run_inference(ens, X)
    rows = np.arange(10)
    ideal, _ = edge_rate(stats.path_len, rows, ens.n_trees, 8, IDEAL_OVERHEADS,
                         1, np.random.default_rng(0))
    charged, _ = edge_rate(stats.path_len, rows, ens.n_trees, 8,
                           EdgeOverheads(2, 2, 4, 0.5), 1, np.random.default_rng(0))
    assert charged < ideal
