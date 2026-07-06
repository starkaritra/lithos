"""The analytical cost model (spike-prereg.md §3-§4).

Both engines are scored on the *same* parsed ensemble and the *same* sampled
rows; only the architecture differs. Nothing here is silicon: every number is
`[modelled]`. Scalar is an N-wide in-order superscalar; EDGE is N dataflow
units fed by list-scheduling independent tree-traversal chains, with five
overheads charged explicitly so we cannot assume the win.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .ensemble import Ensemble
from .inference import InferenceStats


@dataclass
class EdgeOverheads:
    """EDGE cost knobs (spike-prereg.md §3.4 as revised by Amendment A1).

    A1 split occupancy from latency: tag-match/dispatch/gather are Monsoon-style
    *pipeline stages* whose depth is hidden by in-flight tokens (latency), NOT a
    per-node throughput tax. Only the un-pipelinable tag-match tax `tau_tag` and
    the operand-network cap `R` reduce sustained throughput.
    """

    t_tag: float = 1.0      # tag-match pipeline depth (latency only)
    d_disp: float = 1.0     # dynamic-dispatch pipeline depth (latency only)
    g: float = 1.0          # feature-gather pipeline depth (latency only)
    R_factor: float = 1.0   # operand-network bandwidth cap R = R_factor * N
    tau_tag: float = 0.5    # un-pipelinable tag-match THROUGHPUT tax (occupancy)

    def c_edge_occ(self) -> float:
        """Per node-eval unit *occupancy* in cycles — the only throughput charge."""
        return 1.0 + self.tau_tag

    def c_edge_lat(self) -> float:
        """Per node-eval pipeline *depth* — latency only (critical path + reduction)."""
        return 1.0 + self.t_tag + self.d_disp + self.g


IDEAL_OVERHEADS = EdgeOverheads(t_tag=0.0, d_disp=0.0, g=0.0,
                                R_factor=float("inf"), tau_tag=0.0)


# --------------------------------------------------------------------------- #
# Scalar engine (§3.3)
# --------------------------------------------------------------------------- #
def scalar_branchy_rate(N: int, p_mis: float, B: float, batch: int) -> float:
    """Useful node-evals/cycle for the branchy in-order superscalar.

    control_ceiling = N / (1 + p_mis*B). At batch>1 the scalar is credited with
    SIMD amortisation: masking/predication across up to N lanes spreads a flush
    penalty, shrinking EDGE's edge (§3.6). This is modelled so we do not
    over-claim at larger batch sizes.
    """
    lanes = min(max(batch, 1), N)
    eff_penalty = p_mis * B / lanes
    return N / (1.0 + eff_penalty)


def scalar_branchy_naive_rate(p_mis: float, B: float) -> float:
    """Dependency-honest lower bound: a naive in-order scalar single-issues the
    dependent traversal chain (within-tree ILP ~= 1). Reported to characterise
    rival R-A only (spike-prereg.md Amendment A1) — NOT the decision baseline.
    """
    return 1.0 / (1.0 + p_mis * B)


def scalar_branchless_rates(ens: Ensemble, mean_W: float, N: int,
                            pred_factor: float) -> tuple[float, float]:
    """Useful node-evals/cycle for the branchless/QuickScorer-style scalar.

    No mispredicts, but extra work: pessimistic evaluates every internal node
    of every tree; optimistic evaluates the path x a predication factor. Useful
    throughput = N * (useful work) / (work actually done).
    """
    w_bl_pess = max(ens.total_internal, 1)
    w_bl_opt = max(mean_W * pred_factor, 1e-9)
    rate_pess = N * mean_W / w_bl_pess
    rate_opt = N * mean_W / w_bl_opt
    return rate_opt, rate_pess


# --------------------------------------------------------------------------- #
# EDGE engine (§3.4) — greedy list-schedule of independent chains onto N units
# --------------------------------------------------------------------------- #
def schedule_makespan(chain_lengths: np.ndarray, N: int, c_edge_occ: float,
                      R: float) -> float:
    """Makespan (cycles) to run independent dependent chains on N units.

    Each chain is a serial path of node-evals (a child cannot start before its
    parent finishes). Each node-eval occupies one unit for `c_edge_occ` cycles
    (Amendment A1: occupancy = 1 + tau_tag only; gather/dispatch are latency,
    not occupancy). At most `R` node-evals may *start* per integer cycle
    (operand-network cap). Load imbalance is emergent: short chains drain and
    units idle at the tail.
    """
    chain_lengths = chain_lengths[chain_lengths > 0]
    if chain_lengths.size == 0:
        return 0.0

    units: list[float] = [0.0] * N          # min-heap of unit free-times
    ready: list[tuple[float, int]] = [(0.0, i) for i in range(chain_lengths.size)]
    heapq.heapify(ready)
    remaining = chain_lengths.astype(np.int64).tolist()
    starts_in_cycle: dict[int, int] = {}
    makespan = 0.0
    finite_R = math.isfinite(R)

    while ready:
        rt, cid = heapq.heappop(ready)
        unit_free = heapq.heappop(units)
        start = rt if rt > unit_free else unit_free
        if finite_R:
            bucket = int(start)
            while starts_in_cycle.get(bucket, 0) >= R:
                bucket += 1
            if bucket > start:
                start = float(bucket)
            starts_in_cycle[bucket] = starts_in_cycle.get(bucket, 0) + 1
        finish = start + c_edge_occ
        heapq.heappush(units, finish)
        if finish > makespan:
            makespan = finish
        remaining[cid] -= 1
        if remaining[cid] > 0:
            heapq.heappush(ready, (finish, cid))
    return makespan


def edge_rate(path_len: np.ndarray, row_idx: np.ndarray, n_trees: int, N: int,
              ov: EdgeOverheads, batch: int, rng: np.random.Generator) -> tuple[float, float]:
    """Aggregate EDGE throughput (node-evals/cycle) and mean unit utilisation.

    Rows are grouped into inferences of size `batch`; each group's chains are
    scheduled together. Throughput is time-averaged: sum(work)/sum(makespan).
    Occupancy uses c_edge_occ (throughput tax); the reduction tail uses the full
    pipeline depth c_edge_lat (latency), per Amendment A1.
    """
    c_edge_occ = ov.c_edge_occ()
    R = ov.R_factor * N
    reduction = math.ceil(math.log2(max(n_trees, 1))) * ov.c_edge_lat()

    order = row_idx.copy()
    rng.shuffle(order)
    groups = [order[i:i + batch] for i in range(0, len(order), batch)]

    total_work = 0.0
    total_cycles = 0.0
    for grp in groups:
        chains = path_len[grp, :].reshape(-1)
        w = float(chains.sum())
        if w == 0:
            continue
        makespan = schedule_makespan(chains, N, c_edge_occ, R) + reduction
        total_work += w
        total_cycles += makespan

    if total_cycles == 0:
        return 0.0, 0.0
    rate = total_work / total_cycles
    utilisation = rate / N
    return rate, utilisation


# --------------------------------------------------------------------------- #
# Working-set footprints (§3.5)
# --------------------------------------------------------------------------- #
def footprints(ens: Ensemble, stats: InferenceStats, bytes_per_node: int):
    resident = ens.total_nodes * bytes_per_node + ens.n_features * 4 + 4096
    touched = stats.mean_W * bytes_per_node + ens.n_features * 4 + ens.n_trees * 4
    return int(resident), float(touched)
