"""Evaluate one cost-model point and run the pre-registered sweep (§5, §6).

The GO/NO-GO decision is made at the *canonical* config against the
*conservative* overhead corner and the *stronger* scalar baseline; the sweep is
descriptive (robustness), not a search for a favourable point. No thresholds
move after data is seen — this module only measures and records.
"""
from __future__ import annotations

import numpy as np

from .costmodel import (IDEAL_OVERHEADS, EdgeOverheads, edge_rate, footprints,
                        scalar_branchless_rates, scalar_branchy_rate)
from .datasets import DataBundle, train_or_load
from .ensemble import Ensemble, parse_booster
from .inference import InferenceStats, run_inference


def _overheads(cfg: dict, corner: str) -> EdgeOverheads:
    o = cfg["edge"][corner]
    return EdgeOverheads(t_tag=o["t_tag"], d_disp=o["d_disp"], g=o["g"],
                         R_factor=o["R_factor"])


def _sched_rows(n_eval: int, batch: int, base: int = 300) -> np.ndarray:
    """Row indices used for EDGE scheduling (bounded for runtime).

    Footprints and p_mis use all eval rows; only the scheduler subsamples.
    Larger batches need more rows to form representative groups.
    """
    n = min(n_eval, max(base, batch * 4))
    return np.arange(n)


def evaluate_point(ens: Ensemble, stats: InferenceStats, cfg: dict, *,
                   N: int, ov: EdgeOverheads, batch: int, p_mis: float,
                   bytes_per_node: int, seed: int) -> dict:
    """Compute all §3 metrics for a single (config, overhead) point."""
    B = cfg["scalar"]["flush_penalty_B"]
    pred_factor = cfg["scalar"]["branchless_predication_factor"]

    branchy = scalar_branchy_rate(N, p_mis, B, batch)
    bl_opt, bl_pess = scalar_branchless_rates(ens, stats.mean_W, N, pred_factor)
    strong_scalar = max(branchy, bl_opt)  # conservative for EDGE

    rows = _sched_rows(stats.path_len.shape[0], batch)
    rng = np.random.default_rng(seed)
    e_rate, util = edge_rate(stats.path_len, rows, ens.n_trees, N, ov, batch, rng)
    ideal_rate, _ = edge_rate(stats.path_len, rows, ens.n_trees, N,
                              IDEAL_OVERHEADS, batch, np.random.default_rng(seed))

    resident, touched = footprints(ens, stats, bytes_per_node)
    overhead_fraction = 1.0 - (e_rate / ideal_rate) if ideal_rate > 0 else 0.0

    return {
        "N": N, "batch": batch, "p_mis": round(p_mis, 4),
        "bytes_per_node": bytes_per_node,
        "c_edge": ov.c_edge(), "R_factor": ov.R_factor,
        "n_trees": ens.n_trees, "total_nodes": ens.total_nodes,
        "mean_W": round(stats.mean_W, 3), "L_star": round(stats.mean_path_max + np.log2(max(ens.n_trees, 1)), 3),
        "scalar_branchy": round(branchy, 4),
        "scalar_branchless_opt": round(bl_opt, 4),
        "scalar_branchless_pess": round(bl_pess, 4),
        "strong_scalar": round(strong_scalar, 4),
        "edge_rate": round(e_rate, 4),
        "edge_ideal_rate": round(ideal_rate, 4),
        "edge_utilisation": round(util, 4),
        "overhead_fraction": round(overhead_fraction, 4),
        "rho_vs_branchy": round(e_rate / branchy, 4) if branchy else None,
        "rho_vs_branchless_opt": round(e_rate / bl_opt, 4) if bl_opt else None,
        "rho_strong": round(e_rate / strong_scalar, 4) if strong_scalar else None,
        "resident_bytes": resident,
        "touched_bytes": round(touched, 1),
    }


def _prep(bundle: DataBundle, n_estimators: int, max_depth: int, cfg: dict,
          seed: int) -> tuple[Ensemble, InferenceStats]:
    booster = train_or_load(bundle, n_estimators, max_depth, cfg, seed)
    ens = parse_booster(booster, bundle.n_features)
    stats = run_inference(ens, bundle.X_eval)
    return ens, stats


def run_sweep(bundle: DataBundle, cfg: dict, seed: int, log=print) -> dict:
    """Run canonical point + one-axis-at-a-time sweep. Returns a result dict."""
    can = cfg["xgb"]["canonical"]
    N0 = cfg["resources"]["N_units"]
    bpn0 = cfg["resources"]["bytes_per_node"]
    ov_default = _overheads(cfg, "default")
    ov_conservative = _overheads(cfg, "conservative")

    log(f"[prep] training/loading canonical model "
        f"({can['n_estimators']} trees, depth {can['max_depth']}) on {bundle.name}...")
    ens0, stats0 = _prep(bundle, can["n_estimators"], can["max_depth"], cfg, seed)
    log(f"[prep] ensemble: {ens0.n_trees} trees, {ens0.total_nodes} nodes, "
        f"data-grounded p_mis={stats0.p_mis_data:.4f}, mean_W={stats0.mean_W:.2f}")

    def point(ens, stats, *, axis, value, N=None, ov=None, batch=1, p_mis=None,
              bpn=None):
        row = evaluate_point(
            ens, stats, cfg, N=N or N0, ov=ov or ov_default, batch=batch,
            p_mis=stats.p_mis_data if p_mis is None else p_mis,
            bytes_per_node=bpn or bpn0, seed=seed,
        )
        row["axis"], row["value"] = axis, value
        return row

    rows: list[dict] = []

    # Canonical point at DEFAULT and CONSERVATIVE corners (decision uses conservative).
    canonical_default = point(ens0, stats0, axis="canonical", value="default")
    canonical_conservative = point(ens0, stats0, axis="canonical",
                                   value="conservative", ov=ov_conservative)
    rows += [canonical_default, canonical_conservative]

    # #trees and depth axes require retraining.
    for T in cfg["sweep"]["n_estimators"]:
        if T == can["n_estimators"]:
            rows.append({**canonical_default, "axis": "n_estimators", "value": T})
            continue
        log(f"[sweep] n_estimators={T}")
        ens, stats = _prep(bundle, T, can["max_depth"], cfg, seed)
        rows.append(point(ens, stats, axis="n_estimators", value=T))

    for d in cfg["sweep"]["max_depth"]:
        if d == can["max_depth"]:
            rows.append({**canonical_default, "axis": "max_depth", "value": d})
            continue
        log(f"[sweep] max_depth={d}")
        ens, stats = _prep(bundle, can["n_estimators"], d, cfg, seed)
        rows.append(point(ens, stats, axis="max_depth", value=d))

    # Remaining axes reuse the canonical ensemble/paths.
    for b in cfg["sweep"]["batch"]:
        rows.append(point(ens0, stats0, axis="batch", value=b, batch=b))
    for n in cfg["sweep"]["N_units"]:
        rows.append(point(ens0, stats0, axis="N_units", value=n, N=n))
    for pm in cfg["sweep"]["p_mis"]:
        rows.append(point(ens0, stats0, axis="p_mis", value=pm, p_mis=pm))
    for bpn in cfg["sweep"]["bytes_per_node"]:
        rows.append(point(ens0, stats0, axis="bytes_per_node", value=bpn, bpn=bpn))
    # Overhead corner as an explicit sweep point (rival R-B).
    rows.append(point(ens0, stats0, axis="overhead", value="conservative",
                      ov=ov_conservative))

    return {
        "rows": rows,
        "canonical_default": canonical_default,
        "canonical_conservative": canonical_conservative,
        "p_mis_data": stats0.p_mis_data,
        "n_eval_rows": int(bundle.X_eval.shape[0]),
    }
