"""Render the machine-readable GO-gate inputs (spike-prereg.md §5, §9).

This module only *reports* the three pre-registered gate quantities; the final
GO / NO-GO call (with calibrated confidence and the outcome ADR) is made by
experimentAS. A `mechanical_verdict` is included purely as a convenience echo
of the frozen rule — it is not the decision.
"""
from __future__ import annotations


def build_decision_inputs(sweep: dict, cfg: dict) -> dict:
    dcfg = cfg["decision"]
    canon = sweep["canonical_conservative"]  # decision uses conservative corner
    budget = dcfg["sram_budget_bytes"]

    rho_values = [r["rho_strong"] for r in sweep["rows"] if r.get("rho_strong")]
    sweep_min_rho = min(rho_values) if rho_values else None

    primary_rho = canon["rho_strong"]
    resident = canon["resident_bytes"]
    g1_margin = budget / resident if resident else float("inf")
    spilled = resident > budget
    overhead_fraction = canon["overhead_fraction"]

    go = (
        primary_rho is not None and primary_rho >= dcfg["go_primary_rho"]
        and sweep_min_rho is not None and sweep_min_rho >= dcfg["go_sweep_min_rho"]
        and not spilled
        and overhead_fraction < dcfg["overhead_fraction_max"]
    )
    nogo = (
        (primary_rho is not None and primary_rho < dcfg["nogo_rho"])
        or spilled
    )
    verdict = "GO" if go else ("NO-GO" if nogo else "GRAY-ZONE")

    return {
        "canonical_config": {
            "dataset": cfg["dataset"]["canonical"],
            "n_estimators": cfg["xgb"]["canonical"]["n_estimators"],
            "max_depth": cfg["xgb"]["canonical"]["max_depth"],
            "N_units": cfg["resources"]["N_units"],
            "batch": 1,
            "baseline": "stronger of {branchy, branchless-opt}",
            "overheads": "conservative corner",
        },
        "gate_quantities": {
            "primary_rho": primary_rho,
            "sweep_min_rho": sweep_min_rho,
            "g1_resident_bytes": resident,
            "g1_sram_budget_bytes": budget,
            "g1_margin_x": round(g1_margin, 3),
            "g1_spilled": spilled,
            "g3_overhead_fraction": overhead_fraction,
        },
        "thresholds": dict(dcfg),
        "mechanical_verdict": verdict,
        "note": "experimentAS renders the final GO/NO-GO + calibrated confidence; "
                "this verdict is a mechanical echo of the frozen rule (§5).",
    }
