"""One-command entry point for the Grove D-008 cost-model spike.

    python spike/run.py --config spike/config.yaml         # canonical (HIGGS)
    python spike/run.py --smoke                            # fast offline smoke
    python spike/run.py --dataset covertype                # robustness set

Writes results.csv/json, decision_inputs.json, provenance.json, and plots to
spike/out/. Everything is [modelled] (see spike-prereg.md §0).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import yaml

from . import plots
from .datasets import load_dataset
from .decision import build_decision_inputs
from .provenance import collect
from .sweep import run_sweep

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def _apply_smoke(cfg: dict) -> dict:
    """Shrink everything for a fast, offline end-to-end verification run."""
    cfg["dataset"]["canonical"] = "synthetic"
    cfg["dataset"]["n_train_rows"] = 8000
    cfg["dataset"]["n_eval_rows"] = 200
    cfg["xgb"]["canonical"]["n_estimators"] = 40
    cfg["xgb"]["canonical"]["max_depth"] = 4
    cfg["sweep"] = {
        "n_estimators": [20, 40],
        "max_depth": [3, 4],
        "batch": [1, 8],
        "N_units": [8, 16],
        "p_mis": [0.3],
        "bytes_per_node": [16],
    }
    return cfg


def _write_csv(rows: list[dict], path: str) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grove D-008 cost-model spike")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "config.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="fast offline synthetic run to verify the pipeline")
    ap.add_argument("--dataset", default=None,
                    help="override canonical dataset (higgs|covertype|synthetic)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg = _apply_smoke(cfg)
    if args.dataset:
        cfg["dataset"]["canonical"] = args.dataset

    seed = cfg["seed"]
    os.makedirs(args.out, exist_ok=True)

    name = cfg["dataset"]["canonical"]
    print(f"[load] dataset={name} (seed={seed}) ...")
    bundle = load_dataset(name, cfg["dataset"]["n_train_rows"],
                          cfg["dataset"]["n_eval_rows"], seed)
    print(f"[load] {name}: {bundle.X_train.shape[0]} train / "
          f"{bundle.X_eval.shape[0]} eval rows, {bundle.n_features} features, "
          f"{bundle.n_classes} classes")

    sweep = run_sweep(bundle, cfg, seed)
    decision = build_decision_inputs(sweep, cfg)
    prov = collect(cfg, bundle, seed)

    _write_csv(sweep["rows"], os.path.join(args.out, "results.csv"))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(sweep, f, indent=2, default=str)
    with open(os.path.join(args.out, "decision_inputs.json"), "w") as f:
        json.dump(decision, f, indent=2, default=str)
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)
    written = plots.render_all(sweep, cfg, args.out)

    gq = decision["gate_quantities"]
    print("\n===== D-008 spike — gate quantities [modelled] =====")
    print(f"  primary rho (canonical, conservative, strong scalar): {gq['primary_rho']}")
    print(f"  sweep-min rho:            {gq['sweep_min_rho']}")
    print(f"  G1 resident footprint:    {gq['g1_resident_bytes']} B "
          f"(budget {gq['g1_sram_budget_bytes']} B, margin {gq['g1_margin_x']}x, "
          f"spilled={gq['g1_spilled']})")
    print(f"  G3 overhead fraction:     {gq['g3_overhead_fraction']}")
    print(f"  mechanical verdict:       {decision['mechanical_verdict']} "
          f"(experimentAS renders the final call)")
    print(f"\n  artifacts -> {args.out}")
    print(f"  plots: {', '.join(os.path.basename(p) for p in written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
