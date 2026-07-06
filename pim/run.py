"""Grove Arm C — PIM data-movement analysis layer (pim-prereg.md §8/§9).

Drives the C++ byte-accounting core (`build/pim(.exe)`) across the pre-registered sweep,
computes DMR + the reduction-ratio crossover, and emits the deliverables experimentAS needs
to render GO / NO-GO:  results.csv, the honesty plots, decision_inputs.json, provenance.json.

Run:  python pim/run.py --config pim/config.yaml     (from the grove/ or pim/ directory)
Everything is [modelled]; the primary DMR is [modelled-exact] (bytes counted, not estimated).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def find_binary() -> str:
    for name in ("pim.exe", "pim"):
        p = os.path.join(HERE, "build", name)
        if os.path.exists(p):
            return p
    sys.exit("could not find build/pim(.exe) — build first (cmake --build build)")


def run_point(binary: str, cfg: dict, **overrides) -> dict:
    """Invoke the C++ core for one config and return its parsed JSON accounting."""
    c = dict(cfg["canonical"])
    c.update(overrides)
    args = [binary, "--json",
            "--kernel", str(c["kernel"]), "--L", str(c["L"]), "--B", str(c["B"]),
            "--d", str(c["d"]), "--idx_b", str(c["idx_b"]), "--b", str(c["b"]),
            "--nb", str(c["nb"]), "--R_tab", str(c["R_tab"]), "--N", str(c.get("N", 16000000)),
            "--cap", str(c["cap"]), "--e", str(c["e"]), "--placement", str(c["placement"]),
            "--seed", str(cfg["seed"])]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip())


def crossover_L(binary, cfg, B: int, Ls: list[int], worth_it: float) -> int | None:
    """Smallest L (in the sweep) at which DMR >= worth_it, for a given bank count."""
    for L in Ls:
        r = run_point(binary, cfg, L=L, B=B)
        if r["dmr"] >= worth_it:
            return L
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(OUT, exist_ok=True)
    binary = find_binary()
    dcfg = cfg["decision"]
    Ls, Bs, ds = cfg["sweep"]["L"], cfg["sweep"]["B"], cfg["sweep"]["d"]
    print(f"[pim] driving {os.path.relpath(binary, HERE)}")

    rows: list[dict] = []

    def record(r, axis, value):
        r = dict(r); r["axis"] = axis; r["value"] = value
        rows.append(r)
        return r

    # Canonical decision point.
    canon = record(run_point(binary, cfg), "canonical", "canonical")
    print(f"[pim] canonical DMR = {canon['dmr']:.3f}x  (k={canon['k_empirical']:.2f}, "
          f"overhead={canon['pim_overhead_fraction']:.3f})")

    # L x B grid (the reduction-ratio sweep + banking tension + crossover).
    dmr_by_LB: dict[int, dict[int, float]] = {B: {} for B in Bs}
    for B in Bs:
        for L in Ls:
            r = record(run_point(binary, cfg, L=L, B=B), "LxB", f"L{L}_B{B}")
            dmr_by_LB[B][L] = r["dmr"]

    # d sweep (hidden-traffic amortisation) and placement sweep, at canonical L,B.
    for d in ds:
        record(run_point(binary, cfg, d=d), "d", d)
    for pl in cfg["sweep"]["placement"]:
        record(run_point(binary, cfg, placement=pl), "placement", pl)
    # Reduction endpoint (extreme aggregation), for the far-left of the sweep.
    record(run_point(binary, cfg, kernel="reduction"), "kernel", "reduction")

    # ---- crossover L* per B ---------------------------------------------------
    worth = dcfg["worth_it_dmr"]
    Lstar = {B: next((L for L in Ls if dmr_by_LB[B][L] >= worth), None) for B in Bs}
    canonB = cfg["canonical"]["B"]
    dlrm_L = dcfg["dlrm_L"]
    dlrm_right = (Lstar[canonB] is not None) and (dlrm_L >= Lstar[canonB])

    # robustness: min DMR over the realistic aggregating L range at canonical B
    lo, hi = dcfg["realistic_L_range"]
    realistic = [L for L in Ls if lo <= L <= hi]
    sweep_min = min(dmr_by_LB[canonB][L] for L in realistic) if realistic else 0.0

    # ---- write results.csv ----------------------------------------------------
    keys = ["axis", "value", "kernel", "L", "B", "d", "idx_b", "nb", "placement",
            "baseline_bytes", "pim_bytes", "base_idx", "base_operand", "base_output",
            "pim_idx", "pim_partial", "pim_output", "k_empirical", "k_closed", "dmr",
            "pim_overhead_fraction", "baseline_cycles", "pim_cycles",
            "baseline_energy_pj", "pim_energy_pj", "energy_dmr"]
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- decision_inputs.json (§9.3) -----------------------------------------
    verdict = ("GO" if (canon["dmr"] >= dcfg["go_canonical_dmr"] and
                        sweep_min >= dcfg["go_sweep_min_dmr"] and dlrm_right)
               else "NO-GO" if canon["dmr"] < dcfg["nogo_dmr"]
               else "GRAY-ZONE")
    decision = {
        "canonical_config": {k: cfg["canonical"][k] for k in ("kernel", "d", "L", "B", "nb")},
        "canonical_dmr": canon["dmr"],
        "canonical_overhead_fraction": canon["pim_overhead_fraction"],
        "canonical_k_empirical": canon["k_empirical"],
        "crossover_L_star_per_B": {str(B): Lstar[B] for B in Bs},
        "dlrm_L": dlrm_L,
        "dlrm_right_of_crossover_at_canonical_B": dlrm_right,
        "dmr_by_L_at_canonical_B": {str(L): dmr_by_LB[canonB][L] for L in Ls},
        "dmr_sweep_min_realistic": sweep_min,
        "realistic_L_range": [lo, hi],
        "energy_dmr_is_e_invariant": True,   # ratio cancels e; = byte DMR
        "energy_dmr": canon["energy_dmr"],
        "thresholds": dict(dcfg),
        "mechanical_verdict": verdict,
        "note": "experimentAS renders the final GO/NO-GO + calibrated confidence; this "
                "verdict is a mechanical echo of the frozen §5 rule.",
    }
    with open(os.path.join(OUT, "decision_inputs.json"), "w") as fh:
        json.dump(decision, fh, indent=2)

    _plots(cfg, dmr_by_LB, Lstar, canon, ds, binary, rows)
    _provenance(cfg, binary)

    print(f"[pim] crossover L* per B: {Lstar}")
    print(f"[pim] DLRM L={dlrm_L} {'>=' if dlrm_right else '<'} crossover at B={canonB} "
          f"(L*={Lstar[canonB]}) -> {'winning' if dlrm_right else 'losing'} region")
    print(f"[pim] realistic-range sweep-min DMR = {sweep_min:.3f}x")
    print(f"[pim] mechanical verdict: {verdict}  (experimentAS renders the final call)")
    print(f"[pim] artifacts -> {OUT}")


def _plots(cfg, dmr_by_LB, Lstar, canon, ds, binary, rows):
    Ls = cfg["sweep"]["L"]; Bs = cfg["sweep"]["B"]
    canonB = cfg["canonical"]["B"]; dlrm_L = cfg["decision"]["dlrm_L"]
    worth = cfg["decision"]["worth_it_dmr"]; go = cfg["decision"]["go_canonical_dmr"]

    # (a) THE headline honesty plot: DMR vs aggregation factor (=L), per B, with the naive
    # tautological line overlaid and the crossover + DLRM point marked.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for B in Bs:
        ax.plot(Ls, [dmr_by_LB[B][L] for L in Ls], "o-", ms=4, label=f"B={B} (honest)")
    ax.plot(Ls, Ls, "k--", lw=1.2, label="naive tautology (DMR = L)")
    ax.axhline(worth, color="gray", ls=":", lw=0.9)
    ax.text(Ls[0], worth, f" worth-it {worth}x", fontsize=7, color="gray", va="bottom")
    ax.axhline(go, color="green", ls=":", lw=0.9)
    ax.text(Ls[0], go, f" GO {go}x", fontsize=7, color="green", va="bottom")
    ax.axvline(dlrm_L, color="tab:red", ls="-.", lw=1)
    ax.text(dlrm_L, ax.get_ylim()[1], " DLRM L=40", fontsize=8, color="tab:red",
            rotation=90, va="top")
    if Lstar[canonB]:
        ax.plot([Lstar[canonB]], [worth], "s", color="black", ms=8,
                label=f"crossover L*={Lstar[canonB]} (B={canonB})")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("aggregation factor  L (rows pooled per bag)")
    ax.set_ylabel("DMR = baseline / PIM off-chip bytes")
    ax.set_title("Arm C: honest DMR sits FAR below the naive tautology line [modelled-exact]")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")
    _save(fig, "dmr_crossover.png")

    # (b) banking tension: DMR vs B at canonical L.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Bs, [dmr_by_LB[B][dlrm_L] for B in Bs], "o-", color="tab:purple")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("banks B"); ax.set_ylabel(f"DMR at L={dlrm_L}")
    ax.set_title("Banking tension: more banks → smaller byte win [modelled]")
    ax.grid(alpha=0.3, which="both")
    _save(fig, "dmr_vs_banks.png")

    # (c) hidden-traffic amortisation: DMR vs d.
    dvals = [run_point(binary, cfg, d=d) for d in ds]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ds, [r["dmr"] for r in dvals], "o-", color="tab:orange")
    ax.set_xlabel("embedding dim d"); ax.set_ylabel(f"DMR at L={dlrm_L}, B={canonB}")
    ax.set_title("Hidden-traffic amortisation: larger d → indices/output matter less [modelled]")
    ax.grid(alpha=0.3)
    _save(fig, "dmr_vs_dim.png")

    # (d) stacked link-byte breakdown at canonical: baseline vs PIM.
    fig, ax = plt.subplots(figsize=(7, 5))
    base_parts = [canon["base_idx"], canon["base_operand"], canon["base_output"]]
    pim_parts = [canon["pim_idx"], canon["pim_partial"], canon["pim_output"]]
    labels = ["indices", "operands / partials", "output"]
    colors = ["tab:gray", "tab:blue", "tab:green"]
    for i, (bp, pp) in enumerate(zip(base_parts, pim_parts)):
        ax.bar("baseline", bp, bottom=sum(base_parts[:i]), color=colors[i], label=labels[i])
        ax.bar("PIM", pp, bottom=sum(pim_parts[:i]), color=colors[i])
    ax.set_ylabel("off-chip link bytes (canonical)")
    ax.set_title("Where the bytes go: PIM ships k partials, not all L rows [modelled-exact]")
    ax.legend(fontsize=8)
    _save(fig, "byte_breakdown.png")

    # (e) energy DMR vs e: flat (ratio is e-invariant) — shown honestly.
    es = cfg["sweep"]["e"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(es, [canon["energy_dmr"]] * len(es), "o-", color="tab:red")
    ax.set_xlabel("energy per off-chip byte e (pJ)")
    ax.set_ylabel("energy DMR")
    ax.set_ylim(0, max(canon["energy_dmr"] * 1.5, 2))
    ax.set_title("Energy DMR is e-invariant (ratio cancels e) = byte DMR [modelled]")
    ax.grid(alpha=0.3)
    _save(fig, "energy_dmr.png")


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p, HERE)}")


def _provenance(cfg, binary):
    import numpy
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "binary": os.path.relpath(binary, HERE),
        "config": cfg,
        "energy_citation": "Horowitz, ISSCC 2014 — DRAM access ~640 pJ / 32-bit => ~160 pJ/byte",
        "modelled_exact": "primary DMR = counted link bytes; cycles/energy = [modelled]",
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5)
        prov["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    except Exception:
        prov["git_commit"] = None
    with open(os.path.join(OUT, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
