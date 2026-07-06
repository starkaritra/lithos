"""Plots for the spike (spike-prereg.md §9). Headless (Agg) so it runs $0 in CI."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _axis_rows(rows, axis):
    r = [x for x in rows if x.get("axis") == axis]
    return sorted(r, key=lambda x: x["value"])


def _line(ax, rows, axis, ykey, xlabel, title, hline=None):
    r = _axis_rows(rows, axis)
    xs = [str(x["value"]) for x in r]
    ys = [x.get(ykey) for x in r]
    ax.plot(xs, ys, marker="o")
    if hline is not None:
        for y, lbl in hline:
            ax.axhline(y, ls="--", lw=0.8, color="gray")
            ax.text(0, y, f" {lbl}", va="bottom", fontsize=7, color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ykey)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def render_all(sweep: dict, cfg: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    rows = sweep["rows"]
    go = cfg["decision"]["go_primary_rho"]
    nogo = cfg["decision"]["nogo_rho"]
    thresh = [(go, f"GO {go}x"), (nogo, f"NO-GO {nogo}x")]
    written = []

    # 1-4: rho vs each swept axis.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _line(axes[0, 0], rows, "n_estimators", "rho_strong", "#trees",
          "rho vs #trees", thresh)
    _line(axes[0, 1], rows, "max_depth", "rho_strong", "max_depth",
          "rho vs depth", thresh)
    _line(axes[1, 0], rows, "batch", "rho_strong", "batch size",
          "rho vs batch (EDGE edge fades as scalar SIMD kicks in)", thresh)
    _line(axes[1, 1], rows, "N_units", "rho_strong", "N units",
          "rho vs N", thresh)
    fig.suptitle("EDGE / scalar sustained node-evals-per-cycle (rho) [modelled]")
    fig.tight_layout()
    p = os.path.join(out_dir, "rho_sweeps.png")
    fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    # 5: resident footprint vs #trees with SRAM budget lines.
    fig, ax = plt.subplots(figsize=(7, 5))
    r = _axis_rows(rows, "n_estimators")
    xs = [str(x["value"]) for x in r]
    ys = [x["resident_bytes"] / 1024 for x in r]
    ax.plot(xs, ys, marker="s", color="tab:red")
    for kb in (256, 512, 1024, 2048):
        ax.axhline(kb, ls="--", lw=0.8, color="gray")
        ax.text(0, kb, f" {kb} KB", va="bottom", fontsize=7, color="gray")
    ax.set_xlabel("#trees"); ax.set_ylabel("resident footprint (KB)")
    ax.set_title("Resident model footprint vs on-chip SRAM budget (G1) [modelled]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "footprint_vs_trees.png")
    fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    # 6: overhead breakdown (ideal vs charged EDGE throughput) at canonical.
    fig, ax = plt.subplots(figsize=(7, 5))
    cd, cc = sweep["canonical_default"], sweep["canonical_conservative"]
    labels = ["default corner", "conservative corner"]
    ideal = [cd["edge_ideal_rate"], cc["edge_ideal_rate"]]
    charged = [cd["edge_rate"], cc["edge_rate"]]
    x = range(len(labels))
    ax.bar(x, ideal, width=0.5, label="ideal (overhead-free)", color="lightblue")
    ax.bar(x, charged, width=0.5, label="charged (overheads on)", color="tab:blue")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("EDGE node-evals/cycle")
    ax.set_title("EDGE overhead breakdown (G3) [modelled]")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "overhead_breakdown.png")
    fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    # 7: EDGE utilisation vs N.
    fig, ax = plt.subplots(figsize=(7, 5))
    _line(ax, rows, "N_units", "edge_utilisation", "N units",
          "EDGE unit utilisation vs N [modelled]")
    fig.tight_layout()
    p = os.path.join(out_dir, "utilisation_vs_N.png")
    fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    return written
