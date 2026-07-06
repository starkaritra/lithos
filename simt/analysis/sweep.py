"""Lithos Arm A — analysis layer (the Python half of the gem5-pattern split, D-016).

Drives the C++ mini-GPU (`build/simt.exe`) across parameter sweeps and plots the three
microarchitecture effects the simulator models, turning them into curves you can see:

  1. latency hiding  — cycles vs #warps (should stay ~flat: extra warps hide memory latency)
  2. coalescing      — memory transactions & cycles vs access stride
  3. divergence      — cycles & divergent branches vs branch threshold

Run:  python analysis/sweep.py        (from the simt/ directory, after building)
Outputs PNGs + a CSV to analysis/out/. Everything is [modelled], not measured silicon.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIMT_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")


def find_binary() -> str:
    for name in ("simt.exe", "simt"):
        p = os.path.join(SIMT_DIR, "build", name)
        if os.path.exists(p):
            return p
    sys.exit("could not find build/simt(.exe) — build the project first "
             "(cmake --build build)")


def run(binary: str, kernel_text: str, n_threads: int, mem_words: int,
        mem_latency: int = 200, segment_words: int = 8) -> dict:
    """Assemble+run one configuration and return the parsed JSON stats."""
    with tempfile.NamedTemporaryFile("w", suffix=".sasm", delete=False) as fh:
        fh.write(kernel_text)
        path = fh.name
    try:
        out = subprocess.run(
            [binary, path, str(n_threads), "--json",
             "--mem-latency", str(mem_latency),
             "--segment-words", str(segment_words),
             "--mem-words", str(mem_words)],
            capture_output=True, text=True, check=True,
        )
        return json.loads(out.stdout.strip())
    finally:
        os.unlink(path)


# ---- kernels (generated so Python can parameterize them) ---------------------
def k_two_loads() -> str:
    return "tid r0\nld r1, r0\nld r2, r0\nhalt\n"


def k_strided_load(stride: int) -> str:
    return f"tid r0\nmov r2, {stride}\nimul r3, r0, r2\nld r1, r3\nhalt\n"


def k_divergence(threshold: int) -> str:
    return (
        f"mov r5, {threshold}\n"
        "tid r0\nslt r1, r0, r5\nbra r1, else, join\n"
        "mov r2, 100\niadd r3, r2, r0\nst r0, r3\njmp done\n"
        "else:\nmov r2, 200\niadd r3, r2, r0\nst r0, r3\n"
        "join:\ndone:\nhalt\n"
    )


def k_intensity(k_ops: int) -> str:
    """Load one word, do k_ops arithmetic ops on it, store it. Varying k_ops sweeps the
    arithmetic intensity (compute per memory access) — the axis of the roofline / wall."""
    body = "tid r0\nld r1, r0\nmov r2, 1\n"
    body += "iadd r1, r1, r2\n" * k_ops
    return body + "st r0, r1\nhalt\n"


# ---- experiments -------------------------------------------------------------
def sweep_latency_hiding(binary, rows):
    warps = [1, 2, 4, 8, 16]
    measured, no_hiding = [], []
    base = None
    for w in warps:
        n = 32 * w
        r = run(binary, k_two_loads(), n_threads=n, mem_words=n + 8)
        rows.append({"experiment": "latency_hiding", "x": w, **r})
        measured.append(r["cycles"])
        if base is None:
            base = r["cycles"]
        no_hiding.append(base * w)  # if latency were NOT hidden: linear in #warps

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(warps, measured, "o-", label="measured (latency hidden)")
    ax.plot(warps, no_hiding, "s--", color="gray", label="if NOT hidden (× #warps)")
    ax.set_xlabel("resident warps"); ax.set_ylabel("cycles")
    ax.set_title("Latency hiding: more warps ≈ same time [modelled]")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "latency_hiding.png")


def sweep_coalescing(binary, rows):
    strides = [1, 2, 4, 8]
    txns, cyc = [], []
    for s in strides:
        r = run(binary, k_strided_load(s), n_threads=32, mem_words=32 * s + 8)
        rows.append({"experiment": "coalescing", "x": s, **r})
        txns.append(r["mem_transactions"]); cyc.append(r["cycles"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(strides, txns, "o-", color="tab:red")
    a1.set_xlabel("access stride (words)"); a1.set_ylabel("memory transactions")
    a1.set_title("Coalescing: stride → transactions"); a1.grid(alpha=0.3)
    a2.plot(strides, cyc, "o-", color="tab:red")
    a2.set_xlabel("access stride (words)"); a2.set_ylabel("cycles")
    a2.set_title("...and its cost in cycles"); a2.grid(alpha=0.3)
    fig.suptitle("Uncoalesced access is expensive [modelled]")
    fig.tight_layout()
    _save(fig, "coalescing.png")


def sweep_divergence(binary, rows):
    thresholds = [0, 4, 8, 16, 24, 32]
    cyc, div = [], []
    for t in thresholds:
        r = run(binary, k_divergence(t), n_threads=32, mem_words=40)
        rows.append({"experiment": "divergence", "x": t, **r})
        cyc.append(r["cycles"]); div.append(r["divergent_branches"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, cyc, "o-", label="cycles")
    ax.set_xlabel("branch threshold  (predicate: tid < threshold)")
    ax.set_ylabel("cycles")
    ax.set_title("Divergence: a split warp runs both paths serially [modelled]")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(thresholds, div, width=1.2, alpha=0.25, color="tab:orange",
            label="divergent branches")
    ax2.set_ylabel("divergent branches")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    _save(fig, "divergence.png")


def sweep_memory_wall(binary, rows):
    # One memory load + K arithmetic ops + one store, single warp (latency exposed).
    ks = [0, 25, 50, 100, 200, 400]
    cyc = []
    for k in ks:
        r = run(binary, k_intensity(k), n_threads=32, mem_words=40)
        rows.append({"experiment": "memory_wall", "x": k, **r})
        cyc.append(r["cycles"])
    floor = cyc[0]  # K=0: the time is ~entirely the two memory accesses

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ks, cyc, "o-", label="cycles = memory floor + compute")
    ax.axhline(floor, ls="--", color="gray")
    ax.text(ks[-1], floor, f" memory floor ≈ {floor} cyc\n (2 accesses, ~no compute)",
            va="bottom", ha="right", fontsize=8, color="gray")
    # Region where compute (K cycles) is smaller than the memory floor = memory-bound.
    ax.axvspan(0, floor, alpha=0.08, color="tab:red")
    ax.text(floor / 2, max(cyc) * 0.55,
            "MEMORY-BOUND\n(the wall:\ncompute < 1 access)",
            ha="center", fontsize=8, color="tab:red")
    ax.set_xlabel("arithmetic ops per memory access (K)")
    ax.set_ylabel("cycles")
    ax.set_title("The memory wall: ~one access costs hundreds of arithmetic ops [modelled]")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    _save(fig, "memory_wall.png")


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, SIMT_DIR)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    binary = find_binary()
    print(f"[analysis] driving {os.path.relpath(binary, SIMT_DIR)}")
    rows: list[dict] = []
    sweep_latency_hiding(binary, rows)
    sweep_coalescing(binary, rows)
    sweep_divergence(binary, rows)
    sweep_memory_wall(binary, rows)

    csv_path = os.path.join(OUT, "sweeps.csv")
    keys = ["experiment", "x", "n_threads", "n_warps", "cycles", "warp_instructions",
            "mem_ops", "mem_transactions", "divergent_branches", "mem_latency",
            "segment_words"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {os.path.relpath(csv_path, SIMT_DIR)}")
    print("[analysis] done — see analysis/out/")


if __name__ == "__main__":
    main()
