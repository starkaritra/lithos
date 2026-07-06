# Lithos Arm A — GPU Architecture, Learned by Building One

Welcome. This folder is a **course**, not just documentation. It teaches you how a GPU
actually works by walking through a real, cycle-accurate **mini-GPU** you can read, run,
and modify (the code in `../src`, `../include`, `../tests`).

The philosophy (borrowed from the best explainers — Feynman, 3Blue1Brown, Patterson &
Hennessy): **motivate *why* a thing exists before *how* it works; build intuition before
formalism; and make every claim measurable.** Every phenomenon in these chapters is not
just described — it is *emergent from the simulator*, and you can reproduce the number.

---

## Who this is for
Someone who knows basic programming and CS fundamentals and wants to *genuinely
understand* GPU / parallel-hardware architecture — the way you'd need to for an
architecture, systems, HPC, or ML-systems role — rather than memorize buzzwords.

## How to read it
Read in order the first time; each chapter builds on the last. Every chapter has the
same skeleton:

1. **The problem** — why this idea had to be invented.
2. **Intuition** — an analogy and a picture before any formalism.
3. **The mechanism** — precisely what happens, with diagrams.
4. **In our mini-GPU** — exactly where and how the code models it (`file` references).
5. **Measure it yourself** — the command to run and the number to expect.
6. **On real GPUs** — how NVIDIA/AMD hardware does the same thing.
7. **References** — real, citable sources to go deeper.

---

## The learning path

| # | Chapter | The one-sentence payoff |
|---|---------|-------------------------|
| 01 | [Why GPUs Exist](01-why-gpus.md) | Throughput vs latency, the memory wall, and Flynn's taxonomy — why a GPU looks nothing like a CPU. |
| 02 | [The SIMT Execution Model](02-simt-execution-model.md) | Threads, warps, and lanes: how thousands of threads run one instruction stream in lockstep. |
| 03 | [Latency Hiding & Occupancy](03-latency-hiding-occupancy.md) | Why a GPU runs *far* more threads than it has lanes — and how that hides slow memory. |
| 04 | [Memory Coalescing](04-memory-coalescing.md) | Why the *pattern* of your memory accesses can change performance by 8× or more. |
| 05 | [Branch Divergence](05-branch-divergence.md) | Why a single `if` can halve a warp's throughput — the effect that shapes what GPUs are good at. |
| 06 | [Cycle-Accurate Simulation](06-cycle-accurate-simulation.md) | What "cycle-accurate" means, the simulator taxonomy, and a full hand-derivation of our 681-cycle result. |
| 07 | [The ISA & Assembler](07-isa-and-assembler.md) | How a kernel becomes instructions, and how our minimal instruction set compares to real PTX/SASS. |
| 08 | [Reduction & Cross-Thread Communication](08-reduction-and-communication.md) | How independent threads combine into one answer — tree reduction, warp-synchronous sync, and why reduction diverges. |
| 09 | [The Memory Wall, Measured](09-memory-wall-measured.md) | Measure the wall on our own machine (one access ≈ 200 arith ops), and see why it motivates Arm C (PIM). |
| — | [Glossary](glossary.md) | Every term, defined once in plain words. |
| — | [References](references.md) | The full, real bibliography. |

---

## The bigger arc (where this is going)
This mini-GPU is **Arm A** of Lithos. Its job is to teach GPU architecture *and* to expose,
by measurement, the single biggest bottleneck in modern computing: the **memory wall** —
processors are starved waiting for data. **Arm C** (later) builds a **near-memory / PIM**
(Processing-In-Memory) architecture that attacks that wall, and measures the win. So the
journey is: *build a GPU → discover why it stalls → build the thing that fixes it.* See
`../../decisions.md` (D-015) and `../../handoff.md`.

> **A note on honesty.** This simulator is an *analytical, cycle-approximate teaching
> model*, not silicon and not a validated microarchitecture. Where a real GPU differs in
> important ways, the chapters say so explicitly (look for the **Reality check** boxes).
> The goal is a *correct mental model*, with the simplifications made visible.
