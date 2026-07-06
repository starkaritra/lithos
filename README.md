# Grove

> **Working name.** The engine executes *forests* of decision trees on a dataflow fabric — hence "Grove." Rename freely.

A **from-scratch dataflow processor** — its own instruction set, compiler, and cycle-accurate
simulator — built to prove one sharp, measurable claim:

> *A programmable **EDGE** (Explicit Data Graph Execution) dataflow core, minimally extended with
> dynamic-dataflow mechanisms, can beat a GPU at the low-latency **decision-tree ensemble
> inference** (XGBoost / gradient-boosted trees) that runs a large fraction of the world's
> production ML — because that workload is bottlenecked on **branchy control flow**, not on memory
> bandwidth, which is exactly EDGE's home turf.*

## Why this exists (the one-paragraph pitch)
Modern CPUs waste enormous power *rediscovering at runtime* which instructions can run in parallel.
**EDGE** flips that: the compiler expresses the parallelism once, as a dataflow graph, and the
hardware just executes the graph. That idea (from the academic **TRIPS**/Microsoft **E2** projects)
is powerful but has **no clean, open, one-person-readable full-stack implementation** — that gap is
our novelty. We ground it in a real use case — **tabular tree-ensemble inference** — where GPUs
genuinely lose (warp divergence) and the working set stays on-chip (no bandwidth wall).

## Hard constraints
- **$0.** Free tools only (Python; later Verilator / Icarus Verilog). No paid hardware, no FPGA purchase assumed.
- **Simulation-first.** A cycle-accurate software model is the primary artifact; synthesizable RTL is a stretch.
- **Solo, months-scale.** Scope every phase so it finishes.

## Status: PRE-BUILD — one gate remains
We do **not** build the simulator until a cheap ($0, days) **cost-model spike** confirms the win is
real. See `handoffs/experimentAS-handoff.md`. If the spike fails, we fall back to a general-purpose
EDGE contribution **without** the ML framing.

## Honest scope boundary
This is **classical / tabular ML *inference***. It is **NOT** deep-learning or LLM *training* — that
is memory-bandwidth / GEMM territory already owned by open projects (Vortex, Ten-Four, Gemmini) and
by TPUs / SambaNova. Do not let the AI story drift there.

## Folder map
| Path | What |
|---|---|
| `handoff.md` | Master handoff — start here. Context, state, who does what next. |
| `decisions.md` | The decision log (ADRs), D-001…D-011. |
| `architecture.md` | The technical design: EDGE model, ISA sketch, the extension, the phased plan. |
| `handoffs/experimentAS-handoff.md` | The de-risk spike (pre-registration brief). **Do this first.** |
| `handoffs/coderAS-handoff.md` | Build brief: the cost model, then the gated phased build. |
| `research/landscape.md` | researchAS sweep: the architecture-novelty landscape + saturation map. |
| `research/usecase.md` | researchAS sweep: the bottleneck verdict + the tree-inference use case. |

## Provenance
Direction chosen through a discussAS strategy engagement + two researchAS citation-backed sweeps
(2026-07-05). Every load-bearing claim is tagged `[verified]` / `[believed]` / `[guess]` in the
research files; several fabricated citations were caught and discarded there.
