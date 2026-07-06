# Grove — Architecture

This document motivates the design before it specifies it. If a term isn't defined, it's defined at
first use. The **detailed** ISA spec is deliberately deferred to Phase 0 (below) — this is the
orientation, not the final contract.

---

## 1. The problem EDGE attacks
A conventional CPU is **von Neumann**: one instruction stream, and instructions communicate through a
shared **register file** (instruction A writes register `r5`, instruction B reads `r5`). To run fast,
modern out-of-order CPUs spend huge power and silicon *discovering at runtime* which instructions are
independent (register renaming, dependency tracking, wakeup/select, a big scheduler). This scales
badly — roughly, doubling the parallelism quadruples the bookkeeping. That ceiling is the
**instruction-level-parallelism (ILP) wall**.

**EDGE's bet:** stop rediscovering parallelism at runtime. Let the **compiler** express it once, as a
**dataflow graph**, and build hardware that just executes the graph.

## 2. The execution model
**EDGE = Explicit Data Graph Execution.** Two moving parts:

**(a) Block-atomic execution.** The compiler chops the program into **hyperblocks** — chunks of ~tens
of instructions, single-entry, single-exit, with internal branches removed by **predication** (a
branch becomes a conditionally-executed instruction rather than a jump). A block commits all-or-nothing.
Control flow lives *between* blocks; dataflow lives *inside* them. → *dataflow in the small, von
Neumann in the large.*

**(b) Target-form dataflow.** Inside a block, instructions **do not write registers**. Each
instruction names *which instruction(s) consume its result*, and **fires the instant its operands
arrive**. No intra-block register-file traffic → none of the renaming/scheduling machinery is needed.

Tiny example — compute `d = (a+b) * (a-c)`:
```
conventional (register file)       EDGE (target form)
  ADD r3, r1, r2   ; a+b             I0: ADD -> I2.left     ; result routed INTO I2's left input
  SUB r4, r1, r5   ; a-c             I1: SUB -> I2.right    ; result routed INTO I2's right input
  MUL r6, r3, r4                     I2: MUL -> out         ; fires when BOTH inputs have arrived
```
I0 and I1 are independent → they run the **same cycle** on a grid of ALUs; I2 fires when both land.
The hardware is a **tile grid** of small ALUs plus an **operand network** that ferries results
directly between tiles. Parallelism was *named by the compiler*, not hunted at runtime.

## 3. The ISA sketch (finalise in Phase 0)
- **Block header:** #instructions, #block-outputs, #inputs, predicate map.
- **Instruction encoding:** opcode + **target field(s)** (which slot(s) consume the result) + optional predicate.
- **Block commit semantics:** a block's stores/outputs become visible only on atomic commit.
- **Inter-block:** block-address sequencing + a small set of live values passed between blocks.
- Keep it **minimal and readable** — the novelty is an *open, one-person-comprehensible* ISA, not a rich one.

## 4. The dynamic-dataflow extension (what makes it serve trees)
Base EDGE is *static*. Tree inference needs **data-dependent control** (which child you visit depends
on a feature comparison) and **massive independent parallelism** (thousands of trees, each traversed
independently). Add three small mechanisms — this is the **extension, not a pivot**:

1. **Data-dependent next-block dispatch** — the next block to execute is chosen by a comparison
   result (`feature[i] < threshold ?`), i.e. tree branching becomes block routing.
2. **Monsoon-style token tagging** — tag operands with a context id so many independent tree
   traversals (and many input rows) run concurrently on one fabric without interfering. *(Basis:
   tagged-token dataflow, Arvind/Monsoon, ISCA 1990.)*
3. **On-chip indexed feature-load** — a lightweight gather of `feature[i]` from an on-chip feature
   vector (the working set is small — it stays on-chip, which is the whole reason EDGE can win here).

*(WaveScalar, MICRO 2003, is the proof-of-concept that dataflow handles general control cleanly —
cite as the design precedent.)*

## 5. How a tree ensemble maps onto the fabric
- Each **tree node** = a compare-and-route block: load `feature[i]`, compare to `threshold`, dispatch
  to the left/right child block; leaves emit a partial score.
- The **ensemble** = thousands of independent traversals, tagged and interleaved across the tile grid
  → this is the parallelism GPUs *can't* exploit (different rows diverge → warp stalls) but a
  tagged-token dataflow fabric *can*.
- Final score = sum of leaf contributions (a reduction).

## 6. Why it can win (and when it can't)
- **Wins when:** batch-1 / low-latency serving, deep-ish ensembles, working set on-chip → the
  bottleneck is *branch divergence + ILP*, which EDGE eats. `[verified rationale]`
- **Does NOT win when:** the working set spills to memory, or the workload is really GEMM/bandwidth
  bound (i.e. neural nets). Then EDGE's ILP advantage is irrelevant. **The spike (D-008) exists to
  confirm we're in the winning regime before we build.** `[verified]`

## 7. Phased build plan (each phase ships something measurable)
- **Phase 0 — Contract + pre-registered claim (days).** Freeze the ISA (§3) and the *one* metric +
  baseline you'll report, *before* building. Kills the "fuzzy result" failure mode.
- **Phase 1 — Substrate + hand-written blocks (weeks). ← early win, no compiler.** Cycle-accurate
  model of a small ALU grid + operand network + block sequencer. **Hand-write** EDGE blocks for a
  small tree traversal; show sustained instructions/cycle > an equal-resource scalar RISC-V baseline.
  If no ILP gain here, stop — cheaply.
- **Phase 2 — Dynamic extension + minimal compiler (the crux, the novelty).** Implement §4; write a
  compiler that ingests a **real XGBoost model dump** and emits Grove blocks. Scope the input to
  "a tree ensemble," *not* a general language.
- **Phase 3 — End-to-end measurement.** Grove vs scalar RISC-V (and a GPU-sim baseline) on a real
  ensemble; report the pre-registered claim with effect sizes; make it reproducible with one command.
- **Phase 4 — RTL flex (stretch).** Lift the hot datapath to Verilog/Verilator for a "synthesizable
  RTL" credential.

## 8. Baseline & measurement (the differentiator)
- **Baseline:** an *equal-resource* scalar RISC-V model (same #ALUs / issue budget), running the same
  ensemble. Normalise resources so the comparison is honest (see OQ-4).
- **Metrics:** sustained functional-unit utilisation / achievable-parallel-ops-per-cycle; batch-1
  latency; working-set footprint (must fit on-chip); reproducibility.
- Spend the *majority* of effort here and in the compiler — **not** on the RTL.

## 9. Parking lot (explicitly out of scope for v1)
Graphics; deep-learning / LLM training; multi-core scaling; a general-purpose compiler / LLVM
backend; real FPGA/ASIC; the D-011 AI-bottleneck project.
