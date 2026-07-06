# Handoff → coderAS: build Grove

Two stages. **Stage A is gated in front of everything else.** Do **not** start Stage B (the
simulator) until experimentAS's spike (Stage A) returns **GO**. Full context: `../handoff.md`,
`../architecture.md`, `../decisions.md`.

Conventions: keep `../decisions.md` and `../handoff.md` **in sync** as you go (append ADRs for any
real choice; update the status table). Everything must be **$0** and **reproducible with one command**.

---

## Stage A — the cost-model spike (do this first)
**Goal:** implement, to experimentAS's pre-registered spec (`experimentAS-handoff.md`), a Python
**analytical cost model** that compares an EDGE-issue model vs an equal-resource scalar RISC-V model
on **real XGBoost inference**, and reports functional-unit utilisation + working-set footprint.

- **Inputs:** a real trained XGBoost model dump (experimentAS specifies the dataset/config, OQ-2).
- **Do:** parse the ensemble; model batch-1 (and small-batch) inference under both engines; count
  achievable parallel node-evals/cycle & sustained FU utilisation; measure per-inference working-set
  bytes; **explicitly model** feature-gather, dynamic-dispatch, tag-match, and load-imbalance overheads.
- **Output:** the metrics + a sensitivity sweep (#trees, depth, batch) in a form experimentAS can
  analyse; one-command reproducible.
- **Tooling:** Python + xgboost + numpy only. No RTL, no paid services.
- **Definition of done:** experimentAS can render a GO/NO-GO from your numbers.

**⛔ Gate:** if the spike is **NO-GO**, stop here — do not build the simulator. discussAS will re-frame
toward a general-purpose EDGE contribution (D-008).

---

## Stage B — the phased EDGE build (ONLY if Stage A = GO)
Follow `architecture.md` §7. Each phase must ship something measurable; scope every phase to finish.

### Phase 0 — Contract + pre-registered claim (days)
- Freeze the **ISA** (`architecture.md` §3): block header, target-form instruction encoding,
  predication, block-commit semantics, inter-block sequencing. Keep it **minimal and readable**.
- Freeze the **one** headline metric + the baseline definition **before** building (OQ-4). Record in `decisions.md`.

### Phase 1 — Substrate + hand-written blocks (early win, NO compiler)
- Build a **cycle-accurate** model of: a small **ALU tile grid**, the **operand network**, and a
  **block sequencer**. Instructions fire when operands arrive; blocks commit atomically.
- **Hand-write** Grove blocks for a small tree traversal.
- **Prove:** sustained instructions/cycle **>** an equal-resource **scalar RISC-V** baseline on the
  same computation. If there's no ILP gain here, **stop and report** — cheaply.

### Phase 2 — Dynamic extension + minimal compiler (the crux + the novelty)
- Implement the extension (`architecture.md` §4): **data-dependent next-block dispatch**,
  **Monsoon-style token tagging** (concurrent independent traversals), **on-chip indexed feature-load**.
- Write a **minimal compiler** that ingests a **real XGBoost model dump** and emits Grove blocks
  (each tree node → a compare-and-route block; ensemble → tagged, interleaved traversals; leaves → a
  score reduction). Scope the front-end to *"a tree ensemble,"* **not** a general language (OQ-3).

### Phase 3 — End-to-end measurement (the deliverable)
- Run Grove vs the scalar RISC-V baseline (**and** a GPU-sim baseline if feasible) on a real ensemble.
- Report the pre-registered claim with **effect sizes + a sensitivity range** (not a single number).
- Confirm the working set stays **on-chip**. Make the whole run **reproducible with one command**.

### Phase 4 — RTL flex (stretch, optional)
- Lift the hot datapath to **Verilog** and simulate with **Verilator** (free) for a
  "synthesizable RTL" credential. Not required for the core result.

---

## Guardrails
- **The compiler + measurement rig is the differentiator — not the RTL.** Spend effort there.
- **Honest scope:** tree-ensemble *inference* only. No DL/LLM training, no graphics, no LLVM, no
  multi-core (those are the parking lot, `architecture.md` §9).
- **Pre-empt D-010:** be ready to justify a *programmable* dataflow core over a fixed-function FPGA
  tree pipeline (generality + open full-stack).
- **Testing:** each phase needs a small, runnable check that demonstrates its claim; keep baselines
  reproducible.
