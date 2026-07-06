# Grove — Master Handoff

**Read this first.** It orients any agent (human or AI) picking up the project.
For decisions see `decisions.md`; for the design see `architecture.md`; for research evidence see
`research/`.

---

## 1. One-line mission
Build a from-scratch **EDGE dataflow processor** (ISA + compiler + cycle-accurate sim, $0) and prove
it beats a GPU at **control-bound tree-ensemble (XGBoost) inference** — a real, large production-ML
use case.

## 2. Current state — PRE-BUILD, one gate open
Nothing is built yet. Direction is locked (D-004…D-007) **conditional on a de-risk spike (D-008)**.
The spike is cheap ($0, days) and must run **before** any simulator work.
**Update (2026-07-06):** the spike is now **designed & pre-registered** (D-012, `spike-prereg.md`) —
OQ-1/OQ-2/OQ-4 resolved. **coderAS Stage A (implement the cost model to that spec) is unblocked.**

## 3. What happens next (ordered)
| # | Task | Owner | Blocking? | Status |
|---|------|-------|-----------|--------|
| 1 | Design + pre-register the cost-model spike | **experimentAS** (`handoffs/experimentAS-handoff.md`) | — | ✅ done → `spike-prereg.md`, D-012 |
| 2 | Implement the cost model to that spec | **coderAS** (`handoffs/coderAS-handoff.md`, Stage A) | after #1 | ▶ unblocked — build to `spike-prereg.md` §3–§9 |
| 3 | Run + analyse → **GO / NO-GO** decision | experimentAS | after #2 | pending |
| 4 | *If GO:* phased EDGE sim build | coderAS (Stage B) | after #3 = GO | pending |
| 4' | *If NO-GO:* drop ML framing, pursue general-purpose EDGE | discussAS re-frame | after #3 = NO-GO | pending |

**Do not start Stage B (the simulator) until the spike returns GO.** That is the whole point of the gate.

## 4. The non-negotiables
- **$0** — free tools only (Python now; Verilator/Icarus later). No paid hardware.
- **Simulation-first** — cycle-accurate software model is the primary artifact.
- **Honest scope** — tree-ensemble *inference*, **not** DL/LLM *training*. Don't let it drift.
- **The compiler + measurement rig is the differentiator, not the RTL** — budget effort accordingly.
- **Keep `decisions.md` and this file in sync** as work proceeds (append ADRs; update the status table).

## 5. The novelty claim (state it honestly)
Not "we invented EDGE" (TRIPS/E2 did). The contribution is **the first open, minimal, readable,
reproducible full-stack EDGE implementation + a measured control-bound result**, plus our
micro-innovations and the dynamic-dataflow tree extension. Be ready for the reviewer critique in D-010.

## 6. Risk ledger (live)
See `decisions.md` §"Risk & assumption ledger". Top open risk: **R7** — is the GBDT win magnitude
actually good? → the spike retires it.

## 7. Pointers
- Design: `architecture.md`
- Decisions/ADRs: `decisions.md`
- **Spike pre-registration (the D-008 gate design): `spike-prereg.md`** ← build Stage A to this
- Spike brief: `handoffs/experimentAS-handoff.md`
- Build brief: `handoffs/coderAS-handoff.md`
- Evidence: `research/landscape.md`, `research/usecase.md`

## 8. Provenance
Chosen via a discussAS strategy engagement + two researchAS citation-backed sweeps (2026-07-05).
Claims are tagged `[verified]`/`[believed]`/`[guess]`; fabricated citations were caught and discarded.
