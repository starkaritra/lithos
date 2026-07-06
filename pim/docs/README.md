# Grove Arm C — Beating the Memory Wall, Learned by Building PIM

Welcome to the second half of the course. In [Arm A](../../simt/docs/README.md) you built a mini-GPU
and *measured* that memory-bound kernels stall on the **memory wall** — data movement, not arithmetic,
is the bottleneck. Arm A's answer was to *tolerate* the wall (hide latency, coalesce). This folder
teaches the *radical* answer: **stop moving the data** — put the compute inside the memory — by
building and measuring a **near-memory / Processing-In-Memory (PIM)** model.

Same philosophy as Arm A: **motivate *why* before *how*; build intuition before formalism; make every
claim measurable.** And the same honesty: the headline result here is a *conditional* win, and the
chapters show you exactly why — including the trap we had to design around to avoid fooling ourselves.

---

## Who this is for
Anyone who did Arm A (or knows basic GPU/parallel ideas) and wants to understand near-memory computing
— what it is, when it wins, when it doesn't, and how to measure that honestly. This is the frontier of
where computing is going as the memory wall bites harder (recsys, LLM KV-cache, graph analytics).

## How to read it
In order the first time; each chapter builds on the last. Every chapter follows the same skeleton:
**problem → intuition → mechanism → in our model (`file` refs) → measure it yourself → on real hardware
→ references.**

---

## The learning path

| # | Chapter | The one-sentence payoff |
|---|---------|-------------------------|
| 01 | [Recap: the Memory Wall](01-recap-the-memory-wall.md) | Why data movement — not arithmetic — is the bottleneck, and why *tolerating* it isn't enough. |
| 02 | [Near-Memory Computing](02-near-memory-computing.md) | The radical bet: move the compute to the data instead of the data to the compute. |
| 03 | [The Bank Model & a Fair Fight](03-the-bank-model-and-fairness.md) | How we model banks + a shared bandwidth cap so the GPU-vs-PIM comparison can't be rigged. |
| 04 | [Byte-Accounting & the DMR](04-byte-accounting-and-dmr.md) | The assumption-free metric: *count* the off-chip bytes each machine moves. |
| 05 | [The Banking Factor `k`](05-the-banking-factor-k.md) | The crux: why the real win is `L/k`, not the naive `L` — and why that number decides everything. |
| 06 | [The Crossover & Honest Science](06-the-crossover-and-honest-science.md) | The reduction-ratio sweep, the crossover, and how pre-registration kept this from being a rigged demo. |
| — | [Glossary](glossary.md) | Every term, defined once in plain words. |
| — | [References](references.md) | The real bibliography. |

---

## The headline result (so you know where this goes)
On the canonical recommendation workload (embedding-bag sum-pooling, `d=64, L=40, B=16`), PIM moves
**2.53× fewer off-chip bytes** than a competent GPU — a *gray-zone* result. The win rises to >12× for
high-pooling features but **vanishes (<1.5×) for low-pooling ones**. Verdict (project decision D-019):
**CONDITIONAL-GO** — PIM's data-movement win is real but **banking-limited**. The full reasoning is in
[`../out/conclusion.md`](../out/conclusion.md); the pre-registered design is in
[`../../pim-prereg.md`](../../pim-prereg.md).

> **Honesty, up front.** This is a **byte-accounting model** — the primary metric (bytes moved) is
> `[modelled-exact]` (counted, not estimated); cycles and energy are `[modelled]` and secondary. A
> favorable result licenses a *data-movement* claim, **not** a silicon speedup. Where the model
> simplifies real hardware, the chapters say so (look for **Reality check** boxes).
