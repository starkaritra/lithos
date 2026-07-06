# Glossary (Arm C)

Every term used in the PIM course, in plain words. Terms shared with Arm A (warp, coalescing, roofline,
DRAM/SRAM, etc.) are defined in [`../../simt/docs/glossary.md`](../../simt/docs/glossary.md); this list
covers the near-memory / data-movement vocabulary.

**Memory wall** — the large, growing gap between compute speed and memory speed; processors spend much
of their time waiting for data. The bottleneck this whole arm attacks. (See
[ch. 01](01-recap-the-memory-wall.md).)

**Data movement** — bytes physically transferred across the off-chip link between memory and compute.
The dominant cost for memory-bound kernels, and the thing PIM reduces.

**PIM (Processing-In-Memory) / near-memory computing** — an architecture that places small compute
units inside or beside the memory banks, so simple operations happen where the data lives and only
results cross the link. (See [ch. 02](02-near-memory-computing.md).)

**Bank** — a physically independent slice of memory that can be accessed in parallel with other banks.
In our model each bank owns an address slice and has a tiny compute unit (PCU).

**PCU (PIM compute unit)** — the tiny per-bank arithmetic unit (add / multiply-accumulate / reduce) that
aggregates a bank's local data.

**Placement / `place(i)`** — the function deciding which bank holds row/element `i` (round-robin =
`i mod B`; clustered = contiguous chunks). Determines how many banks a set of accesses touches.

**Bandwidth cap** — the shared off-chip link limit (bytes/cycle) applied equally to both the GPU
baseline and PIM. The fairness spine of the comparison. (See [ch. 03](03-the-bank-model-and-fairness.md).)

**Baseline (coalesced floor)** — the strongest reasonable GPU: it moves each needed byte across the link
exactly once, perfectly coalesced. Using the *floor* (not a strawman) is what makes the comparison fair.

**DMR (Data-Movement Reduction)** — the primary metric: `baseline off-chip bytes ÷ PIM off-chip bytes`.
"×-fewer-bytes." Bytes are *counted*, so it's assumption-free (`[modelled-exact]`). (See
[ch. 04](04-byte-accounting-and-dmr.md).)

**Embedding-bag sum-pooling** — the headline recsys/DLRM kernel: gather `L` scattered rows from a huge
table and sum them into one vector. Memory-bound (mostly gathering), lightly aggregating.

**Pooling factor `L`** — how many rows a bag sums. The aggregation-factor axis of the sweep; `L=1` is
pure gather (no aggregation), large `L` aggregates hard.

**Banking factor `k` = `k(L,B) = B·(1−(1−1/B)^L)`** — the number of *distinct banks* a bag's `L` rows
touch, i.e. the partials PIM must ship. The crux: the honest win is `DMR ≈ L/k`, not the naive `L`.
(See [ch. 05](05-the-banking-factor-k.md).)

**Banking tax** — the tension that *more* banks (`B`) give more parallelism but a *larger* `k` for a
given `L`, hence a *smaller* data-movement win. The binding limiter on Arm C's result.

**Hidden traffic** — the link bytes PIM can't avoid: indices it must fetch and the output it must write.
Charged in full to both machines so the win can't hide unaccounted cost.

**PIM overhead fraction** — the share of PIM's link traffic that is hidden traffic (indices + output)
rather than pre-aggregated partials.

**Aggregation factor / reduction ratio** — how aggressively a kernel boils input down to output
(`output/input`). The sweep axis: pure gather (no reduction) → embedding-bag → full reduction.

**Crossover `L*`** — the smallest pooling factor at which PIM becomes "worth it" (DMR ≥ 1.5×). Below it
PIM doesn't help. Locating it — and where the real workload sits relative to it — is the honesty
deliverable. (See [ch. 06](06-the-crossover-and-honest-science.md).)

**Tautology (the trap)** — "proving" PIM wins by only testing kernels that must aggregate. Defeated by
sweeping the whole spectrum and showing the crossover.

**Pre-registration** — fixing hypotheses, metric, and decision rule *before* seeing data, so the
conclusion can't be bent to fit the numbers. (`../../pim-prereg.md`.)

**CONDITIONAL-GO** — the Arm C verdict (D-019): the data-movement win is real but banking-limited —
present for high-pooling features, absent for low-pooling ones.

**`[modelled-exact]` / `[modelled]`** — honesty tags: `[modelled-exact]` = counted exactly within the
model (the byte DMR); `[modelled]` = analytical estimate (cycles, energy). Neither claims silicon.

→ Back to [index](README.md)
