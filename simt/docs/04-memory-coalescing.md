# 04 — Memory Coalescing

> **Goal:** understand why the *pattern* of memory addresses a warp touches — not just how
> many bytes — can change performance by an order of magnitude. You'll derive our
> "4 transactions vs 32 transactions" result and connect it to real DRAM.

---

## 1. The problem: memory is delivered in chunks, not bytes

Intuitively you might think reading 32 numbers costs "32 reads." It doesn't. DRAM and the
memory bus don't hand you one byte at a time — they deliver a whole **chunk** (a cache line
/ memory segment, e.g. 32 or 128 bytes) per transaction, because opening a row of DRAM and
driving the bus has a large fixed cost. Fetching 4 bytes and fetching 32 bytes from the same
segment cost about the same.

This changes everything about how a *warp* should read memory. A warp has 32 lanes issuing
32 addresses at once. The question that decides performance is:

> **How many distinct memory segments do those 32 addresses fall into?**

Because you pay per **segment (transaction)**, not per lane.

---

## 2. Intuition: the mailroom

You have 32 letters to collect from a wall of PO boxes, where boxes are grouped into
**panels** of 8, and opening a panel is the expensive part.

- **Coalesced:** all 32 letters are in 4 adjacent panels (boxes 0–31). You open **4 panels**,
  grab everything. Fast.
- **Scattered:** your 32 letters are one-per-panel, spread across 32 different panels. You
  open **32 panels**. 8× the work for the *same 32 letters*.

```
   COALESCED (contiguous addresses)          SCATTERED (strided addresses)
   32 lanes → words 0..31                    32 lanes → words 0,8,16,...,248

   seg0[0..7] seg1[8..15] seg2 seg3          seg0 seg1 seg2 ... seg31
   ▓▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓▓    ▓▓▓  ▓▓▓            ▓... ▓... ▓...     ▓...
   └── 4 segments touched ──┘                └──── 32 segments touched ────┘
        = 4 transactions                            = 32 transactions
```

Same number of lanes, same number of *useful* bytes — but the scattered pattern issues **8×
the memory transactions**, and on the memory wall (chapter 01) that is exactly where the
time goes. This is **coalescing**: neighboring threads touching neighboring memory so their
accesses fuse into few transactions.

---

## 3. The mechanism

For each memory instruction, the hardware (and our simulator) looks at the addresses from
all *active* lanes, maps each to its segment (`segment = address / segment_size`), and counts
the **distinct** segments. That count is the number of transactions the instruction costs.

```
   coalescing rule:
       transactions = | { addr_lane / SEGMENT  :  lane is active } |    (set = distinct)

   contiguous:  {0/8, 1/8, ..., 31/8} = {0,1,2,3}                 → 4 transactions
   strided×8:   {0/8, 8/8, 16/8, ...} = {0,1,2,...,31}            → 32 transactions
```

More transactions cost more, in two ways: more bus cycles (bandwidth) and, if uncoalesced
enough, less effective use of every byte fetched (you pay for a whole segment but use one
word of it — **wasted bandwidth**).

---

## 4. In our mini-GPU: coalescing in the memory model

Look at `Core::memory_access` in `src/core.cpp`. It is the coalescing rule, verbatim:

```cpp
std::set<std::int64_t> segments;                 // distinct segments touched
for (int l = 0; l < WARP_SIZE; ++l) {
    if (!w.active[l]) continue;                  // inactive lanes touch nothing
    std::int32_t addr = w.regs[l][in.ra];
    /* ... perform the load/store ... */
    segments.insert(addr / cfg_.segment_words);  // which segment is this address in?
}
const std::uint64_t txns = segments.size();      // COALESCING: pay per segment
stats_.mem_transactions += txns;

// latency = one round trip + a bandwidth penalty for each EXTRA transaction
const std::uint64_t extra = txns > 0 ? txns - 1 : 0;
return cfg_.mem_latency + extra * cfg_.mem_txn_penalty;
```

Two design choices to understand:

- **`std::set`** gives us "distinct segments" for free — inserting the same segment twice
  counts once. That is the fusing of neighboring accesses into one transaction.
- **The latency formula** charges `mem_latency` once (the round trip) plus
  `mem_txn_penalty` for each *additional* transaction. So a coalesced load pays the base
  latency; a scattered load pays a growing bandwidth tax. Coalescing thus shows up in *both*
  the transaction count *and* the cycle count.

`segment_words = 8` means a segment is 8 four-byte words = 32 bytes — a reasonable stand-in
for a small cache line. (Real NVIDIA transactions are commonly 32 or 128 bytes.)

---

## 5. Measure it yourself — and derive the number

The test `test_coalescing` (in `tests/test_simt.cpp`) runs two loads with 32 threads:

**Coalesced** — address = thread id, so lanes touch words 0..31:
```
tid r0
ld  r1, r0      # addr = tid  →  segments {0,1,2,3}  →  4 transactions
halt
```

**Scattered** — address = thread id × 8, so every lane lands in its own segment:
```
tid  r0
mov  r2, 8
imul r3, r0, r2   # addr = tid*8
ld   r1, r3       # segments {0,1,2,...,31}  →  32 transactions
halt
```

Derivation of the transaction counts:
- Coalesced: `{ i/8 : i in 0..31 } = {0,1,2,3}` → **4 transactions.**
- Scattered: `{ (8i)/8 : i in 0..31 } = {0,1,...,31}` → **32 transactions.**

The test asserts exactly these (`== 4` and `== 32`) *and* that the scattered version takes
**more cycles** (`c2.cycles > c1.cycles`) — the bandwidth penalty. That single 8× gap, for
the *same amount of useful data*, is the whole lesson. Run:

```
build\test_simt.exe        # includes test_coalescing
```

> **Try it:** write a kernel with stride 2 (`imul r3, r0, r2` with `r2 = 2`). Predict the
> transaction count before running (hint: `{2i/8 : i in 0..31}`), then check. Watch how the
> transaction count — and the cycle count — climbs smoothly as the stride grows. You are
> plotting a coalescing curve by hand.

---

## 6. On real GPUs

- A warp's 32 lane addresses are inspected by the memory subsystem and grouped into the
  minimum number of **cache-line-sized transactions**. Contiguous, aligned access from
  consecutive threads is the ideal ("perfectly coalesced").
- **This is the #1 practical GPU optimization.** The canonical rule — *"thread `i` should
  access element `i`"* (so consecutive threads hit consecutive addresses) — exists entirely
  to get coalescing. Transposing a matrix naively, or using an array-of-structs layout,
  breaks it and can cost 8–32× bandwidth.
- Coalescing interacts with the **roofline model** (Williams et al., 2009): a kernel is
  either compute-bound or bandwidth-bound. Uncoalesced access pushes you toward the
  bandwidth roof — the memory wall — where adding ALUs doesn't help. *This is precisely the
  regime Arm C (PIM) is built to attack.*

> **Reality check.** Real coalescing also depends on **alignment** (a contiguous but
> mis-aligned access can straddle an extra line) and on cache behavior. Our model captures
> the essential "pay per distinct segment" rule and ignores alignment/caching for clarity.

---

## Check your understanding
1. Two kernels read the same 32 numbers. One issues 4 transactions, the other 32. What is
   different, and why does it matter given the memory wall?
2. In `memory_access`, why is a `std::set` the right data structure for the transaction
   count?
3. Predict the transaction count for a stride-4 access pattern over 32 threads with
   `segment_words = 8`. (Work out `{4i/8 : i = 0..31}`.)

---

## References
- NVIDIA, *CUDA C++ Best Practices Guide* — "Coalesced Access to Global Memory."
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model
  for Multicore Architectures," *Communications of the ACM*, 2009.
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., §2.1–2.3
  (memory hierarchy and DRAM) and §4.4 (GPU memory).
- M. Harris, "How to Access Global Memory Efficiently in CUDA C/C++ Kernels," NVIDIA
  Developer Blog.

→ Previous: [03 — Latency Hiding & Occupancy](03-latency-hiding-occupancy.md) · Next: [05 — Branch Divergence](05-branch-divergence.md)
