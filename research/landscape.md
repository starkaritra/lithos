# Computer-Architecture Novelty Landscape (CANL)
### Prior-art & white-space sweep for a solo, $0, simulation-first "build an architecture from the ground up" project

**Date of sweep:** 2026-07-05
**Acronym:** CANL (Computer Architecture Novelty Landscape)
**Prepared for:** a solo builder with strong CS fundamentals, $0 budget, simulation-only (Verilator / Icarus / cycle-accurate C++ sims / LLVM / open ISAs), months of runway, wanting (1) genuine novelty and (2) a strong research/engineering portfolio signal.

> **Plain-language promise of this doc:** every project or paper named here was checked. `[verified]` = I opened the repo (via the GitHub API) or the paper page and confirmed it exists and says what I claim. `[unverified]` = it came up in search but I did **not** open the primary source — treat with caution. `[inference]` = my judgment, not a cited fact. Saturation labels (crowded / moderate / open) are **my inference** from the evidence unless stated otherwise.

---

## 0. The one finding that reshapes the whole decision

**Bottom line: the builder's literal dream — "a CPU with ISA + compiler, plus a connected GPU with a CUDA-like model and tensor cores" — is no longer open white space. An open-source project already does essentially all of it, and is actively filling the last gap (tensor cores) as of Dec 2025.**

- **Vortex** is an open, synthesizable RISC-V **GPGPU** (SIMT — Single Instruction Multiple Threads, the GPU execution model) with a full **LLVM-based compiler**, **OpenCL and CUDA** front-ends, a cycle-level simulator, driver, and runtime. It runs on FPGAs and in simulation. `[verified]` repo `vortexgpgpu/vortex`, ~2,135 stars, last pushed 2026-07-04; `[verified]` paper *"Vortex: OpenCL Compatible RISC-V GPGPU"* (arXiv 2002.12151).
- **Tensor cores** — the "CUDA tensor core" piece the builder is drawn to — are now open too: *"Ten-Four: An Open-Source Fused Dot Product Unit for Mixed-Precision GPGPU Tensor Cores"* (arXiv 2512.00053, Dec 2025) adds a mixed-precision **MMA** (matrix-multiply-accumulate) unit to Vortex covering FP16/BF16/INT8/FP8/INT4 with FP32 accumulation. `[verified]` (paper page opened).
- **Gemmini** is an open full-stack **systolic-array** (grid of multiply-accumulate cells) matmul accelerator with a compiler path and SoC integration. `[verified]` `ucb-bar/gemmini`, ~1,381 stars.

**Implication (`[inference]`):** re-implementing "a GPU + CUDA-like language + tensor cores from scratch" is now a *learning/portfolio* project, **not** a novelty contribution. Vortex + Ten-Four + Gemmini have taken that ground and are maintained by funded labs. Chasing it head-on means competing with well-resourced teams on their turf. **Novelty must come from a different axis** — a co-designed ISA property, a paradigm that has *no* open full-stack version, or an open, measured reproduction of a proprietary idea. The rest of this report finds those axes.

> Note on citation hygiene: the search tooling repeatedly fabricated plausible-looking GitHub URLs and one wrong arXiv ID (2109.13507 is a physics paper, **not** the Vortex MICRO paper). Everything below was re-checked against primary sources; fabricated links were discarded.

---

## A. Saturation map

| Sub-domain | Representative prior art (verified unless noted) | Saturation | Why (`[inference]` unless cited) |
|---|---|---|---|
| **From-scratch CPU + custom ISA + emulator/RTL** (hobby + academic) | `olofk/serv` (bit-serial RISC-V, ~1,826★) `[verified]`; `stnolting/neorv32` (~2,169★) `[verified]`; `chipsalliance/rocket-chip` (~3,809★) `[verified]`; nand2tetris; hundreds of RISC-V cores | **Crowded (saturated)** | RISC-V made "design an ISA + core + emulator" a solved teaching exercise. Thousands of cores exist; a new plain scalar core is a portfolio piece, not novelty. |
| **Open GPU / SIMT / GPGPU from scratch** | `adam-maj/tiny-gpu` (~12,683★, educational) `[verified]`; `vortexgpgpu/vortex` (~2,135★, research-grade, active) `[verified]`; `VerticalResearchGroup/miaow` (AMD Southern Islands, ~1,386★, older) `[verified]` | **Moderate → crowding fast** | Educational niche is saturated (tiny-gpu). Research-grade is dominated by Vortex, which is funded and actively extended. Room left is in *microarchitectural policy studies*, not "build a GPU." |
| **CUDA-like model / kernel language / GPU compiler from scratch** | Vortex (LLVM + OpenCL + CUDA) `[verified]`; `pocl/pocl` (portable OpenCL on LLVM, ~1,074★) `[verified]`; tinygrad (software); Halide/TVM/MLIR ecosystems | **Moderate** | The *stack pattern* (DSL → LLVM/MLIR → device) is well-trodden. A brand-new CUDA clone is derivative unless it carries a genuinely new programming-model idea. |
| **Tensor cores / systolic arrays / matmul accelerators (dense)** | `ucb-bar/gemmini` (~1,381★) `[verified]`; `nvdla/hw` (NVIDIA NVDLA RTL, ~2,111★, last push 2022 — stale) `[verified]`; Ten-Four MMA unit (arXiv 2512.00053) `[verified]` | **Moderate → closing** | Dense matmul acceleration in the open is increasingly covered end-to-end. Building "another systolic array" is derivative. |
| **Sparse tensor cores (2:4 structured etc.)** | NVIDIA Ampere 2:4 is **proprietary silicon**; academic designs (S2TA HPCA'22, SCNN, Eyeriss-class) rarely release full RTL `[unverified — secondary]` | **Open** | No open, full-stack (prune → compiler → sparse-MMA RTL → *measured* end-to-end) 2:4 accelerator appears to exist. **But the builder explicitly rejected sparsity anchoring — listed only for completeness.** |
| **CGRA / spatial reconfigurable arrays** | `ecolab-nus/morpher` (~84★, active) `[verified]`; `pnnl/OpenCGRA` (~175★, active) `[verified]`; CGRA-ME (Toronto) `[unverified — secondary]` | **Moderate** | Academic-grade generators + compilers exist and are maintained, but they are heavy and hard to run. A *minimal, understandable, measured* end-to-end CGRA is still under-served. |
| **In-/near-memory compute (PIM/CIM)** | `CMU-SAFARI/ramulator2` (~584★) `[verified]`; `UVA-LavaLab/PIMeval-PIMbench` (~51★) `[verified]`; PIMSIM-NN (arXiv 2402.18089) `[unverified]`; CNM/CIM landscape survey (arXiv 2401.14428) `[verified]` | **Moderate–crowded (sim), open (RTL)** | Simulators are plentiful; transparent full-RTL PIM you can synthesize for free is rarer, but the space is academically dense and hardware-modeling-heavy for a solo. |
| **Stochastic / approximate computing** | Approximate Computing Survey Part I (arXiv 2307.11124) `[verified]`; Part II (arXiv 2307.11128) `[unverified]` | **Open but niche** | Many surveys, fewer *open full-stack* (ISA+compiler+sim+measured-error) demos. Niche → lower portfolio signal outside the sub-community. |
| **Capability / memory-safe ISAs (CHERI-style)** | `riscv/riscv-cheri` (~114★) `[verified]`; CTSRD-CHERI org (Cambridge/SRI); ARM **Morello** (commercial CHERI AArch64 prototype) `[verified — general/well-documented]` | **Academic-strong, open-hobby OPEN** | The concept is heavily published and funded, but a *minimal, transparent, solo-reproducible* capability core + compiler + measured overhead is under-served in the hobby/open space. |
| **Formally verified CPU + compiler stack** | `riscv/sail-riscv` (ISA formal model, ~731★) `[verified]`; CakeML; Kami (Coq HW); CompCert `[unverified — secondary]` | **Academic-strong, solo OPEN (hard)** | End-to-end verified stacks are elite research. A *tiny* verified ISA→compiler→core with an actual machine-checked proof is rare from a solo and a very strong signal — but very hard. |
| **Transport-triggered architecture (TTA)** | `cpc/openasip` (formerly TCE, ~188★, active) `[verified]` | **Moderate (mature tool exists)** | OpenASIP is a mature, maintained TTA co-design toolchain. Another TTA core is derivative; a *novel use* of TTA co-design could work. |
| **EDGE / dataflow (TRIPS/WaveScalar) block-atomic** | TRIPS / WaveScalar are **papers + research prototypes**; **no open full-stack RTL+compiler** `[unverified — secondary, consistent across sources]` | **Open** | Explicit-Data-Graph-Execution has essentially *no* open, minimal, measured full stack. Highest-novelty axis found — see White-space W1. |
| **Async / clockless logic** | Manchester **Amulet** (async ARM, historical) `[unverified — secondary]`; open async CPUs are rare | **Open but $0-hostile** | Genuinely under-built in open source, but async needs special design flows/timing analysis that free synchronous sim tools don't support well → high risk for a solo. |
| **Bit-serial designs** | `olofk/serv` (the canonical bit-serial RISC-V) `[verified]` | **Crowded (the flagship exists)** | SERV already owns "smallest bit-serial RISC-V." Derivative unless paired with a new angle. |
| **Novel number systems (posit / unum / LNS)** | `stillwater-sc/universal` (posit + many number systems, software, ~498★) `[verified]`; hardware units scattered `[unverified]` | **Open (hardware full-stack)** | Software libraries are mature; an open *full-stack* (ISA extension + compiler + sim + measured accuracy/area/energy vs IEEE-754) posit or LNS core is relatively open. |
| **Deterministic / time-predictable ISA (PRET)** | `pretis/flexpret` (~62★, last push 2024-11) `[verified]`; FlexPRET RTAS'14 (UC Berkeley, E. Lee) `[unverified — secondary]` | **Open (niche, low activity)** | A real but small community. Timing-as-a-first-class-contract, with a *modern open compiler*, is under-served. |
| **Security / side-channel-resistant microarch** | Large paper literature (Spectre/Meltdown mitigations, secure caches); few clean open reproducible full-stack teaching/eval artifacts `[unverified — secondary]` | **Open (reproducibility gap)** | Concepts are hot and published, but an open cycle-accurate core + reproducible attack+defense *measurement harness* a solo can run for $0 is under-served and high-signal. |
| **VLIW** | Many DSP VLIW cores; well-understood | **Crowded** | Mature, well-understood; low novelty. |

---

## B. White-space findings (the 5–8 most promising directions)

Each entry: **the gap → evidence it's under-explored → minimal novel contribution → honest novelty caveat.** These are deliberately spread across the space (dataflow, safety, timing, security, spatial, number systems) — **not** anchored on ML sparsity, per the builder's instruction.

### W1 — Open, minimal, *measured* EDGE / block-atomic dataflow full stack (ISA + compiler + cycle sim)
- **Gap:** EDGE (Explicit Data Graph Execution — instructions grouped into atomically-executed blocks with dataflow inside) was proven in TRIPS/WaveScalar research but has **no open, minimal, full-stack (ISA→compiler→cycle-accurate sim), end-to-end-measured** implementation.
- **Evidence it's under-explored:** consistent across sources that TRIPS released papers/artifacts but **no open RTL/compiler**; no maintained open EDGE core exists (contrast with the dozens of open RISC-V cores). `[unverified — secondary, but corroborated]`
- **Minimal novel contribution:** a tiny EDGE micro-ISA + a simple block-forming compiler pass + a cycle-accurate C++/Verilator sim, with a **measured ILP (instruction-level parallelism) comparison** against a scalar RISC-V baseline on small kernels. An honest open reproduction of a proprietary/dead research idea *is itself a contribution*.
- **Caveat:** EDGE compilers (block formation, register/operand routing) are genuinely hard; the *concept* is well-published (TRIPS papers), so novelty is in the **open, minimal, reproducible, measured** artifact, not the idea.

### W2 — Capability / memory-safe minimal ISA ("CHERI-lite") as a transparent, measured solo full stack
- **Gap:** CHERI is heavily funded and published, and ARM Morello exists `[verified — general]`, but there is **no minimal, transparent, solo-reproducible** capability core + compiler + measured-overhead + demonstrated-exploits-blocked artifact aimed at a builder/engineer audience.
- **Evidence:** `riscv/riscv-cheri` is a **spec** repo (~114★) `[verified]`; the real implementations are large academic/industrial stacks (CTSRD-CHERI, Morello). The "understandable minimal version" niche is empty. `[inference]`
- **Minimal novel contribution:** a small capability-tagged ISA (bounded, unforgeable pointers), a compiler pass that enforces bounds, a sim, and a **measurement**: overhead (%) vs a baseline core, plus a suite of memory-safety exploits (buffer overflow, use-after-free) shown to be *blocked*.
- **Caveat:** the security concept and results are well-established by the CHERI project; you are not inventing capabilities. The contribution is the **minimal, reproducible, measured full stack** + clear teaching value. Do not over-claim novelty of the mechanism.

### W3 — "Timing-as-a-contract" deterministic ISA + modern open compiler (PRET, revisited)
- **Gap:** PRET/FlexPRET showed time-predictable hardware, but the community is small and `pretis/flexpret` shows **low recent activity** (last push 2024-11, ~62★) `[verified]`. A modern take where **worst-case timing is a first-class language/ISA contract** (e.g., timing "types" the compiler checks) is under-served.
- **Evidence:** FlexPRET is essentially the flagship and it's quiet; no vibrant open ecosystem around timing-as-a-type. `[inference]`
- **Minimal novel contribution:** a small ISA with deterministic-latency instructions + a compiler that carries and checks per-block timing bounds + a sim that *verifies* measured cycle counts match the declared bounds. Measurable claim: "every annotated region's measured WCET equals its statically declared bound across N benchmarks."
- **Caveat:** PRET is well-published (RTAS'14 etc.); the fresh angle is the **language/compiler contract + measured guarantee**, not time-predictability per se.

### W4 — Open, reproducible microarchitectural **security evaluation harness** (attack + defense, measured)
- **Gap:** transient-execution/side-channel attacks and mitigations are hugely published, but an **open cycle-accurate core + a reproducible attack (cache-timing / Spectre-class) + a toggleable defense + measured leakage** that a solo can run for $0 is under-served.
- **Evidence:** the literature is enormous but reproducible open full-stack teaching/eval artifacts are scarce (`[unverified — secondary]`; corroborated by the general reproducibility gap in the field).
- **Minimal novel contribution:** a small out-of-order or speculative sim core + a working covert/side channel + a mitigation you can switch on/off + a **quantified leakage-rate (bits/s) with vs. without** the defense.
- **Caveat:** individual attacks/defenses are published; novelty is the **open reproducible measurement harness** and any new mitigation micro-tweak you measure. Very high signal for security/architecture roles.

### W5 — Minimal, understandable, *measured* CGRA / spatial-dataflow full stack with a novel mapping heuristic
- **Gap:** Morpher (`[verified]`), OpenCGRA (`[verified]`), CGRA-ME exist but are **heavy and hard to run**; there's no "minimal, readable, one-person-can-understand-it" end-to-end CGRA with a clean measured story.
- **Evidence:** the existing tools are research frameworks with steep setup; accessibility is the gap. `[inference]`
- **Minimal novel contribution:** a small CGRA sim + a simple compiler that maps loop kernels onto the array + a **novel mapping/scheduling heuristic measured** (mapping quality / utilization / II — initiation interval) against a baseline greedy mapper.
- **Caveat:** CGRA compilers and mapping algorithms are an established field; you must bring a *measured* heuristic improvement or a novel array/ISA twist, not just a re-implementation.

### W6 — Open full-stack **posit (or LNS) co-designed core**: ISA extension + compiler + measured accuracy/area/energy Pareto
- **Gap:** posit software (`stillwater-sc/universal`, ~498★) `[verified]` is mature, but an **open full-stack** where a posit/LNS number system is wired through *ISA → compiler → cycle sim* and **measured** (accuracy vs IEEE-754, cycle/area cost) is relatively open.
- **Evidence:** hardware posit units are scattered and mostly standalone arithmetic blocks, not full ISA+compiler+measured stacks. `[unverified]`
- **Minimal novel contribution:** add a posit/LNS datatype to a small ISA, teach a compiler to emit it, and produce a **measured accuracy-vs-cost Pareto** on real kernels (dot products, iterative solvers) vs IEEE-754.
- **Caveat:** posits are well-studied and somewhat controversial; the contribution is the **open full-stack + honest measurement**, and you must avoid over-claiming posit superiority (report where it loses too).

### W7 — Open, reproducible **GPGPU microarchitecture policy study** (divergence / warp scheduling) on a small SIMT sim
- **Gap:** now that Vortex + Ten-Four cover "build a GPU + tensor cores," the open white space moved *up a level* to **reproducible micro-policy studies** — e.g., a novel branch-divergence reconvergence or warp-scheduling mechanism, measured.
- **Evidence:** Vortex/tiny-gpu give the substrate; GPGPU-Sim/Accel-Sim exist for policy studies but a fresh, small, reproducible mechanism comparison is still publishable. `[inference]`
- **Minimal novel contribution:** implement 2–3 divergence/scheduling policies (incl. one of your own) in a small SIMT sim and **measure** SIMD-lane utilization / cycles on divergent kernels.
- **Caveat:** GPGPU scheduling/divergence is a mature research area; you need a genuinely new mechanism or a surprising measured result, else it's incremental.

### W8 — Open, minimal formally-*checked* ISA→compiler slice (tiny verified stack)
- **Gap:** end-to-end verified stacks (CakeML, Kami, sail) are elite research; a **tiny, solo, machine-checked** "this compiler's output preserves this ISA's semantics" slice is rare and a very strong signal.
- **Evidence:** `riscv/sail-riscv` (~731★) `[verified]` gives formal ISA semantics to build on; but solo end-to-end verified artifacts are scarce. `[inference]`
- **Minimal novel contribution:** a minimal ISA in a proof assistant (or via Sail) + a tiny verified compiler pass + a machine-checked correctness theorem for a subset.
- **Caveat:** hardest on this list; formal-methods depth required. Novelty is the *scope-minimal reproducible* proof, not the technique.

---

## C. Feasibility rank (solo / $0 / simulation, months)

Scores 1–5 (5 = best). **Novelty** = how much genuine new ground. **Solo-feasibility** = deliverable in months alone with free tools. **Signal** = portfolio/paper value for advanced eng/research roles.

| # | Direction | Novelty | Solo-feasibility | Signal | Single biggest risk |
|---|---|:--:|:--:|:--:|---|
| **W1** | Open minimal EDGE/dataflow full stack (measured) | **5** | **3** | **5** | The block-forming **compiler** is the hard part — easy to under-scope and stall. |
| **W2** | Capability / memory-safe minimal ISA (measured) | **3** | **4** | **5** | Over-claiming novelty vs CHERI; keep it *minimal + measured* or it looks like a CHERI clone. |
| **W3** | Timing-as-a-contract deterministic ISA + compiler | **4** | **4** | **4** | Proving measured WCET == declared bound rigorously (must eliminate sim non-determinism). |
| **W4** | Open microarch security eval harness (attack+defense) | **4** | **4** | **5** | Getting a *real, reproducible* side-channel to leak in a simple sim (not a toy). |
| **W5** | Minimal measured CGRA + novel mapping heuristic | **3** | **3** | **4** | Mapping/compiler complexity; needing a *measured* win over a baseline mapper. |
| **W6** | Open full-stack posit/LNS core (measured Pareto) | **3** | **4** | **3** | Lower prestige/niche; must report honest losses, not just wins. |
| **W7** | GPGPU divergence/scheduling policy study (measured) | **3** | **4** | **4** | Mature field — risk of being "incremental" without a novel mechanism. |
| **W8** | Tiny formally-checked ISA→compiler slice | **4** | **2** | **5** | Formal-methods learning curve can eat the whole timeline. |

---

## D. Top 3 recommendations

These three best satisfy *novel + solo-feasible in sim + high signal*, and — importantly — they **keep the builder's core dream** (design an ISA + a compiler + a microarchitecture, measured end-to-end) while pointing it at ground that is actually open, instead of the now-occupied "GPU + CUDA + tensor cores."

### #1 — Capability-safe minimal ISA + compiler + measured safety ("CHERI, but the version one person can read")
- **One-line pitch:** *"An open, minimal, transparent memory-safe CPU: a capability ISA + a compiler that enforces pointer bounds + a sim, with a measured overhead number and a suite of real exploits it provably blocks."*
- **Concrete measurable claim the finished project can make:** *"On N kernels, capability enforcement adds X% cycle overhead and Y% code-size overhead while blocking 100% of a documented buffer-overflow / use-after-free exploit suite that the baseline core executes successfully."*
- **Why this one:** memory safety is the single hottest topic in systems security right now; the *concept* is proven (so you're de-risked on "does it work?") and your novelty is the **minimal, reproducible, measured full stack** — the exact artifact that is missing in the open/hobby world. Strong hiring signal across security + architecture + compilers.

### #2 — Open, minimal, measured EDGE / block-atomic dataflow full stack
- **One-line pitch:** *"The first open, minimal, end-to-end EDGE machine — a block-atomic dataflow ISA + a block-forming compiler + a cycle-accurate sim — reproducing a dead research architecture (TRIPS-style) and measuring the ILP it buys."*
- **Concrete measurable claim:** *"Our open EDGE core extracts Z× more instruction-level parallelism than a scalar RISC-V baseline on M dataflow-heavy kernels, in a fully open, reproducible sim."*
- **Why this one:** highest genuine novelty on the board (no open full-stack exists), and it is *maximally aligned* with "build an architecture + ISA + compiler from the ground up." It's the most paper-worthy. **Trade-off:** the compiler is the risk — scope the block-formation aggressively (start with straight-line hyperblocks) so you finish.

### #3 — Open, reproducible microarchitectural security evaluation harness (attack + toggleable defense, measured leakage)
- **One-line pitch:** *"A $0, open, cycle-accurate core where you can turn a Spectre-class side channel — and its mitigation — on and off, and watch the measured leakage rate change."*
- **Concrete measurable claim:** *"With the defense off, the channel leaks at R bits/second; with our mitigation on, leakage drops below the noise floor at a measured performance cost of C%."*
- **Why this one:** best novelty-feasibility-signal balance for someone with strong CS fundamentals; reproducible security artifacts are scarce and highly valued; every number is measurable in a free sim. **Trade-off:** you must get a *real* channel to leak, not a contrived one.

**Cross-cutting advice (`[inference]`):** for all three, the **compiler + measurement rig is the differentiator**, not the RTL. Free tooling is fully sufficient: Verilator/Icarus for RTL sim, LLVM/MLIR for the compiler, Python for the measurement harness and plots. Pre-commit to a single crisp measurable claim *before* building, so the project has a paper-shaped spine.

---

## E. Citations

**Verified — repository opened via GitHub API (name / stars / last push confirmed on 2026-07-05):**
- Vortex GPGPU — https://github.com/vortexgpgpu/vortex (~2,135★, pushed 2026-07-04) `[verified]`
- tiny-gpu — https://github.com/adam-maj/tiny-gpu (~12,683★) `[verified]` *(search tools mis-attributed this to "naotoishii" — wrong)*
- Gemmini — https://github.com/ucb-bar/gemmini (~1,381★) `[verified]`
- MIAOW GPU — https://github.com/VerticalResearchGroup/miaow (~1,386★) `[verified]`
- OpenASIP (formerly TCE, TTA co-design) — https://github.com/cpc/openasip (~188★) `[verified]`
- Sail RISC-V formal model — https://github.com/riscv/sail-riscv (~731★) `[verified]`
- RISC-V CHERI spec — https://github.com/riscv/riscv-cheri (~114★) `[verified]`
- FlexPRET (precision-timed processor) — https://github.com/pretis/flexpret (~62★, pushed 2024-11) `[verified]`
- Morpher (CGRA framework) — https://github.com/ecolab-nus/morpher (~84★) `[verified]`
- OpenCGRA — https://github.com/pnnl/OpenCGRA (~175★) `[verified]`
- PIMeval / PIMbench — https://github.com/UVA-LavaLab/PIMeval-PIMbench (~51★) `[verified]`
- Ramulator 2 (memory-system sim) — https://github.com/CMU-SAFARI/ramulator2 (~584★) `[verified]`
- Stillwater Universal (posit / number systems) — https://github.com/stillwater-sc/universal (~498★) `[verified]`
- NVDLA RTL — https://github.com/nvdla/hw (~2,111★, pushed 2022 — stale) `[verified]`
- OpenPiton — https://github.com/PrincetonUniversity/openpiton (~800★) `[verified]`
- Rocket Chip — https://github.com/chipsalliance/rocket-chip (~3,809★) `[verified]`
- SERV (bit-serial RISC-V) — https://github.com/olofk/serv (~1,826★) `[verified]`
- NEORV32 — https://github.com/stnolting/neorv32 (~2,169★) `[verified]`
- POCL (portable OpenCL) — https://github.com/pocl/pocl (~1,074★) `[verified]`
- Multi2Sim — https://github.com/Multi2Sim/multi2sim (~137★, pushed 2019 — stale) `[verified]`

**Verified — paper/landing page opened (title confirmed):**
- "Vortex: OpenCL Compatible RISC-V GPGPU" — https://arxiv.org/abs/2002.12151 `[verified]`
- "Ten-Four: An Open-Source Fused Dot Product Unit for Mixed-Precision GPGPU Tensor Cores" — https://arxiv.org/abs/2512.00053 (Dec 2025) `[verified]`
- "The Landscape of Compute-near-memory and Compute-in-memory: A Research and Commercial Overview" — https://arxiv.org/abs/2401.14428 `[verified]`
- "Approximate Computing Survey, Part I: Terminology and Software & Hardware Approximation Techniques" — https://arxiv.org/abs/2307.11124 `[verified]`
- "Accelerating Edge AI with Morpher: An Integrated Design, Compilation and Simulation Framework for CGRAs" — https://arxiv.org/abs/2309.06127 `[verified]`

**Unverified — surfaced by search, primary source NOT opened (treat with caution):**
- Approximate Computing Survey, Part II — https://arxiv.org/abs/2307.11128 `[unverified]`
- PIMSIM-NN — https://arxiv.org/abs/2402.18089 `[unverified]`
- FlexPRET original paper (Zimmer, Broman, Shaver, Lee, RTAS 2014, UC Berkeley) `[unverified — secondary]`
- Gemmini paper (DAC 2021, "full-stack integration") `[unverified — secondary]`
- CGRA-ME (University of Toronto) `[unverified — secondary]`
- ARM Morello / Digital Security by Design — `[verified — general/well-documented]` but primary spec not opened here
- CTSRD-CHERI org (Cambridge/SRI) — https://github.com/CTSRD-CHERI `[unverified — org not opened this session]`
- TRIPS/WaveScalar have no open full-stack RTL+compiler — `[unverified — secondary, corroborated across multiple sources]`
- Amulet (Manchester async ARM) — `[unverified — secondary]`

**Known-bad (flagged so they are never reused):**
- arXiv 2109.13507 is *"Tunable Gyromagnetic Augmentation of Nuclear Spins in Diamond"* (a physics paper), **NOT** the Vortex MICRO 2021 paper — the search tool hallucinated this mapping. Cite arXiv 2002.12151 + the repo instead.
- Multiple fabricated GitHub URLs from search (e.g., "pzhaonet/S2TA", "Srammy/scal-sysarr", "asi1406/PositFPGA", "naotoishii/tiny-gpu") did **not** verify and were discarded.

---

## Research quality checks (uncertainties & how to raise confidence)

- **Confidence: Medium-High overall.** High on *what exists* (repos verified directly). Medium on *saturation labels* and on "no open full-stack X exists" claims (W1 EDGE, sparse 2:4, security harness) — these are absence-of-evidence arguments; absence is hard to prove. To raise to High: a targeted GitHub/Google-Scholar/Papers-with-Code sweep per direction before committing.
- **Anti-bias check honored:** the survey deliberately spans dataflow, safety, timing, security, spatial, number systems, verification — and does **not** anchor on NN sparsity/pruning/quantization (flagged only for completeness in the saturation map, per the builder's explicit rejection).
- **Biggest decision-relevant fact:** the "GPU + CUDA + tensor cores" dream is now occupied by Vortex + Ten-Four + Gemmini (all verified). Treat it as a learning/portfolio path, not a novelty path.
- **Before building:** for the chosen direction, (1) do one more focused prior-art pass to confirm no close open competitor appeared recently, (2) pre-commit the single measurable claim, (3) scope the compiler ruthlessly — it is the long pole and the differentiator in every top-3 option.
- **Not legal advice:** if any direction (e.g., a capability or timing mechanism) looks patent-worthy, note that public disclosure (GitHub, arXiv, blog) can bar patent rights in some jurisdictions — consult a patent attorney *before* publishing if that matters to you.
