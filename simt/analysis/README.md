# Arm A — Analysis layer

The Python half of the gem5-pattern split (decision D-016): the C++ core *runs* the machine;
this layer *sweeps and visualizes* it. `sweep.py` drives `build/simt.exe` (via its `--json`
output and timing-model flags) across parameter sweeps and plots the three microarchitecture
effects, turning each into a curve you can see.

## Run
```
# from simt/, after building the C++ core (cmake --build build)
python analysis/sweep.py
```
Outputs to `analysis/out/` (git-ignored — regenerate any time):

| File | What it shows |
|------|----------------|
| `latency_hiding.png` | cycles vs #warps — stays ~flat while a "if not hidden" line grows linearly: **16× the work costs ~7% more time** because extra warps hide memory latency (chapter 03). |
| `coalescing.png` | transactions & cycles vs access stride — contiguous (stride 1) = 4 transactions, scattered (stride 8) = 32, with cycles rising in step (chapter 04). |
| `divergence.png` | cycles & divergent-branch count vs branch threshold — uniform endpoints are ~half the cost of a split warp in the middle (chapter 05). |
| `memory_wall.png` | cycles vs arithmetic-ops-per-access — a straight line `cycles = memory-floor + K` showing ~one memory access costs hundreds of arithmetic ops; the shaded region is the memory-bound "wall" (chapter 09). |
| `sweeps.csv` | the raw numbers behind all three plots. |

## Why a separate layer?
This mirrors how real architecture research is structured (gem5 = C++ core + Python scripts):
the simulator stays a fast, headless, testable engine, while experimentation, sweeping, and
plotting live in Python where they're quick to iterate. It also keeps the C++ core free of
plotting dependencies.

All results are `[modelled]` — a deterministic teaching model, not silicon (see
`../docs/06-cycle-accurate-simulation.md`).
