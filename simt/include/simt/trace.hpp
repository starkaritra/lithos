#pragma once
// Optional execution trace for the mini-GPU (used by the WASM playground to ANIMATE
// execution). The headless core does not record a trace unless explicitly enabled, so
// the tested engine path and its cycle counts are completely unaffected.
#include <cstdint>
#include <string>
#include <vector>

namespace simt {

// One record per issued warp-instruction, in issue order — enough for the frontend to
// replay execution cycle by cycle (which warp issued, its active lanes, memory cost, and
// whether it stalled/diverged).
struct TraceEvent {
    std::uint64_t cycle = 0;      // the cycle this instruction issued at
    int warp_id = 0;
    int pc = 0;
    std::string op;               // mnemonic (mov/tid/ld/bra/...)
    std::uint32_t active_mask = 0;// bit l set = lane l active on this issue
    std::uint64_t latency = 0;    // issue latency (1 for ALU; memory-dependent for LD/ST)
    int mem_txns = 0;             // memory transactions this issue caused (0 if not a mem op)
    bool diverged = false;        // this BRA split the warp (divergence)
    bool after_stall = false;     // the clock jumped before this issue (a real stall gap)
};

}  // namespace simt
