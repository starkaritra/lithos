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

    // ---- richer fields for the browser visualization (only filled when tracing) -------
    // Operand register indices from the instruction (data-flow view). -1 = unused.
    int rd = -1, ra = -1, rb = -1;
    std::int32_t imm = 0;         // immediate (MOV)
    bool writes_rd = false;       // op produces a destination value (draw rd arrow)
    bool is_mem = false;          // ld/st (memory-block view)
    bool is_store = false;        // st (vs ld)
    // Per-lane detail, indexed by lane [0..WARP_SIZE). Empty when not applicable.
    // lane_addr: word address each active lane touched (memory ops only).
    // lane_val:  value each active lane produced (writes_rd) or stored (is_store).
    // Inactive lanes carry a sentinel; the frontend uses active_mask to decide.
    std::vector<std::int32_t> lane_addr;
    std::vector<std::int32_t> lane_val;
};

}  // namespace simt
