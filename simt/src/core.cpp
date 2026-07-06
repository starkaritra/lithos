#include "simt/core.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>

namespace simt {

Core::Core(std::vector<Instr> program, GlobalMemory& mem, CoreConfig cfg)
    : prog_(std::move(program)), mem_(mem), cfg_(cfg) {}

std::uint64_t Core::memory_access(Warp& w, bool is_store) {
    const Instr& in = prog_[w.pc];
    // Coalescing: count distinct memory segments touched by the active lanes.
    // One transaction per segment -> contiguous access is cheap, scattered is not.
    std::set<std::int64_t> segments;
    for (int l = 0; l < WARP_SIZE; ++l) {
        if (!w.active[l]) continue;
        std::int32_t addr = w.regs[l][in.ra];
        if (is_store) {
            mem_.store(static_cast<std::size_t>(addr), w.regs[l][in.rb]);
        } else {
            w.regs[l][in.rd] = mem_.load(static_cast<std::size_t>(addr));
        }
        segments.insert(addr / cfg_.segment_words);
    }
    const std::uint64_t txns = segments.empty() ? 0 : segments.size();
    stats_.mem_transactions += txns;
    stats_.mem_ops += 1;
    last_txns_ = static_cast<int>(txns);
    // Latency = one round trip, plus a bandwidth penalty for each extra transaction
    // beyond the first (an uncoalesced access serializes more segment fetches).
    const std::uint64_t extra = txns > 0 ? txns - 1 : 0;
    return static_cast<std::uint64_t>(cfg_.mem_latency) +
           extra * static_cast<std::uint64_t>(cfg_.mem_txn_penalty);
}

std::uint64_t Core::execute(Warp& w) {
    const Instr& in = prog_[w.pc];
    std::uint64_t latency = 1;  // ALU/issue default

    switch (in.op) {
        case Op::MOV:
            for (int l = 0; l < WARP_SIZE; ++l)
                if (w.active[l]) w.regs[l][in.rd] = in.imm;
            break;
        case Op::TID:
            for (int l = 0; l < WARP_SIZE; ++l)
                if (w.active[l]) w.regs[l][in.rd] = w.id * WARP_SIZE + l;
            break;
        case Op::IADD:
            for (int l = 0; l < WARP_SIZE; ++l)
                if (w.active[l]) w.regs[l][in.rd] = w.regs[l][in.ra] + w.regs[l][in.rb];
            break;
        case Op::IMUL:
            for (int l = 0; l < WARP_SIZE; ++l)
                if (w.active[l]) w.regs[l][in.rd] = w.regs[l][in.ra] * w.regs[l][in.rb];
            break;
        case Op::SLT:  // set-less-than -> per-lane predicate (1 or 0)
            for (int l = 0; l < WARP_SIZE; ++l)
                if (w.active[l])
                    w.regs[l][in.rd] = (w.regs[l][in.ra] < w.regs[l][in.rb]) ? 1 : 0;
            break;
        case Op::LD:
            latency = memory_access(w, /*is_store=*/false);
            break;
        case Op::ST:
            latency = memory_access(w, /*is_store=*/true);
            break;
        case Op::JMP:  // unconditional branch for all active lanes
            w.pc = in.imm;
            stats_.warp_instructions += 1;
            return latency;
        case Op::BRA:
            branch(w);   // handles uniform vs divergent; sets pc / pushes frames
            stats_.warp_instructions += 1;
            return latency;
        case Op::HALT:
            w.halted = true;
            break;
    }

    if (in.op != Op::HALT) w.pc += 1;
    stats_.warp_instructions += 1;
    return latency;
}

void Core::branch(Warp& w) {
    // Encoding: ra = predicate register, imm = else-target, rd = join-target.
    const Instr& in = prog_[w.pc];
    std::array<bool, WARP_SIZE> taken{};      // predicate != 0  -> "then" (fall through)
    std::array<bool, WARP_SIZE> not_taken{};  // predicate == 0  -> "else"
    bool any_taken = false, any_not = false;
    for (int l = 0; l < WARP_SIZE; ++l) {
        if (!w.active[l]) continue;
        if (w.regs[l][in.ra] != 0) { taken[l] = true; any_taken = true; }
        else                        { not_taken[l] = true; any_not = true; }
    }

    const int then_pc = w.pc + 1;
    const int else_pc = in.imm;
    const int join_pc = in.rd;

    if (!(any_taken && any_not)) {
        // Uniform: the whole warp agrees -> no divergence cost, just redirect.
        w.pc = any_taken ? then_pc : else_pc;
        return;
    }

    // Divergent: run both paths serially via the reconvergence stack. Push the join
    // frame (full current mask resumes at join_pc), then the else path, and make the
    // live frame the "then" path. Each path pops when its pc reaches join_pc.
    Frame join_frame{w.active, join_pc, w.rpc};
    Frame else_frame{not_taken, else_pc, join_pc};
    w.stack.push_back(join_frame);
    w.stack.push_back(else_frame);
    w.active = taken;
    w.pc = then_pc;
    w.rpc = join_pc;
    stats_.divergent_branches += 1;
    last_diverged_ = true;
}

void Core::run(int n_threads) {
    // Partition threads into warps; lane l of warp w = global tid (w*WARP_SIZE + l).
    const int n_warps = (n_threads + WARP_SIZE - 1) / WARP_SIZE;
    warps_.assign(static_cast<std::size_t>(std::max(n_warps, 0)), Warp{});
    for (int wi = 0; wi < n_warps; ++wi) {
        Warp& w = warps_[wi];
        w.id = wi;
        w.pc = 0;
        w.rpc = -1;           // base frame: no reconvergence point
        w.stack.clear();
        for (int l = 0; l < WARP_SIZE; ++l)
            w.active[l] = (wi * WARP_SIZE + l) < n_threads;
    }

    // Single-issue, round-robin warp scheduler. Exactly one warp issues per cycle;
    // a warp that issued a long memory op is "busy" (ready_at in the future), so the
    // scheduler naturally fills those cycles with OTHER ready warps -> latency hiding.
    std::uint64_t cycle = 0;
    std::size_t rr = 0;
    while (true) {
        bool any_alive = false;
        for (const Warp& w : warps_)
            if (!w.halted) { any_alive = true; break; }
        if (!any_alive) break;

        int chosen = -1;
        for (std::size_t k = 0; k < warps_.size(); ++k) {
            std::size_t idx = (rr + k) % warps_.size();
            if (!warps_[idx].halted && warps_[idx].ready_at <= cycle) {
                chosen = static_cast<int>(idx);
                break;
            }
        }

        if (chosen < 0) {
            // No warp can issue now; jump to the next moment one becomes ready.
            std::uint64_t next = std::numeric_limits<std::uint64_t>::max();
            for (const Warp& w : warps_)
                if (!w.halted && w.ready_at > cycle) next = std::min(next, w.ready_at);
            if (next == std::numeric_limits<std::uint64_t>::max()) break;
            cycle = next;
            pending_stall_ = true;   // the clock had to jump — a real stall gap
            continue;
        }

        rr = (static_cast<std::size_t>(chosen) + 1) % warps_.size();
        Warp& w = warps_[static_cast<std::size_t>(chosen)];

        // Reconvergence: if the live path has reached its reconvergence point, pop back
        // to the frame below (rejoining lanes) before issuing. Bookkeeping, not a cycle.
        while (!w.stack.empty() && w.pc == w.rpc) {
            Frame f = w.stack.back();
            w.stack.pop_back();
            w.active = f.mask;
            w.pc = f.pc;
            w.rpc = f.rpc;
        }

        // Snapshot pre-issue state for the trace (only if tracing).
        int ev_pc = w.pc;
        std::string ev_op = (w.pc >= 0 && w.pc < static_cast<int>(prog_.size()))
                                ? op_name(prog_[static_cast<std::size_t>(w.pc)].op) : "?";
        std::uint32_t ev_mask = 0;
        if (tracing_)
            for (int l = 0; l < WARP_SIZE; ++l)
                if (w.active[l]) ev_mask |= (1u << l);
        last_txns_ = 0;
        last_diverged_ = false;

        const std::uint64_t latency = execute(w);
        w.ready_at = cycle + latency;
        stats_.cycles = std::max(stats_.cycles, w.ready_at);

        if (tracing_) {
            TraceEvent ev;
            ev.cycle = cycle;
            ev.warp_id = w.id;
            ev.pc = ev_pc;
            ev.op = ev_op;
            ev.active_mask = ev_mask;
            ev.latency = latency;
            ev.mem_txns = last_txns_;
            ev.diverged = last_diverged_;
            ev.after_stall = pending_stall_;
            trace_.push_back(std::move(ev));
            pending_stall_ = false;
        }
        cycle += 1;  // single instruction issued this cycle
    }
}

}  // namespace simt
