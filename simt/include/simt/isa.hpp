#pragma once
// Grove Arm A — mini-GPU (SIMT) instruction set (v1, minimal by design, D-015/OQ-6).
// A tiny register-based ISA executed per-warp in lockstep. Kept small on purpose:
// just enough to run real parallel kernels (vector-add -> reduction -> matmul) and
// expose GPU microarchitecture effects. Branch/predicate ops arrive with the
// divergence slice; this v1 covers the vector-add spine.
#include <cstdint>

namespace simt {

enum class Op {
    MOV,   // rd = imm                     (immediate load)
    TID,   // rd = global thread id        (lane's position in the grid)
    IADD,  // rd = ra + rb
    IMUL,  // rd = ra * rb
    LD,    // rd = global_mem[ ra ]        (word-addressed load)
    ST,    // global_mem[ ra ] = rb        (word-addressed store)
    HALT   // stop this warp
};

struct Instr {
    Op op{Op::HALT};
    int rd{0};       // destination register index
    int ra{0};       // source A register index (or address register for LD/ST)
    int rb{0};       // source B register index (value register for ST)
    std::int32_t imm{0};  // immediate (MOV)
};

const char* op_name(Op op);

}  // namespace simt
