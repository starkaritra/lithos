#pragma once
// A tiny text assembler for the mini-GPU ISA. One instruction per line; '#' or ';'
// starts a comment. Registers are written rN (r0..r15). Examples:
//   mov r1, 0        ; base address of A
//   tid r0           ; r0 = global thread id
//   iadd r4, r1, r0  ; addrA = baseA + tid
//   ld  r7, r4       ; r7 = mem[r4]
//   st  r6, r9       ; mem[r6] = r9
//   halt
#include "simt/isa.hpp"

#include <string>
#include <vector>

namespace simt {

// Parses assembly text into a program. Throws std::runtime_error on a bad line.
std::vector<Instr> assemble(const std::string& source);

}  // namespace simt
