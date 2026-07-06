#include "simt/isa.hpp"

namespace simt {

const char* op_name(Op op) {
    switch (op) {
        case Op::MOV:  return "mov";
        case Op::TID:  return "tid";
        case Op::IADD: return "iadd";
        case Op::IMUL: return "imul";
        case Op::SLT:  return "slt";
        case Op::LD:   return "ld";
        case Op::ST:   return "st";
        case Op::JMP:  return "jmp";
        case Op::BRA:  return "bra";
        case Op::HALT: return "halt";
    }
    return "?";
}

}  // namespace simt
