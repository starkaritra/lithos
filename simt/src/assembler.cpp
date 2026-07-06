#include "simt/assembler.hpp"
#include "simt/core.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace simt {
namespace {

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Strip a leading 'r'/'R' and parse the register index; validates range.
int parse_reg(const std::string& tok) {
    if (tok.size() < 2 || (tok[0] != 'r' && tok[0] != 'R'))
        throw std::runtime_error("expected register, got '" + tok + "'");
    int idx = std::stoi(tok.substr(1));
    if (idx < 0 || idx >= NREGS)
        throw std::runtime_error("register out of range: '" + tok + "'");
    return idx;
}

// Split a line into whitespace/comma separated tokens, dropping comments.
std::vector<std::string> tokenize(const std::string& line) {
    std::string s = line;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '#' || s[i] == ';') { s = s.substr(0, i); break; }
        if (s[i] == ',') s[i] = ' ';
    }
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) out.push_back(t);
    return out;
}

// A parsed instruction whose branch/jump targets are still symbolic label names,
// resolved to instruction indices in a second pass.
struct Pending {
    Instr in;
    std::string label_a;   // JMP target / BRA else-target
    std::string label_b;   // BRA join-target
    int lineno = 0;
};

}  // namespace

std::vector<Instr> assemble(const std::string& source) {
    static const std::unordered_map<std::string, Op> ops = {
        {"mov", Op::MOV}, {"tid", Op::TID}, {"iadd", Op::IADD}, {"imul", Op::IMUL},
        {"slt", Op::SLT}, {"ld", Op::LD}, {"st", Op::ST}, {"jmp", Op::JMP},
        {"bra", Op::BRA}, {"halt", Op::HALT},
    };

    std::vector<Pending> pending;
    std::unordered_map<std::string, int> labels;  // name -> instruction index

    // Pass 1: parse instructions, record label positions.
    std::istringstream lines(source);
    std::string line;
    int lineno = 0;
    while (std::getline(lines, line)) {
        ++lineno;
        auto tok = tokenize(line);
        if (tok.empty()) continue;

        // A lone "name:" token defines a label at the next instruction index.
        if (tok.size() == 1 && tok[0].size() > 1 && tok[0].back() == ':') {
            labels[tok[0].substr(0, tok[0].size() - 1)] = static_cast<int>(pending.size());
            continue;
        }

        auto it = ops.find(lower(tok[0]));
        if (it == ops.end())
            throw std::runtime_error("line " + std::to_string(lineno) +
                                     ": unknown op '" + tok[0] + "'");
        Pending p;
        p.lineno = lineno;
        p.in.op = it->second;
        try {
            switch (p.in.op) {
                case Op::MOV:  // mov rd, imm
                    p.in.rd = parse_reg(tok.at(1));
                    p.in.imm = std::stoi(tok.at(2));
                    break;
                case Op::TID:  // tid rd
                    p.in.rd = parse_reg(tok.at(1));
                    break;
                case Op::IADD:
                case Op::IMUL:
                case Op::SLT:  // op rd, ra, rb
                    p.in.rd = parse_reg(tok.at(1));
                    p.in.ra = parse_reg(tok.at(2));
                    p.in.rb = parse_reg(tok.at(3));
                    break;
                case Op::LD:   // ld rd, ra
                    p.in.rd = parse_reg(tok.at(1));
                    p.in.ra = parse_reg(tok.at(2));
                    break;
                case Op::ST:   // st ra, rb
                    p.in.ra = parse_reg(tok.at(1));
                    p.in.rb = parse_reg(tok.at(2));
                    break;
                case Op::JMP:  // jmp label
                    p.label_a = tok.at(1);
                    break;
                case Op::BRA:  // bra rp, else_label, join_label
                    p.in.ra = parse_reg(tok.at(1));
                    p.label_a = tok.at(2);
                    p.label_b = tok.at(3);
                    break;
                case Op::HALT:
                    break;
            }
        } catch (const std::out_of_range&) {
            throw std::runtime_error("line " + std::to_string(lineno) +
                                     ": too few operands for '" + tok[0] + "'");
        }
        pending.push_back(p);
    }

    // Pass 2: resolve symbolic labels to instruction indices.
    auto resolve = [&](const std::string& name, int ln) -> int {
        auto it = labels.find(name);
        if (it == labels.end())
            throw std::runtime_error("line " + std::to_string(ln) +
                                     ": undefined label '" + name + "'");
        return it->second;
    };

    std::vector<Instr> prog;
    prog.reserve(pending.size());
    for (auto& p : pending) {
        if (p.in.op == Op::JMP) {
            p.in.imm = resolve(p.label_a, p.lineno);
        } else if (p.in.op == Op::BRA) {
            p.in.imm = resolve(p.label_a, p.lineno);  // else-target
            p.in.rd = resolve(p.label_b, p.lineno);   // join-target
        }
        prog.push_back(p.in);
    }
    return prog;
}

}  // namespace simt
