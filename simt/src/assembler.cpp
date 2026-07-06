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

}  // namespace

std::vector<Instr> assemble(const std::string& source) {
    static const std::unordered_map<std::string, Op> ops = {
        {"mov", Op::MOV}, {"tid", Op::TID}, {"iadd", Op::IADD},
        {"imul", Op::IMUL}, {"ld", Op::LD}, {"st", Op::ST}, {"halt", Op::HALT},
    };

    std::vector<Instr> prog;
    std::istringstream lines(source);
    std::string line;
    int lineno = 0;
    while (std::getline(lines, line)) {
        ++lineno;
        auto tok = tokenize(line);
        if (tok.empty()) continue;

        auto it = ops.find(lower(tok[0]));
        if (it == ops.end())
            throw std::runtime_error("line " + std::to_string(lineno) +
                                     ": unknown op '" + tok[0] + "'");
        Instr in;
        in.op = it->second;
        try {
            switch (in.op) {
                case Op::MOV:  // mov rd, imm
                    in.rd = parse_reg(tok.at(1));
                    in.imm = std::stoi(tok.at(2));
                    break;
                case Op::TID:  // tid rd
                    in.rd = parse_reg(tok.at(1));
                    break;
                case Op::IADD:
                case Op::IMUL:  // op rd, ra, rb
                    in.rd = parse_reg(tok.at(1));
                    in.ra = parse_reg(tok.at(2));
                    in.rb = parse_reg(tok.at(3));
                    break;
                case Op::LD:   // ld rd, ra   (rd = mem[ra])
                    in.rd = parse_reg(tok.at(1));
                    in.ra = parse_reg(tok.at(2));
                    break;
                case Op::ST:   // st ra, rb   (mem[ra] = rb)
                    in.ra = parse_reg(tok.at(1));
                    in.rb = parse_reg(tok.at(2));
                    break;
                case Op::HALT:
                    break;
            }
        } catch (const std::out_of_range&) {
            throw std::runtime_error("line " + std::to_string(lineno) +
                                     ": too few operands for '" + tok[0] + "'");
        }
        prog.push_back(in);
    }
    return prog;
}

}  // namespace simt
