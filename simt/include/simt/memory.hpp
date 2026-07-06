#pragma once
// Global (off-chip) memory model. v1 stores int32 words and range-checks access.
// Transaction/coalescing accounting lives in the Core (it needs the per-warp
// address set), so this stays a clean storage primitive.
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace simt {

class GlobalMemory {
public:
    explicit GlobalMemory(std::size_t words) : data_(words, 0) {}

    std::int32_t load(std::size_t word_addr) const {
        check(word_addr);
        return data_[word_addr];
    }

    void store(std::size_t word_addr, std::int32_t val) {
        check(word_addr);
        data_[word_addr] = val;
    }

    std::size_t size() const { return data_.size(); }

private:
    void check(std::size_t a) const {
        if (a >= data_.size())
            throw std::out_of_range("global memory access out of range");
    }
    std::vector<std::int32_t> data_;
};

}  // namespace simt
