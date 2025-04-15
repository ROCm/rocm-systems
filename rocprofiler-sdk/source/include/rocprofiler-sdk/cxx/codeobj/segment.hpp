// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace sdk
{
namespace codeobj
{
namespace segment
{
using marker_id_t = size_t;

struct address_range_t
{
    uint64_t    addr{0};
    uint64_t    size{0};
    marker_id_t id{0};

    bool operator==(const address_range_t& other) const
    {
        return (addr >= other.addr && addr < other.addr + other.size) ||
               (other.addr >= addr && other.addr < addr + size);
    }
    bool operator<(const address_range_t& other) const
    {
        if(*this == other) return false;
        return addr < other.addr;
    }
    bool inrange(uint64_t _addr) const { return addr <= _addr && addr + size > _addr; };
};

/**
 * @brief Finds a candidate codeobj for the given vaddr
 */
class CodeobjTableTranslator : public std::set<address_range_t>
{
    using Super = std::set<address_range_t>;

public:
    address_range_t find_codeobj_in_range(uint64_t addr)
    {
        if(!cached_segment.inrange(addr))
        {
            auto it = this->find(address_range_t{addr, 0, 0});
            if(it == this->end()) throw std::exception();
            cached_segment = *it;
        }
        return cached_segment;
    }

    void clear_cache() { cached_segment = {}; }
    bool remove(const address_range_t& range)
    {
        clear_cache();
        return this->erase(range) != 0;
    }
    bool remove(uint64_t addr) { return remove(address_range_t{addr, 0, 0}); }

private:
    address_range_t cached_segment{};
};

}  // namespace segment
}  // namespace codeobj
}  // namespace sdk
}  // namespace rocprofiler
