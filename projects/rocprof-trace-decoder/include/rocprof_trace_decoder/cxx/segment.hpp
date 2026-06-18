// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace rocprof_trace_decoder
{
namespace codeobj
{
using code_object_id_t = size_t;

struct address_range_t
{
    uint64_t addr{0};
    uint64_t size{0};
    code_object_id_t id{0};

    address_range_t() = default;
    address_range_t(uint64_t a, uint64_t s, code_object_id_t i = 0) : addr(a), size(s), id(i) {}

    bool operator==(const address_range_t& other) const
    {
        return (addr >= other.addr && addr < other.addr + other.size) ||
               (other.addr >= addr && other.addr < addr + size);
    }
    bool operator<(const address_range_t& other) const
    {
        if (*this == other) return false;
        return addr < other.addr;
    }
    bool inrange(uint64_t _addr) const { return addr <= _addr && addr + size > _addr; };
};

/**
 * @brief Maps virtual address ranges to code object IDs.
 *
 * Supports safe insertion (evicts overlapping ranges), cached lookup,
 * and removal by address or range.
 */
class CodeobjTableTranslator
{
public:
    /// Insert a range, erasing any existing overlapping ranges first.
    std::pair<std::set<address_range_t>::iterator, bool> insert(const address_range_t& value)
    {
        if (value.size == 0) return {ranges_.end(), false};

        clear_cache();

        // Erase all existing ranges that overlap with the new one
        auto it = ranges_.lower_bound(address_range_t{value.addr, 0, 0});
        if (it != ranges_.begin())
        {
            auto prev = std::prev(it);
            if (prev->addr + prev->size > value.addr) ranges_.erase(prev);
        }
        while (it != ranges_.end() && it->addr < value.addr + value.size) ranges_.erase(it++);

        return ranges_.insert(value);
    }

    /**
     * @brief Find the code object range containing @p addr.
     * @param[in]  addr  Virtual address to look up.
     * @param[out] out   Populated with the matching range on success.
     * @return true if a matching range was found, false otherwise.
     */
    bool find_codeobj_in_range(uint64_t addr, address_range_t& out)
    {
        if (!cached_segment.inrange(addr))
        {
            auto it = ranges_.find(address_range_t{addr, 0, 0});
            if (it == ranges_.end()) return false;
            cached_segment = *it;
        }
        out = cached_segment;
        return true;
    }

    void clear_cache() { cached_segment = {}; }

    bool remove(const address_range_t& range)
    {
        clear_cache();
        return ranges_.erase(range) != 0;
    }
    bool remove(uint64_t addr) { return remove(address_range_t{addr, 0, 0}); }

    const std::set<address_range_t>& ranges() const { return ranges_; }

private:
    std::set<address_range_t> ranges_;
    address_range_t cached_segment{};
};

} // namespace codeobj
} // namespace rocprof_trace_decoder
