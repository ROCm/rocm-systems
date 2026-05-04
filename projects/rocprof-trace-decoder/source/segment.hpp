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
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include "common.hpp"

inline bool operator==(const pcinfo_t& a, const pcinfo_t& b)
{
    return a.address == b.address && a.code_object_id == b.code_object_id;
}
inline bool operator!=(const pcinfo_t& a, const pcinfo_t& b)
{
    return a.address != b.address || a.code_object_id != b.code_object_id;
}

template <> struct std::hash<pcinfo_t>
{
    uint64_t operator()(const pcinfo_t& d) const
    {
        return (d.address >> 2) ^ (d.code_object_id << 24) ^ (d.code_object_id >> 40);
    }
};

struct address_range_t
{
    uint64_t vbegin{0};
    uint64_t size{0};
    uint64_t id{0};

    bool operator==(const address_range_t& other) const
    {
        return (vbegin >= other.vbegin && vbegin < other.vbegin + other.size) ||
               (other.vbegin >= vbegin && other.vbegin < vbegin + size);
    }
    bool operator<(const address_range_t& other) const
    {
        if (*this == other) return false;
        return vbegin < other.vbegin;
    }
    bool inrange(uint64_t _addr) const { return vbegin <= _addr && vbegin + size > _addr; };
};

class CodeobjTableTranslator
{
public:
    // Insert a range, erasing any existing overlapping ranges first.
    std::pair<std::set<address_range_t>::iterator, bool> insert(const address_range_t& value)
    {
        if (value.size == 0) return {ranges_.end(), false};

        clear_cache();

        // Erase all existing ranges that overlap with the new one
        auto it = ranges_.lower_bound(address_range_t{value.vbegin, 0, 0});
        if (it != ranges_.begin())
        {
            auto prev = std::prev(it);
            if (prev->vbegin + prev->size > value.vbegin) ranges_.erase(prev);
        }
        while (it != ranges_.end() && it->vbegin < value.vbegin + value.size) ranges_.erase(it++);

        return ranges_.insert(value);
    }

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
    pcinfo_t ToPcV2(uint64_t pc);

private:
    std::set<address_range_t> ranges_;
    address_range_t cached_segment{};
};
