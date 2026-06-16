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

#include "cow_ptr.hpp"
#include "rocprof_trace_decoder/cxx/common.hpp"
#include "rocprof_trace_decoder/cxx/segment.hpp"

// Pull the unified types into the global namespace for internal library use.
using address_range_t = rocprof_trace_decoder::codeobj::address_range_t;
using CodeobjTableTranslator = rocprof_trace_decoder::codeobj::CodeobjTableTranslator;

// A CowPtr-shared CodeobjTableTranslator paired with a single-entry lookup cache.
// The cache lives here, next to the CowPtr, not inside the shared translator: each
// CSRegisterHandler owns its own CachedTable and is driven single-threaded, so the
// cache is never shared across threads, while the underlying translator (handed out
// as shared_ptr<const T> via read()) stays immutable and safe for concurrent reads.
struct CachedTable
{
    // Cached lookup over the shared translator. Uses read() so it never forks the CowPtr.
    bool find(uint64_t addr, address_range_t& out) const
    {
        if (!cached_segment.inrange(addr))
            if (!table.read().find_codeobj_in_range(addr, cached_segment)) return false;
        out = cached_segment;
        return true;
    }

    const CodeobjTableTranslator& read() const { return table.read(); }

    // Mutating access invalidates the cache, since the ranges may change.
    CodeobjTableTranslator& write()
    {
        cached_segment = {};
        return table.write();
    }

    bool null() const { return table.null(); }

    CowPtr<CodeobjTableTranslator> table{};
    mutable address_range_t cached_segment{};
};

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

/// Internal helper: translate a raw virtual address to a pcinfo_t, going through
/// a CachedTable so repeated lookups hit the cache.
pcinfo_t ToPcV2(const CachedTable& table, uint64_t pc);
