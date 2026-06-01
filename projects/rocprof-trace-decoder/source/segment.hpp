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

#include "rocprof_trace_decoder/cxx/common.hpp"
#include "rocprof_trace_decoder/cxx/segment.hpp"

// Pull the unified types into the global namespace for internal library use.
using address_range_t = rocprof_trace_decoder::codeobj::address_range_t;
using CodeobjTableTranslator = rocprof_trace_decoder::codeobj::CodeobjTableTranslator;

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

/// Internal helper: translate a raw virtual address to a pcinfo_t using the table.
pcinfo_t ToPcV2(CodeobjTableTranslator& table, uint64_t pc);
