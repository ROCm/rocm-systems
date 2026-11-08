// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Memory a recorded dispatch is handed that the replay snapshot does not cover.
//
// The snapshot restores tracked coarse-grained device allocations and module-scope variables before
// every replayed pass. A kernel that writes anything else (host-pinned, managed, fine-grained or
// stream-ordered memory, or pageable host memory under XNACK) would have its writes applied again
// by every pass, so a range that hands a kernel such memory is not replayed. Only pointer-sized
// words of the explicit kernel arguments are examined: a pointer stored inside other memory, or
// packed into a by-value struct at an unaligned offset, is not seen.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace range_replay
{
struct range_context_t;

namespace untracked_memory
{
// HIP code object v5 appends a fixed block of hidden arguments to the explicit ones. It holds
// runtime pointers (printf buffer, hostcall buffer, heap) that are not the kernel's data.
constexpr size_t implicit_arg_block_bytes = 256;

enum class kind_t
{
    host_pinned,
    managed,
    virtual_mem,
    device_other,
    locked_host,
    host_pageable,
};

// What hsa_amd_pointer_info (and, under XNACK, the process's writable mappings) says about an
// address, reduced to what the classification needs.
struct pointer_facts_t
{
    enum class type_t
    {
        unknown,
        hsa,
        locked,
        reserved,
        vmem,
        other,
    };

    type_t type                     = type_t::unknown;
    bool   owner_is_cpu             = false;
    bool   in_writable_host_mapping = false;
};

using pointer_facts_fn_t = std::function<pointer_facts_t(uint64_t address)>;

struct address_span_t
{
    uint64_t begin = 0;
    uint64_t end   = 0;
};

struct hit_t
{
    size_t   kernarg_offset = 0;
    uint64_t address        = 0;
    kind_t   kind           = kind_t::host_pinned;
};

const char*
to_string(kind_t kind);

pointer_facts_fn_t
make_hsa_pointer_facts();

// Sorted spans of the snapshot's blocks.
std::vector<address_span_t>
covered_spans(const kernel_replay::memory_snapshot::device_snapshot_t& snapshot);

std::optional<kind_t>
classify_pointer(const pointer_facts_t& facts);

// The first explicit kernarg word that points at untracked memory. `seen` holds values already
// looked up, so a buffer passed to every dispatch of a range is classified once.
std::optional<hit_t>
find_untracked(const uint8_t*                     kernarg,
               size_t                             size,
               const std::vector<address_span_t>& covered,
               const pointer_facts_fn_t&          facts,
               std::unordered_set<uint64_t>&      seen);
}  // namespace untracked_memory

// Declines the range (SNAPSHOT_FAILED) if a dispatch recorded since the last call was handed
// memory the entry snapshot does not cover. Called after the entry snapshot is taken.
void
decline_untracked(range_context_t& ctx);

// The same, with the pointer lookup supplied, so the decision can be tested without HSA.
void
decline_untracked(range_context_t& ctx, const untracked_memory::pointer_facts_fn_t& facts);
}  // namespace range_replay
}  // namespace rocprofiler
