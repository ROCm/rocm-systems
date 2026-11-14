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

// Memory a replayed dispatch is handed that the snapshot does not cover.
//
// The snapshot restores tracked coarse-grained device allocations and module-scope variables
// exposed as HSA symbols before every replayed pass. A checked pointer to memory outside that
// snapshot means the dispatch runs once without replay, including fine-grained device, host-pinned,
// managed, locked or pageable host memory, another agent's memory and virtual-memory mappings.
// Which words of the kernarg segment hold a pointer comes from the kernel's own `amdhsa` metadata.
// Complete descriptions exclude small scalars; partial descriptions also scan the pre-hidden words.
// Metadata-described printf, hostcall and heap pointers are checked; without metadata every word is
// a candidate. What is not seen is a pointer the kernel finds inside memory it was handed, or one
// stored in a by-value argument too small to be told from a scalar.

#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace untracked_memory
{
enum class kind_t
{
    host_pinned,
    managed,
    virtual_mem,
    device_other,
    locked_host,
    host_pageable,
    unclassified,
};

// What hsa_amd_pointer_info (and, for UNKNOWN pointers, the process's writable mappings)
// says about an address, reduced to what classification and kernarg bounds checks need.
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
    bool   owner_is_gpu             = false;
    bool   in_agent_allocation      = false;
    bool   in_writable_host_mapping = false;
    bool   pointer_info_succeeded   = false;
    bool   host_mappings_ok         = false;
    // Allocation extent in the queried address view; zero when HSA does not report one.
    uint64_t allocation_base = 0;
    uint64_t allocation_size = 0;
};

using pointer_facts_fn_t = std::function<pointer_facts_t(uint64_t address)>;

struct address_span_t
{
    uint64_t begin = 0;
    uint64_t end   = 0;
};

struct host_mappings_t
{
    std::vector<address_span_t> spans = {};
    bool                        ok    = false;
};

struct hit_t
{
    size_t   kernarg_offset = 0;
    uint64_t address        = 0;
    kind_t   kind           = kind_t::unclassified;
};

const char*
to_string(kind_t kind);

// Writable mappings of this process that hold host memory, sorted by address. Mappings through
// /dev/kfd and /dev/dri are left out. Read once per dispatch
// that needs it, not cached across dispatches: this is the application's address space, and it
// changes as the application maps and unmaps. A failed open or read returns ok=false; a
// successfully read empty mapping set returns ok=true.
host_mappings_t
read_writable_host_mappings();

// The same mapping scan from a stream in /proc/self/maps format.
host_mappings_t
read_writable_host_mappings(std::istream& file);

pointer_facts_fn_t
make_hsa_pointer_facts();

// The runtime's own copy and fill kernels. They are handed HIP's staging and scratch memory rather
// than application data, and writing the same value again in a later pass is harmless -- provided a
// copy's source and destination do not overlap, which is not detected here, since nothing restores
// the staging memory between passes. They are replayed without the check below. Only these are
// exempt: the other __amd_rocclr_ kernels (streamOpsIncrement, streamOpsWait, scheduler, initHeap,
// ...) change runtime state on every execution, and replaying those would corrupt it. Anything not
// listed takes the strict path.
// `kernel_name` is the name the code object registry holds, with or without the ".kd" suffix its
// kernel descriptors carry.
bool
is_idempotent_runtime_kernel(std::string_view kernel_name);

// Sorted spans of the snapshot's blocks.
std::vector<address_span_t>
covered_spans(const memory_snapshot::device_snapshot_t& snapshot);

std::optional<kind_t>
classify_pointer(const pointer_facts_t& facts);

// The first kernarg word that identifies memory outside the snapshot. `pointer_regions` holds
// ranges that can carry a pointer, or is null when every word is a candidate. The caller supplies
// readable CPU storage for segment_size bytes. Words are read from
// each region's metadata offset, including unaligned regions. A successful UNKNOWN query with no
// writable-host-mapping hit is not treated as unsupported memory.
std::optional<hit_t>
scan_kernarg(const uint8_t*                                    kernarg,
             size_t                                            segment_size,
             const std::vector<code_object::kernarg_region_t>* pointer_regions,
             const std::vector<address_span_t>&                covered,
             const pointer_facts_fn_t&                         facts);

// Copies a verified HSA kernarg allocation into host storage before scanning it. GPU addresses
// use a GPU-to-host copy; CPU views use a fault-aware read. facts is
// queried for both the buffer and candidate argument pointers. Returns false when the buffer
// extent or complete read cannot be verified, or a checked pointer identifies unsupported memory
// or cannot be classified; the dispatch must then run once without replay.
bool
can_replay_dispatch(const uint8_t*                                    kernarg,
                    size_t                                            segment_size,
                    const std::vector<code_object::kernarg_region_t>* pointer_regions,
                    const memory_snapshot::device_snapshot_t&         snapshot,
                    const pointer_facts_fn_t&                         facts);

// The same for a dispatch packet, with its kernel and pointer ranges found by kernel_object.
// A kernel the code object registry does not know about also runs once.
bool
can_replay_dispatch(const hsa_kernel_dispatch_packet_t&       packet,
                    const memory_snapshot::device_snapshot_t& snapshot);
}  // namespace untracked_memory
}  // namespace kernel_replay
}  // namespace rocprofiler
