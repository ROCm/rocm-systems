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
// The snapshot restores tracked coarse-grained device allocations and module-scope variables before
// every replayed pass. A kernel that writes anything else would have its writes applied again by
// every pass. Fine-grained device memory of the replaying agent that the dispatch is handed is
// saved before the first pass and restored before every later pass. Other dispatches to the agent
// wait on the replay lock, but host stores to this memory do not, so a host thread must not write
// it while the dispatch is being replayed. Any other such memory (host-pinned, managed, locked or
// pageable host memory, another agent's memory, virtual-memory mappings) means the dispatch runs
// once without replay. Only pointer-sized words of the explicit kernel arguments are examined: a
// pointer stored inside other memory, or packed into a by-value struct at an unaligned offset, is
// not seen.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_set>
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
};

// What hsa_amd_pointer_info (and, when it reports no owner, the process's writable mappings)
// says about an address, reduced to what the classification needs.
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

    type_t   type                     = type_t::unknown;
    bool     owner_is_cpu             = false;
    bool     in_writable_host_mapping = false;
    bool     fine_grained             = false;
    uint64_t agent_base               = 0;  // GPU-visible base of the allocation, 0 if none
    uint64_t size                     = 0;  // extent of the allocation, 0 if unknown
    uint64_t owner                    = 0;  // owning agent's handle, 0 if none
};

using pointer_facts_fn_t = std::function<pointer_facts_t(uint64_t address)>;

struct address_span_t
{
    uint64_t begin = 0;
    uint64_t end   = 0;
};

// An allocation as it would be saved. `size` is 0 when it cannot be saved.
struct region_t
{
    kind_t   kind  = kind_t::host_pinned;
    uint64_t base  = 0;
    uint64_t size  = 0;
    uint64_t owner = 0;
};

struct hit_t
{
    size_t   kernarg_offset = 0;
    uint64_t address        = 0;
    region_t region         = {};
};

const char*
to_string(kind_t kind);

// Writable mappings of this process that hold host memory, sorted by address. Device mappings and
// the pages the kernel manages itself (vvar, vdso, vsyscall) are left out.
std::vector<address_span_t>
read_writable_host_mappings();

pointer_facts_fn_t
make_hsa_pointer_facts();

// The runtime's own copy and fill kernels. They are handed HIP's staging and scratch memory rather
// than application data, and writing the same value again in a later pass is harmless, so they are
// replayed without the check below. Only these are exempt: the other __amd_rocclr_ kernels
// (streamOpsIncrement, streamOpsWait, scheduler, initHeap, ...) change runtime state on every
// execution, and replaying those would corrupt it. Anything not listed takes the strict path.
// `kernel_name` is the name the code object registry holds, with or without the ".kd" suffix its
// kernel descriptors carry.
bool
is_idempotent_runtime_kernel(std::string_view kernel_name);

// Sorted spans of the snapshot's blocks.
std::vector<address_span_t>
covered_spans(const memory_snapshot::device_snapshot_t& snapshot);

std::optional<kind_t>
classify_pointer(const pointer_facts_t& facts);

// The whole allocation `facts` describes, if it is fine-grained device memory that can be saved.
region_t
region_of(kind_t kind, const pointer_facts_t& facts);

// Leading bytes of a kernarg segment that hold the kernel's own arguments. `metadata_size` is where
// that kernel's own metadata says its arguments end, or nullopt when no metadata was read for it.
// Assuming a fixed block of hidden arguments at the end of the segment would hide pointers the
// passes write through, because a kernel's own arguments can run past such a block; an unknown
// boundary leaves the whole segment to scan, which costs a replay rather than correctness.
size_t
explicit_arg_size(size_t segment_size, std::optional<size_t> metadata_size);

// Kernarg words that point at untracked memory, scanning the first `explicit_size` bytes of the
// segment. `seen` holds values already looked up, so a buffer passed in several arguments is
// classified once.
std::vector<hit_t>
scan_kernarg(const uint8_t*                     kernarg,
             size_t                             explicit_size,
             const std::vector<address_span_t>& covered,
             const pointer_facts_fn_t&          facts,
             std::unordered_set<uint64_t>&      seen);

// Saved copies of untracked allocations, taken before the first replayed pass.
class region_set_t
{
public:
    // Total bytes one replay may save.
    static constexpr uint64_t max_total_bytes = uint64_t{512} << 20;

    using copy_fn_t = std::function<bool(void* dst, const void* src, size_t size)>;

    enum class save_status_t
    {
        saved,
        already_saved,
        not_restorable,
        too_large,
        copy_failed,
    };

    region_set_t();
    explicit region_set_t(copy_fn_t copy);

    save_status_t save(const region_t& region);

    bool     contains(uint64_t address) const;
    bool     empty() const { return m_saved.empty(); }
    size_t   size() const { return m_saved.size(); }
    uint64_t bytes() const { return m_bytes; }

    bool restore() const;

private:
    struct saved_t
    {
        region_t             region = {};
        std::vector<uint8_t> data   = {};
    };

    copy_fn_t            m_copy  = {};
    std::vector<saved_t> m_saved = {};
    uint64_t             m_bytes = 0;
};

// Saves the untracked memory one dispatch's explicit kernel arguments point at into `saved`.
// `explicit_size` is the part of the segment holding the kernel's own arguments (see
// explicit_arg_size). Returns false, after logging why, when an argument points at memory that
// cannot be saved or belongs to another agent; the dispatch must then run once without replay.
bool
save_dispatch_memory(region_set_t&                             saved,
                     const uint8_t*                            kernarg,
                     size_t                                    explicit_size,
                     const memory_snapshot::device_snapshot_t& snapshot,
                     hsa_agent_t                               agent,
                     const pointer_facts_fn_t&                 facts);

// The same for a dispatch packet, with the argument layout taken from the loaded kernel. A kernel
// the code object registry does not know about also runs once.
bool
save_dispatch_memory(region_set_t&                             saved,
                     const hsa_kernel_dispatch_packet_t&       packet,
                     const memory_snapshot::device_snapshot_t& snapshot,
                     hsa_agent_t                               agent);
}  // namespace untracked_memory
}  // namespace kernel_replay
}  // namespace rocprofiler
