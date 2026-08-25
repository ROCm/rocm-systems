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

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace rocprofiler
{
namespace kernel_replay
{
// Minimal inventory of live device allocations used by kernel-replay snap/restore.
//
// Only directly-allocated device memory is snapshotted (the HSA pool and region allocators).
// Unified / managed memory and virtual-memory mappings are out of scope for restore, but they are
// *counted*: the VMEM map/unmap paths are hooked so the replay window can tell the difference
// between "this agent holds only snapshottable memory" and "this agent holds application data the
// snapshot will miss". Without that distinction a snapshot of a stream-ordered or
// expandable-segment allocator reports success having captured almost nothing, and every pass after
// the first runs on mutated inputs with no diagnostic. See untracked_summary_t.
//
// The tracker chains the existing HSA table function pointers; when tracking is disabled the hooks
// cost a single relaxed atomic load on top of the chained call.
namespace memory_tracker
{
// Per-allocation record: byte size plus the agent that owns the memory. The agent is used to scope
// snapshots so a replay only saves/restores its own agent's device memory.
//
// `generation` is a process-monotonic stamp assigned at record_alloc. It exists because (base,
// size) does not identify an allocation: the alloc/free wrappers are deliberately outside the
// per-agent replay lock, so another thread can free a region and allocate a new one at the same
// base address inside the replay window. Caching allocators make same-address reuse the common
// case rather than an exotic one. Restoring the old bytes into the new allocation would silently
// corrupt it, so snap records the generation it saw and restore refuses to write unless it still
// matches.
struct alloc_info_t
{
    size_t      size       = 0;
    hsa_agent_t agent      = {.handle = 0};
    uint64_t    generation = 0;
};

// ptr -> allocation size in bytes.
using alloc_map_t   = std::unordered_map<void*, size_t>;
using tracked_map_t = std::unordered_map<void*, alloc_info_t>;

// Byte/region counts for device-visible memory that the snapshot cannot capture. A kernel writing
// to any of it observes values accumulated across passes instead of identical inputs, so the replay
// window uses this to decline (or at minimum to warn) rather than return numbers derived from
// mutated inputs.
//
// The two classes are reported separately because they justify different policies:
//
//  * `vmem_*` counts live hsa_amd_vmem_map ranges. Nothing maps virtual memory unless the
//    application opted in -- `hipMallocAsync` with the default (VM-backed) pool,
//    `hipMemAddressReserve`/`hipMemMap`, PyTorch's `expandable_segments:True`, Kokkos built with
//    `KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC` (the default for HIP < 7.0.0). A non-zero count is
//    therefore unambiguous evidence that application data is outside the snapshot.
//  * `pool_*` counts pool allocations that are GPU-owned and not kernarg but failed the
//    coarse-grained test -- fine-grained device memory and (when it lands on the GPU rather than
//    pinned host) managed memory. Runtime-internal allocations can land here too, so a non-zero
//    count is weaker evidence and only warns by default.
struct untracked_summary_t
{
    size_t vmem_bytes   = 0;
    size_t vmem_regions = 0;
    size_t pool_bytes   = 0;
    size_t pool_regions = 0;
};

// True when the summary names at least one region the snapshot cannot capture. Region counts rather
// than byte counts are the evidence: a zero-length mapping is still a mapping, and a caller that
// tested bytes would miss it.
bool
any_untracked(const untracked_summary_t& summary);

// Enable/disable inventory population. Disabled by default until a replay context is configured.
bool
set_tracking_enabled(bool enabled);

bool
tracking_enabled();

void
record_alloc(void* ptr, size_t size);

void
record_free(void* ptr);

// Frozen ptr->alloc_info view of the inventory restricted to allocations owned by `agent`, taken
// under a brief read lock. Used by memory_snapshot at snap time so each replay only
// snapshots/restores its own agent's device memory. The generation in each record is what restore
// re-checks to reject an address that was freed and reallocated inside the window.
tracked_map_t
snap_inventory(hsa_agent_t agent);

// Total bytes/regions of tracked allocations owned by `agent`, without copying anything. The replay
// window uses this for admission control: a snapshot larger than the host can hold, or one whose
// projected copy time dwarfs the kernel being measured, should be declined before any device->host
// traffic is issued rather than discovered by std::bad_alloc partway through.
struct footprint_t
{
    size_t bytes   = 0;
    size_t regions = 0;
};

footprint_t
tracked_footprint(hsa_agent_t agent);

// Device-visible memory on `agent` that snap() cannot capture. See untracked_summary_t.
untracked_summary_t
untracked_device_memory(hsa_agent_t agent);

// Side inventory of live pool allocations that carried HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG and are
// otherwise trackable (coarse device VRAM). The main inventory never records those pointers -- the
// flag is shared by HIP kernarg pools and profiler buffers, which must not be snapshotted -- but a
// direct-HSA application may put ordinary writable device data behind the same flag. Those app
// buffers are omitted from snapshots (unsupported allocation class for beta). Declining replay
// whenever this side inventory is non-empty is not viable: the HIP runtime and the SDK's own AQL
// pools routinely keep trackable+executable allocations live. Exposed so tests can assert the
// omission and so callers can detect the unsupported class.
alloc_map_t
unsupported_executable(hsa_agent_t agent);

bool
agent_has_unsupported_executable(hsa_agent_t agent);

// Test/helper entry points that invoke the same allocate/free wrappers the HSA table interceptor
// installs. get_amd_ext_table() returns the internal (unwrapped) table, so unit tests that need to
// exercise the executable-flag path must call these rather than the internal table pointers.
hsa_status_t
tracking_pool_allocate(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr);

hsa_status_t
tracking_pool_free(void* ptr);

// Generation currently recorded for `ptr`, or 0 if it is not a live tracked allocation. Exposed for
// tests that need to observe the ABA stamp directly.
uint64_t
generation_of(void* ptr);

// Same purpose for the virtual-memory path. The agent is explicit rather than resolved from `va`
// so a test can drive the accounting without a real VMM allocation (which needs a physical handle
// and a reserved address range); the interceptor resolves it from hsa_amd_pointer_info.
void
record_vmem_map(void* va, size_t size, hsa_agent_t agent);

// `size` is the number of bytes being unmapped, which HSA permits to be a prefix of a larger
// mapping. Only an unmap that covers the whole recorded range drops the record; a partial unmap
// shrinks it. Erasing on any unmap would let a still-mapped range disappear from the accounting,
// and the consequence of under-counting is admitting a replay that should have been declined.
void
record_vmem_unmap(void* va, size_t size);

// The Synchronized allocation inventory. Exposed so restore() can look up a block and copy it under
// the read lock, which blocks a concurrent free for the copy. Callers reachable during finalization
// must first gate on registration::get_fini_status(): the alloc/free wrappers outlive this static,
// so touching it after teardown locks a freed mutex. (restore() only runs during active replay, so
// it is safe without the guard.)
rocprofiler::common::Synchronized<tracked_map_t>&
inventory();
}  // namespace memory_tracker

void
memory_tracker_init(hsa::hsa_core_table_t* table, uint64_t lib_instance);

void
memory_tracker_init(hsa::hsa_amd_ext_table_t* table, uint64_t lib_instance);
}  // namespace kernel_replay
}  // namespace rocprofiler
