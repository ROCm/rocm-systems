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
// Only directly-allocated device memory is tracked (the HSA pool and region allocators). Unified /
// managed memory is intentionally out of scope, so the VMEM map/unmap paths are not hooked.
//
// The tracker chains the existing HSA table function pointers; when tracking is disabled the hooks
// cost a single relaxed atomic load on top of the chained call.
namespace memory_tracker
{
// Per-allocation record: byte size plus the agent that owns the memory. The agent is used to scope
// snapshots so a replay only saves/restores its own agent's device memory.
struct alloc_info_t
{
    size_t      size  = 0;
    hsa_agent_t agent = {.handle = 0};
};

// ptr -> allocation size in bytes.
using alloc_map_t   = std::unordered_map<void*, size_t>;
using tracked_map_t = std::unordered_map<void*, alloc_info_t>;

// Enable/disable inventory population. Disabled by default until a replay context is configured.
bool
set_tracking_enabled(bool enabled);

bool
tracking_enabled();

void
record_alloc(void* ptr, size_t size);

void
record_free(void* ptr);

// Frozen ptr->size view of the inventory restricted to allocations owned by `agent`, taken under a
// brief read lock. Used by memory_snapshot at snap time so each replay only snapshots/restores its
// own agent's device memory.
alloc_map_t
snap_inventory(hsa_agent_t agent);

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
