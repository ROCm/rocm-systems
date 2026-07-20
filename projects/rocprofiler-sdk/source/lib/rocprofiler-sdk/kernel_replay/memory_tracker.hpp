// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>

#include <cstddef>
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
// ptr -> allocation size in bytes.
using alloc_map_t = std::unordered_map<void*, size_t>;

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

// Install inventory wrappers on top of the existing table function pointers.
void
update_table(hsa::hsa_core_table_t* table);

void
update_table(hsa::hsa_amd_ext_table_t* table);
}  // namespace memory_tracker
}  // namespace kernel_replay
}  // namespace rocprofiler
