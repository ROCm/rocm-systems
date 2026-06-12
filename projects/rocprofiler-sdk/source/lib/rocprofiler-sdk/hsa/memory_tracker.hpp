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

#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
// Maintains a flat inventory of all live GPU memory allocations across all agents for kernel
// replay snapshot/restore. Ported from Kerncap (intellikit/kerncap, kerncap.hip) which maintains
// the same flat pointer->size map plus a VMEM set, hooking pool/region allocate+free and the VMEM
// map/unmap pair. Installed on top of the existing memory_allocation wrappers by replacing HSA
// table function pointers a second time (see design doc Section 4.2); does not touch
// memory_allocation.cpp.
namespace memory_tracker
{
// Inventory entry. agent_id is informational; the map is flat (not per-agent) so cross-agent
// allocations (e.g. RCCL buffers on GPU-1 read by GPU-0) are snapshotted unconditionally.
struct memory_allocation_info_t
{
    size_t                 size     = 0;
    rocprofiler_agent_id_t agent_id = {.handle = 0};
    bool                   is_vmem  = false;
};

using alloc_map_t = std::unordered_map<void*, memory_allocation_info_t>;

// Enable/disable inventory population. When disabled (no replay context configured), each hook is
// a single relaxed atomic load past the chained call -- effectively free.
void
set_tracking_enabled(bool enabled);

bool
tracking_enabled();

void
record_alloc(void* ptr, size_t size, bool is_vmem);

void
record_free(void* ptr);

// Frozen copy of the inventory taken under a brief read lock. Used by memory_snapshot at snap time.
alloc_map_t
snap_inventory();

// Install inventory wrappers on top of the existing table function pointers.
void
update_table(hsa_core_table_t* table);
void
update_table(hsa_amd_ext_table_t* table);
}  // namespace memory_tracker
}  // namespace hsa
}  // namespace rocprofiler
