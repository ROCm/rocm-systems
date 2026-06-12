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

#include "lib/rocprofiler-sdk/hsa/memory_tracker.hpp"

#include "lib/common/logging.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>

namespace rocprofiler
{
namespace hsa
{
namespace memory_tracker
{
namespace
{
// Fast-path gate. When false (no replay context configured), the hooks only do a chained call plus
// this relaxed load. Matches design Section 4.2: "If no replay context is active, cost is a single
// null-check."
std::atomic<bool>&
tracking_flag()
{
    static std::atomic<bool> _v{false};
    return _v;
}

std::shared_mutex&
inventory_mutex()
{
    static std::shared_mutex _v;
    return _v;
}

alloc_map_t&
inventory()
{
    static alloc_map_t _v;
    return _v;
}

// Saved "next" function pointers (the already-installed tracing wrappers) we chain through. Types
// taken from the HSA table members so signatures match exactly.
decltype(AmdExtTable{}.hsa_amd_memory_pool_allocate_fn) next_pool_allocate = nullptr;
decltype(AmdExtTable{}.hsa_amd_memory_pool_free_fn)     next_pool_free     = nullptr;
decltype(AmdExtTable{}.hsa_amd_vmem_map_fn)             next_vmem_map      = nullptr;
decltype(AmdExtTable{}.hsa_amd_vmem_unmap_fn)           next_vmem_unmap    = nullptr;
decltype(CoreApiTable{}.hsa_memory_allocate_fn)         next_memory_allocate = nullptr;
decltype(CoreApiTable{}.hsa_memory_free_fn)             next_memory_free     = nullptr;

hsa_status_t
pool_allocate_wrapper(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr)
{
    auto st = next_pool_allocate(pool, size, flags, ptr);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && ptr && *ptr)
        record_alloc(*ptr, size, /*is_vmem=*/false);
    return st;
}

hsa_status_t
pool_free_wrapper(void* ptr)
{
    auto st = next_pool_free(ptr);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && ptr)
        record_free(ptr);
    return st;
}

hsa_status_t
memory_allocate_wrapper(hsa_region_t region, size_t size, void** ptr)
{
    auto st = next_memory_allocate(region, size, ptr);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && ptr && *ptr)
        record_alloc(*ptr, size, /*is_vmem=*/false);
    return st;
}

hsa_status_t
memory_free_wrapper(void* ptr)
{
    auto st = next_memory_free(ptr);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && ptr)
        record_free(ptr);
    return st;
}

// VMEM: the usable VA is established at map time (not at handle creation), so the map/unmap pair is
// the third allocator path we must track (Kerncap kerncap.hip:613-644).
hsa_status_t
vmem_map_wrapper(void*                       va,
                 size_t                      size,
                 size_t                      in_offset,
                 hsa_amd_vmem_alloc_handle_t memory_handle,
                 uint64_t                    flags)
{
    auto st = next_vmem_map(va, size, in_offset, memory_handle, flags);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && va)
        record_alloc(va, size, /*is_vmem=*/true);
    return st;
}

hsa_status_t
vmem_unmap_wrapper(void* va, size_t size)
{
    auto st = next_vmem_unmap(va, size);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && va)
        record_free(va);
    return st;
}
}  // namespace

void
set_tracking_enabled(bool enabled)
{
    tracking_flag().store(enabled, std::memory_order_relaxed);
}

bool
tracking_enabled()
{
    return tracking_flag().load(std::memory_order_relaxed);
}

void
record_alloc(void* ptr, size_t size, bool is_vmem)
{
    auto _lk = std::unique_lock<std::shared_mutex>{inventory_mutex()};
    inventory()[ptr] =
        memory_allocation_info_t{.size = size, .agent_id = {.handle = 0}, .is_vmem = is_vmem};
}

void
record_free(void* ptr)
{
    auto _lk = std::unique_lock<std::shared_mutex>{inventory_mutex()};
    inventory().erase(ptr);
}

alloc_map_t
snap_inventory()
{
    auto _lk = std::shared_lock<std::shared_mutex>{inventory_mutex()};
    return inventory();
}

void
update_table(hsa_core_table_t* table)
{
    if(!table) return;

    next_memory_allocate          = table->hsa_memory_allocate_fn;
    next_memory_free              = table->hsa_memory_free_fn;
    table->hsa_memory_allocate_fn = memory_allocate_wrapper;
    table->hsa_memory_free_fn     = memory_free_wrapper;
}

void
update_table(hsa_amd_ext_table_t* table)
{
    if(!table) return;

    next_pool_allocate                   = table->hsa_amd_memory_pool_allocate_fn;
    next_pool_free                       = table->hsa_amd_memory_pool_free_fn;
    table->hsa_amd_memory_pool_allocate_fn = pool_allocate_wrapper;
    table->hsa_amd_memory_pool_free_fn     = pool_free_wrapper;

    // VMEM hooks may be absent on older HSA tables; only wrap if present.
    if(table->hsa_amd_vmem_map_fn)
    {
        next_vmem_map          = table->hsa_amd_vmem_map_fn;
        table->hsa_amd_vmem_map_fn = vmem_map_wrapper;
    }
    if(table->hsa_amd_vmem_unmap_fn)
    {
        next_vmem_unmap          = table->hsa_amd_vmem_unmap_fn;
        table->hsa_amd_vmem_unmap_fn = vmem_unmap_wrapper;
    }
}
}  // namespace memory_tracker
}  // namespace hsa
}  // namespace rocprofiler
