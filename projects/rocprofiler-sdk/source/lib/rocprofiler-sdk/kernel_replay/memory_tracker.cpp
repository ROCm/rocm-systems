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

#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_tracker
{
namespace
{
// Fast-path gate. When false the hooks only do the chained call plus this relaxed load.
std::atomic<bool>&
tracking_flag()
{
    static auto*& _v = common::static_object<std::atomic<bool>>::construct(false);
    return *_v;
}

// Tracked allocations. The map and its lock are bundled in a Synchronized wrapper so every access
// goes through rlock/wlock (no bare mutex to mismanage).
common::Synchronized<alloc_map_t>&
inventory()
{
    static auto*& _v = common::static_object<common::Synchronized<alloc_map_t>>::construct();
    return *_v;
}

// Saved "next" function pointers (the already-installed wrappers) we chain through. Types are taken
// from the HSA table members so signatures match exactly.
decltype(AmdExtTable{}.hsa_amd_memory_pool_allocate_fn) next_pool_allocate   = nullptr;
decltype(AmdExtTable{}.hsa_amd_memory_pool_free_fn)     next_pool_free       = nullptr;
decltype(CoreApiTable{}.hsa_memory_allocate_fn)         next_memory_allocate = nullptr;
decltype(CoreApiTable{}.hsa_memory_free_fn)             next_memory_free     = nullptr;

hsa_status_t
pool_allocate_wrapper(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr)
{
    auto st = next_pool_allocate(pool, size, flags, ptr);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && ptr && *ptr)
        record_alloc(*ptr, size);
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
        record_alloc(*ptr, size);
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
}  // namespace

bool
set_tracking_enabled(bool enabled)
{
    tracking_flag().store(enabled, std::memory_order_relaxed);
    return tracking_flag().load();
}

bool
tracking_enabled()
{
    return tracking_flag().load(std::memory_order_relaxed);
}

void
record_alloc(void* ptr, size_t size)
{
    inventory().wlock([ptr, size](auto& inv) { inv[ptr] = size; });
}

void
record_free(void* ptr)
{
    inventory().wlock([ptr](auto& inv) { inv.erase(ptr); });
}

alloc_map_t
snap_inventory()
{
    return inventory().get();
}

void
update_table(hsa::hsa_core_table_t* table)
{
    if(!table) return;

    next_memory_allocate          = table->hsa_memory_allocate_fn;
    next_memory_free              = table->hsa_memory_free_fn;
    table->hsa_memory_allocate_fn = memory_allocate_wrapper;
    table->hsa_memory_free_fn     = memory_free_wrapper;
}

void
update_table(hsa::hsa_amd_ext_table_t* table)
{
    if(!table) return;

    next_pool_allocate                     = table->hsa_amd_memory_pool_allocate_fn;
    next_pool_free                         = table->hsa_amd_memory_pool_free_fn;
    table->hsa_amd_memory_pool_allocate_fn = pool_allocate_wrapper;
    table->hsa_amd_memory_pool_free_fn     = pool_free_wrapper;
}
}  // namespace memory_tracker
}  // namespace kernel_replay
}  // namespace rocprofiler
