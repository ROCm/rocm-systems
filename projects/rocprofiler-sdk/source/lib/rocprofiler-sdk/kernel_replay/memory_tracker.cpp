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
#include "lib/rocprofiler-sdk/registration.hpp"

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

// Per-allocation record: byte size plus the agent that owns the memory. The agent is used to scope
// snapshots so a replay only saves/restores its own agent's device memory.
struct alloc_info_t
{
    size_t      size  = 0;
    hsa_agent_t agent = {.handle = 0};
};

using tracked_map_t = std::unordered_map<void*, alloc_info_t>;

// Tracked allocations. The map and its lock are bundled in a Synchronized wrapper so every access
// goes through rlock/wlock (no bare mutex to mismanage).
//
// Callers must gate on registration::get_fini_status() first: the HSA alloc/free wrappers we
// install stay live for the whole process, so HIP's own teardown can call the free wrapper AFTER
// this static object has been destroyed. Touching it then would lock a freed mutex and abort inside
// HIP's noexcept finalization.
common::Synchronized<tracked_map_t>&
inventory()
{
    static auto*& _v = common::static_object<common::Synchronized<tracked_map_t>>::construct();
    return *_v;
}

// Saved "next" function pointers (the already-installed wrappers) we chain through. Types are taken
// from the HSA table members so signatures match exactly.
decltype(AmdExtTable{}.hsa_amd_memory_pool_allocate_fn) next_pool_allocate   = nullptr;
decltype(AmdExtTable{}.hsa_amd_memory_pool_free_fn)     next_pool_free       = nullptr;
decltype(CoreApiTable{}.hsa_memory_allocate_fn)         next_memory_allocate = nullptr;
decltype(CoreApiTable{}.hsa_memory_free_fn)             next_memory_free     = nullptr;
decltype(AmdExtTable{}.hsa_amd_pointer_info_fn)         next_pointer_info    = nullptr;

// Per-allocation properties resolved from a single hsa_amd_pointer_info query.
struct alloc_query_t
{
    hsa_agent_t agent = {.handle = 0};  // owning (preferred-access) agent, for per-agent scoping
    bool        trackable = false;      // coarse-grained device VRAM, excluding kernarg
};

// Classify a freshly-allocated pointer. We snapshot only coarse-grained device (VRAM) memory:
//  - kernarg is excluded because it holds kernel pointer arguments; a torn/stale restore of it
//    faults the GPU -- this is the primary reason snap/restore must not touch it.
//  - fine-grained / host memory is out of scope (and precarious to restore).
// agentOwner additionally lets snapshots be scoped to the replaying agent.
alloc_query_t
query_alloc(void* ptr)
{
    alloc_query_t q = {};
    if(next_pointer_info == nullptr || ptr == nullptr) return q;

    hsa_amd_pointer_info_t info = {};
    info.size                   = sizeof(info);
    if(next_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS) return q;

    q.agent               = info.agentOwner;
    const bool is_kernarg = (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0;
    const bool is_coarse =
        (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    q.trackable = is_coarse && !is_kernarg;
    return q;
}

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
    // The HSA alloc/free wrappers outlive this tracker, so skip once rocprofiler has finalized (the
    // inventory static object may already be destroyed -- see inventory()).
    if(registration::get_fini_status() > 0) return;

    const auto q = query_alloc(ptr);
    if(!q.trackable) return;  // skip kernarg / host / fine-grained memory
    inventory().wlock([&](auto& _map) { _map[ptr] = alloc_info_t{size, q.agent}; });
}

void
record_free(void* ptr)
{
    if(registration::get_fini_status() > 0) return;
    inventory().wlock([ptr](auto& _map) { _map.erase(ptr); });
}

alloc_map_t
snap_inventory(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    alloc_map_t out{};
    inventory().rlock([&](const auto& _map) {
        for(const auto& [ptr, info] : _map)
            if(info.agent.handle == agent.handle) out.emplace(ptr, info.size);
    });
    return out;
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
    next_pointer_info                      = table->hsa_amd_pointer_info_fn;
    table->hsa_amd_memory_pool_allocate_fn = pool_allocate_wrapper;
    table->hsa_amd_memory_pool_free_fn     = pool_free_wrapper;
}
}  // namespace memory_tracker
}  // namespace kernel_replay
}  // namespace rocprofiler
