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

#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/utils.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstdint>
#include <utility>

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
    static auto*& _v = rocprofiler::common::static_object<std::atomic<bool>>::construct(false);
    return *_v;
}
}  // namespace

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

common::Synchronized<tracked_map_t>&
unsupported_executable_inventory()
{
    // Distinct ContextT so this singleton does not share storage with inventory().
    struct unsupported_executable_tag
    {};
    static auto*& _v = common::static_object<common::Synchronized<tracked_map_t>,
                                            unsupported_executable_tag>::construct();
    return *_v;
}

// Live hsa_amd_vmem_map ranges. Never restored -- the snapshot has no way to reason about a range
// that can be partially unmapped or remapped to a different physical handle mid-window -- but
// counted, because their presence means application data is outside the snapshot. See
// untracked_summary_t.
common::Synchronized<tracked_map_t>&
vmem_inventory()
{
    struct vmem_tag
    {};
    static auto*& _v =
        common::static_object<common::Synchronized<tracked_map_t>, vmem_tag>::construct();
    return *_v;
}

// GPU-owned, non-kernarg pool allocations that failed the coarse-grained test: fine-grained device
// memory and managed memory that landed on the device. Counted for the same reason as
// vmem_inventory(), but reported separately because runtime-internal allocations can land here and
// so a non-zero count is weaker evidence (see untracked_summary_t).
common::Synchronized<tracked_map_t>&
untracked_pool_inventory()
{
    struct untracked_pool_tag
    {};
    static auto*& _v =
        common::static_object<common::Synchronized<tracked_map_t>, untracked_pool_tag>::construct();
    return *_v;
}

namespace
{
// Monotonic stamp handed to each allocation so (base, size) alone cannot be mistaken for identity.
// Starts at 1 so a default-constructed alloc_info_t (generation 0) never compares equal to a real
// record. Never reused, including across free: that is the whole point.
std::atomic<uint64_t>&
generation_counter()
{
    struct generation_tag
    {};
    static auto*& _v =
        common::static_object<std::atomic<uint64_t>, generation_tag>::construct(uint64_t{0});
    return *_v;
}

// Saved "next" function pointers (the already-installed wrappers) we chain through. Types are taken
// from the HSA table members so signatures match exactly.
decltype(AmdExtTable{}.hsa_amd_memory_pool_allocate_fn) next_pool_allocate   = nullptr;
decltype(AmdExtTable{}.hsa_amd_memory_pool_free_fn)     next_pool_free       = nullptr;
decltype(CoreApiTable{}.hsa_memory_allocate_fn)         next_memory_allocate = nullptr;
decltype(CoreApiTable{}.hsa_memory_free_fn)             next_memory_free     = nullptr;
decltype(AmdExtTable{}.hsa_amd_vmem_map_fn)             next_vmem_map        = nullptr;
decltype(AmdExtTable{}.hsa_amd_vmem_unmap_fn)           next_vmem_unmap      = nullptr;

// HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG is absent from older HSA headers
constexpr uint32_t memory_pool_executable_flag = (1U << 2);

void
record_unsupported_executable(void* ptr, size_t size)
{
    // Same fini gate as record_alloc: the wrappers outlive this static object.
    if(registration::get_fini_status() > 0) return;

    const auto q = query_alloc(ptr);
    // Only ordinary coarse device VRAM. HIP kernarg pools / fine-grained regions fail trackable and
    // stay out of both inventories (silent omit is correct for those).
    if(!q.trackable) return;
    unsupported_executable_inventory().wlock([&](auto& _map) {
        _map[ptr] = alloc_info_t{size, q.agent, 0};
    });
}

// Record a pool allocation that is GPU-resident but not snapshottable, so the replay window can
// report how much application data the snapshot is missing. Called only when query_alloc already
// said the allocation is not trackable, so this never competes with the main inventory.
void
record_untracked_pool(void* ptr, size_t size, const alloc_query_t& q)
{
    if(registration::get_fini_status() > 0) return;
    if(!q.untracked_device_visible()) return;

    untracked_pool_inventory().wlock([&](auto& _map) {
        _map[ptr] = alloc_info_t{size, q.agent, 0};
    });
}

// Sum bytes/regions of one inventory for a single agent under a brief read lock.
std::pair<size_t, size_t>
sum_for_agent(common::Synchronized<tracked_map_t>& inv, hsa_agent_t agent)
{
    size_t bytes   = 0;
    size_t regions = 0;
    inv.rlock([&](const tracked_map_t& _map) {
        for(const auto& [ptr, info] : _map)
        {
            (void) ptr;
            if(info.agent.handle != agent.handle) continue;
            bytes += info.size;
            ++regions;
        }
    });
    return {bytes, regions};
}

hsa_status_t
pool_allocate_wrapper(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr)
{
    auto st = next_pool_allocate(pool, size, flags, ptr);
    // Never snapshot executable allocations into the main inventory. HIP places its
    // per-stream/per-graph kernarg pools -- and rocprofiler its trace buffers -- in the
    // coarse-grained segment with the executable flag, so they slip past query_alloc's
    // kernarg-pool check. They hold live kernel arguments / runtime state, not application
    // data; snapshotting them means restore() clobbers the in-flight kernargs of a concurrent
    // dispatch. We already have the flag here, so skip the main inventory before query_alloc.
    //
    // Direct-HSA apps can put ordinary writable device data behind the same flag. Those
    // allocations are trackable (coarse VRAM); record them in the unsupported side inventory so
    // tests and tools can detect the unsupported omitted class. Declining snap() whenever the side
    // inventory is non-empty is not viable -- HIP and the SDK's own AQL pools routinely keep such
    // allocations live -- so they remain documented as an unsupported omission for beta.
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && ptr && *ptr)
    {
        const bool is_executable = (flags & memory_pool_executable_flag) != 0;
        if(is_executable)
            record_unsupported_executable(*ptr, size);
        else
            record_alloc(*ptr, size);
    }
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

// hsa_amd_vmem_map is the entry point every virtual-memory allocator funnels through:
// hipMallocAsync with the default (VM-backed) pool, hipMemAddressReserve/hipMemMap, PyTorch's
// expandable_segments, and Kokkos when built with KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC. None of it
// reaches hsa_amd_memory_pool_allocate, so without this hook a snapshot of such an application
// captures almost nothing and reports success. We only count the mapping -- restoring a range whose
// physical backing can be swapped underneath us is not something snap/restore can reason about.
hsa_status_t
vmem_map_wrapper(void*                       va,
                 size_t                      size,
                 size_t                      in_offset,
                 hsa_amd_vmem_alloc_handle_t memory_handle,
                 uint64_t                    flags)
{
    auto st = next_vmem_map(va, size, in_offset, memory_handle, flags);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && va)
        record_vmem_map(va, size, query_alloc(va).agent);
    return st;
}

hsa_status_t
vmem_unmap_wrapper(void* va, size_t size)
{
    auto st = next_vmem_unmap(va, size);
    if(tracking_flag().load(std::memory_order_relaxed) && st == HSA_STATUS_SUCCESS && va)
        record_vmem_unmap(va);
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
    if(!q.trackable)
    {
        // Not ours to snapshot. Host and kernarg memory is simply out of scope, but GPU-resident
        // memory we cannot capture makes replay unsound, so count that separately.
        record_untracked_pool(ptr, size, q);
        return;
    }
    const auto generation = generation_counter().fetch_add(1, std::memory_order_relaxed) + 1;
    inventory().wlock([&](auto& _map) { _map[ptr] = alloc_info_t{size, q.agent, generation}; });
}

void
record_free(void* ptr)
{
    if(registration::get_fini_status() > 0) return;
    inventory().wlock([ptr](auto& _map) { _map.erase(ptr); });
    unsupported_executable_inventory().wlock([ptr](auto& _map) { _map.erase(ptr); });
    untracked_pool_inventory().wlock([ptr](auto& _map) { _map.erase(ptr); });
}

void
record_vmem_map(void* va, size_t size, hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return;
    vmem_inventory().wlock([&](auto& _map) { _map[va] = alloc_info_t{size, agent, 0}; });
}

void
record_vmem_unmap(void* va)
{
    if(registration::get_fini_status() > 0) return;
    vmem_inventory().wlock([va](auto& _map) { _map.erase(va); });
}

tracked_map_t
snap_inventory(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    tracked_map_t out{};
    inventory().rlock([&](const auto& _map) {
        for(const auto& [ptr, info] : _map)
            if(info.agent.handle == agent.handle) out.emplace(ptr, info);
    });
    return out;
}

footprint_t
tracked_footprint(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    const auto [bytes, regions] = sum_for_agent(inventory(), agent);
    return footprint_t{bytes, regions};
}

untracked_summary_t
untracked_device_memory(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    auto out = untracked_summary_t{};
    // A vmem mapping whose owning agent could not be resolved (handle 0) is still evidence that the
    // application is using virtual memory, and attributing it to no agent would hide it from every
    // replay window. Count it against whichever agent asks.
    vmem_inventory().rlock([&](const tracked_map_t& _map) {
        for(const auto& [ptr, info] : _map)
        {
            (void) ptr;
            if(info.agent.handle != agent.handle && info.agent.handle != 0) continue;
            out.vmem_bytes += info.size;
            ++out.vmem_regions;
        }
    });

    const auto [pool_bytes, pool_regions] = sum_for_agent(untracked_pool_inventory(), agent);
    out.pool_bytes                        = pool_bytes;
    out.pool_regions                      = pool_regions;
    return out;
}

alloc_map_t
unsupported_executable(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    alloc_map_t out{};
    unsupported_executable_inventory().rlock([&](const auto& _map) {
        for(const auto& [ptr, info] : _map)
            if(info.agent.handle == agent.handle) out.emplace(ptr, info.size);
    });
    return out;
}

bool
agent_has_unsupported_executable(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return false;

    bool found = false;
    unsupported_executable_inventory().rlock([&](const auto& _map) {
        for(const auto& [ptr, info] : _map)
        {
            (void) ptr;
            if(info.agent.handle == agent.handle)
            {
                found = true;
                break;
            }
        }
    });
    return found;
}

hsa_status_t
tracking_pool_allocate(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr)
{
    if(!next_pool_allocate) return HSA_STATUS_ERROR;
    return pool_allocate_wrapper(pool, size, flags, ptr);
}

hsa_status_t
tracking_pool_free(void* ptr)
{
    if(!next_pool_free) return HSA_STATUS_ERROR;
    return pool_free_wrapper(ptr);
}

uint64_t
generation_of(void* ptr)
{
    if(registration::get_fini_status() > 0) return 0;

    uint64_t generation = 0;
    inventory().rlock([&](const tracked_map_t& _map) {
        auto itr = _map.find(ptr);
        if(itr != _map.end()) generation = itr->second.generation;
    });
    return generation;
}

}  // namespace memory_tracker

void
memory_tracker_init(hsa::hsa_core_table_t* table, uint64_t lib_instance)
{
    if(!table) return;

    // Install only for the first library instance. A later instance would capture our own wrapper
    // as next_memory_allocate and recurse. (Keying on lib_instance matches the copy/update_table
    // convention -- see scratch_memory.cpp.)
    if(lib_instance > 0) return;

    memory_tracker::next_memory_allocate = table->hsa_memory_allocate_fn;
    memory_tracker::next_memory_free     = table->hsa_memory_free_fn;
    table->hsa_memory_allocate_fn        = memory_tracker::memory_allocate_wrapper;
    table->hsa_memory_free_fn            = memory_tracker::memory_free_wrapper;
}

void
memory_tracker_init(hsa::hsa_amd_ext_table_t* table, uint64_t lib_instance)
{
    if(!table) return;

    // Install only for the first library instance. A later instance would capture our own wrapper
    // as next_pool_allocate and recurse. (Keying on lib_instance matches the copy/update_table
    // convention -- see scratch_memory.cpp.)
    if(lib_instance > 0) return;

    memory_tracker::next_pool_allocate     = table->hsa_amd_memory_pool_allocate_fn;
    memory_tracker::next_pool_free         = table->hsa_amd_memory_pool_free_fn;
    table->hsa_amd_memory_pool_allocate_fn = memory_tracker::pool_allocate_wrapper;
    table->hsa_amd_memory_pool_free_fn     = memory_tracker::pool_free_wrapper;

    // Virtual-memory map/unmap. Only counted, never snapshotted (see vmem_map_wrapper). Guard on
    // the table entries being present: an older ROCr exposes a shorter AmdExtTable and leaves these
    // null, in which case the accounting degrades to query_alloc's pointer-type check alone.
    if(table->hsa_amd_vmem_map_fn != nullptr && table->hsa_amd_vmem_unmap_fn != nullptr)
    {
        memory_tracker::next_vmem_map   = table->hsa_amd_vmem_map_fn;
        memory_tracker::next_vmem_unmap = table->hsa_amd_vmem_unmap_fn;
        table->hsa_amd_vmem_map_fn      = memory_tracker::vmem_map_wrapper;
        table->hsa_amd_vmem_unmap_fn    = memory_tracker::vmem_unmap_wrapper;
    }
}
}  // namespace kernel_replay
}  // namespace rocprofiler
