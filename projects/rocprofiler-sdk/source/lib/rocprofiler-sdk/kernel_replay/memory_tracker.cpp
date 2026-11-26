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

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <optional>

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

namespace
{
// Saved "next" function pointers (the already-installed wrappers) we chain through. Types are taken
// from the HSA table members so signatures match exactly.
decltype(AmdExtTable{}.hsa_amd_memory_pool_allocate_fn) next_pool_allocate   = nullptr;
decltype(AmdExtTable{}.hsa_amd_memory_pool_free_fn)     next_pool_free       = nullptr;
decltype(CoreApiTable{}.hsa_memory_allocate_fn)         next_memory_allocate = nullptr;
decltype(CoreApiTable{}.hsa_memory_free_fn)             next_memory_free     = nullptr;

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
        _map[ptr] = alloc_info_t{size, q.agent};
    });
}

// Inventory entries pulled out ahead of an HSA free. snap() copies from every address the
// inventory lists, so the entry has to be retired *before* the device memory is released --
// leaving it in place across the free lets a concurrent replay on another thread read a retired
// allocation. The removed values are kept so the entry can be put back when the free itself fails
// and the allocation is therefore still live.
struct retired_alloc_t
{
    std::optional<alloc_info_t> tracked{};
    std::optional<alloc_info_t> unsupported{};

    bool any() const { return tracked.has_value() || unsupported.has_value(); }
};

retired_alloc_t
retire_recorded_alloc(void* ptr)
{
    auto out = retired_alloc_t{};
    if(registration::get_fini_status() > 0) return out;

    inventory().wlock([&](auto& _map) {
        if(auto itr = _map.find(ptr); itr != _map.end())
        {
            out.tracked = itr->second;
            _map.erase(itr);
        }
    });
    unsupported_executable_inventory().wlock([&](auto& _map) {
        if(auto itr = _map.find(ptr); itr != _map.end())
        {
            out.unsupported = itr->second;
            _map.erase(itr);
        }
    });
    return out;
}

void
reinstate_recorded_alloc(void* ptr, const retired_alloc_t& prev)
{
    if(registration::get_fini_status() > 0) return;

    if(prev.tracked) inventory().wlock([&](auto& _map) { _map[ptr] = *prev.tracked; });
    if(prev.unsupported)
        unsupported_executable_inventory().wlock(
            [&](auto& _map) { _map[ptr] = *prev.unsupported; });
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
    const bool tracking = tracking_flag().load(std::memory_order_relaxed) && ptr != nullptr;
    const auto retired  = tracking ? retire_recorded_alloc(ptr) : retired_alloc_t{};

    auto st = next_pool_free(ptr);

    if(tracking && st != HSA_STATUS_SUCCESS && retired.any())
        reinstate_recorded_alloc(ptr, retired);
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
    const bool tracking = tracking_flag().load(std::memory_order_relaxed) && ptr != nullptr;
    const auto retired  = tracking ? retire_recorded_alloc(ptr) : retired_alloc_t{};

    auto st = next_memory_free(ptr);

    if(tracking && st != HSA_STATUS_SUCCESS && retired.any())
        reinstate_recorded_alloc(ptr, retired);
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
    unsupported_executable_inventory().wlock([ptr](auto& _map) { _map.erase(ptr); });
}

alloc_map_t
snap_inventory(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    alloc_map_t out{};
    inventory().rlock([&](const auto& _map) {
        for(const auto& [ptr, info] : _map)
            if(info.agent == agent) out.emplace(ptr, info.size);
    });
    return out;
}

alloc_map_t
unsupported_executable(hsa_agent_t agent)
{
    if(registration::get_fini_status() > 0) return {};

    alloc_map_t out{};
    unsupported_executable_inventory().rlock([&](const auto& _map) {
        for(const auto& [ptr, info] : _map)
            if(info.agent == agent) out.emplace(ptr, info.size);
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
            if(info.agent == agent)
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

}  // namespace memory_tracker

void
memory_tracker_init(hsa::hsa_core_table_t* table, uint64_t lib_instance)
{
    if(!table) return;

    // Install only for the first library instance. A later instance would capture our own wrapper
    // as next_memory_allocate and recurse. (Keying on lib_instance matches the copy/update_table
    // convention -- see scratch_memory.cpp.)
    if(lib_instance > 0) return;

    // Idempotent on this table: restore_table may call memory_tracker_init again after putting the
    // original pointers back; std::call_once would skip that re-hook. Guard against a second
    // install on an already-wrapped table (which would capture our wrapper as next_* and recurse).
    if(table->hsa_memory_allocate_fn == memory_tracker::memory_allocate_wrapper) return;

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

    // Idempotent on this table -- see memory_tracker_init(hsa_core_table_t*) above.
    if(table->hsa_amd_memory_pool_allocate_fn == memory_tracker::pool_allocate_wrapper) return;

    memory_tracker::next_pool_allocate     = table->hsa_amd_memory_pool_allocate_fn;
    memory_tracker::next_pool_free         = table->hsa_amd_memory_pool_free_fn;
    table->hsa_amd_memory_pool_allocate_fn = memory_tracker::pool_allocate_wrapper;
    table->hsa_amd_memory_pool_free_fn     = memory_tracker::pool_free_wrapper;
}
}  // namespace kernel_replay
}  // namespace rocprofiler
