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

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/snapshot-plan.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/snapshot-storage.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace
{
using blit::copy_region_t;

hsa_status_t
dma_copy(void* dst, const void* src, size_t n)
{
    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_memory_copy_fn) return HSA_STATUS_ERROR;
    return core->hsa_memory_copy_fn(dst, src, n);
}

/// @brief Run @p copy under the tracker read lock, but only while [@p gpu_addr, +@p size) is still
/// a live tracked allocation of >= @p size bytes, so a concurrent free (write lock) can't retire it
/// mid-copy. Direction is caller-supplied: snap reads device->host, restore writes host->device.
/// @return @p copy's status, or @c std::nullopt if the region was freed/shrunk since it was
/// recorded.
template <typename CopyFn>
std::optional<hsa_status_t>
with_inventory_check(void* gpu_addr, size_t size, CopyFn&& copy)
{
    return memory_tracker::inventory().rlock(
        [&](const memory_tracker::tracked_map_t& map) -> std::optional<hsa_status_t> {
            auto itr = map.find(gpu_addr);
            if(itr == map.end() || itr->second.size < size) return std::nullopt;
            return copy();
        });
}

// A module-scope variable (__device__ / __constant__ global) discovered in a loaded executable.
struct module_variable_t
{
    void*  gpu_addr = nullptr;
    size_t size     = 0;
};

// Upper bound on a single module-scope variable the snapshot will capture. This guards against a
// mis-reported HSA symbol size turning into a huge host allocation; it is not a supported limit,
// and exceeding it is reported rather than ignored (see collect_module_variable).
constexpr uint64_t module_variable_size_cap = 1ULL << 30;  // 1 GiB

// Result of enumerating module-scope variables. `incomplete` means HSA could not be asked about at
// least one executable or symbol, so the set below may be missing a writable __device__ global.
// Treated as a failed snapshot rather than a partial one: a variable we never captured is a
// variable we never restore, and passes 2..N would silently read state accumulated by pass 1.
struct module_variable_scan_t
{
    std::vector<module_variable_t> found{};
    bool                           incomplete = false;
};

// hsa_executable_iterate_agent_symbols callback: collect HSA_SYMBOL_KIND_VARIABLE symbols
// (device address + size) into the scan passed via `data`. The HSA callback cannot capture, so
// state is threaded through the void* argument. A query failure marks the scan incomplete and
// keeps iterating, so one bad symbol does not hide the rest.
hsa_status_t
collect_module_variable(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t symbol, void* data)
{
    auto* out  = static_cast<module_variable_scan_t*>(data);
    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_executable_symbol_get_info_fn)
    {
        out->incomplete = true;
        return HSA_STATUS_SUCCESS;
    }

    hsa_symbol_kind_t kind{};
    if(core->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) !=
       HSA_STATUS_SUCCESS)
    {
        out->incomplete = true;
        return HSA_STATUS_SUCCESS;
    }

    if(kind != HSA_SYMBOL_KIND_VARIABLE) return HSA_STATUS_SUCCESS;

    uint64_t addr = 0;
    uint32_t size = 0;
    if(core->hsa_executable_symbol_get_info_fn(
           symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr) != HSA_STATUS_SUCCESS ||
       core->hsa_executable_symbol_get_info_fn(
           symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &size) != HSA_STATUS_SUCCESS)
    {
        out->incomplete = true;
        return HSA_STATUS_SUCCESS;
    }

    // HIP emits a one-byte compilation-unit marker into every code object. It is linker/runtime
    // metadata rather than application state, and snapshotting it would make the replay-owned blit
    // code object recursively add another restore region.
    if(size == 1)
    {
        constexpr auto hip_cuid_prefix = std::string_view{"__hip_cuid_"};
        uint32_t       name_length     = 0;
        if(core->hsa_executable_symbol_get_info_fn(symbol,
                                                   HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                                                   &name_length) == HSA_STATUS_SUCCESS &&
           name_length >= hip_cuid_prefix.size())
        {
            auto name = std::vector<char>(name_length + 1, '\0');
            if(core->hsa_executable_symbol_get_info_fn(
                   symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()) == HSA_STATUS_SUCCESS &&
               std::string_view{name.data()}.substr(0, hip_cuid_prefix.size()) == hip_cuid_prefix)
                return HSA_STATUS_SUCCESS;
        }
    }

    if(addr == 0 || size == 0) return HSA_STATUS_SUCCESS;

    // A variable above the cap is skipped, which means a kernel's writes to it leak across replay
    // passes and passes 2..N see mutated inputs. That is a wrong-counters outcome, so it cannot be
    // silent -- warn rather than drop it on the floor. The cap itself is a sanity bound against a
    // mis-reported symbol size, not a supported limit.
    if(size > module_variable_size_cap)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kernel-replay snapshot: module-scope variable at 0x{:x} is {} bytes, above the {} "
            "byte "
            "per-variable cap, and is not captured. A kernel writing to it will observe values "
            "accumulated across replay passes instead of identical inputs.",
            addr,
            size,
            module_variable_size_cap);
        return HSA_STATUS_SUCCESS;
    }

    // HSA reports the variable's device address as an integer; converting to a pointer is required.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    out->found.push_back(module_variable_t{reinterpret_cast<void*>(addr), size});
    return HSA_STATUS_SUCCESS;
}

// Enumerate module-scope variables visible to `agent` across all loaded executables. They live in
// the executable's data segment -- not in the allocation tracker's inventory -- so a kernel that
// mutates a __device__ global would otherwise leak that mutation across replay passes. Must run at
// snap time (not executable-load time): constant memory may not be populated at load.
module_variable_scan_t
discover_module_variables(hsa_agent_t agent)
{
    auto scan = module_variable_scan_t{};

    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_executable_iterate_agent_symbols_fn)
    {
        scan.incomplete = true;
        return scan;
    }

    code_object::iterate_loaded_code_objects([&](const code_object::hsa::code_object& co) {
        // Iterating for `agent` naturally scopes to executables loaded on this agent (others yield
        // no symbols), matching snap()'s per-agent contract.
        if(core->hsa_executable_iterate_agent_symbols_fn(
               co.hsa_executable, agent, collect_module_variable, &scan) != HSA_STATUS_SUCCESS)
            scan.incomplete = true;
    });
    return scan;
}

// At most one retained high-water arena per pool. A larger returned arena replaces and frees the
// old one, and smaller returns are freed. A snapshot that needs more than the retained arena frees
// it before allocating, so retained memory never exceeds the largest snapshot so far.
using retained_backing_t = std::unordered_map<uint64_t, block_t>;

common::Synchronized<retained_backing_t>&
retained_backing()
{
    // Keep backing allocations for process lifetime. HSA may already be unavailable during static
    // destruction, so do not attempt to free them there.
    static auto* value = new common::Synchronized<retained_backing_t>{};
    return *value;
}

void
free_pool_block(block_t block) noexcept
{
    if(!block) return;
    if(auto* ext = hsa::get_amd_ext_table(); ext && ext->hsa_amd_memory_pool_free_fn)
        ext->hsa_amd_memory_pool_free_fn(block.data);
}

// Returns the pool's retained arena when it can hold `size`. One that is too small is freed
// instead, so the larger allocation that follows can use its memory.
block_t
take_retained_block(hsa_amd_memory_pool_t pool, size_t size) noexcept
{
    if(pool.handle == 0 || size == 0) return {};
    auto taken   = block_t{};
    auto evicted = block_t{};
    try
    {
        retained_backing().wlock([&](auto& retained) {
            auto itr = retained.find(pool.handle);
            if(itr == retained.end() || !itr->second) return;
            auto& out = itr->second.size >= size ? taken : evicted;
            out       = std::exchange(itr->second, block_t{});
        });
    } catch(...)
    {
        return {};
    }
    free_pool_block(evicted);
    return taken;
}

size_t
retained_capacity(hsa_amd_memory_pool_t pool) noexcept
{
    if(pool.handle == 0) return 0;
    try
    {
        return retained_backing().rlock([&](const auto& retained) {
            auto itr = retained.find(pool.handle);
            return itr == retained.end() ? size_t{0} : itr->second.size;
        });
    } catch(...)
    {
        return 0;
    }
}

void
retain_block(hsa_amd_memory_pool_t pool, block_t block) noexcept
{
    if(pool.handle == 0 || !block)
    {
        free_pool_block(block);
        return;
    }

    // Map insertion may allocate. Destruction must remain noexcept, so retain nothing if the map
    // cannot grow and release the arena directly.
    try
    {
        retained_backing().wlock([&](auto& retained) {
            auto& current = retained[pool.handle];
            if(!current || block.size > current.size) std::swap(current, block);
        });
    } catch(...)
    {}
    free_pool_block(block);
}

constexpr size_t arena_alignment = 16;

std::optional<size_t>
pool_allocation_max(hsa_amd_memory_pool_t pool)
{
    auto* ext = hsa::get_amd_ext_table();
    if(pool.handle == 0 || !ext || !ext->hsa_amd_memory_pool_get_info_fn) return std::nullopt;

    bool allowed = false;
    if(ext->hsa_amd_memory_pool_get_info_fn(
           pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed) != HSA_STATUS_SUCCESS ||
       !allowed)
        return std::nullopt;

    size_t maximum = 0;
    if(ext->hsa_amd_memory_pool_get_info_fn(
           pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &maximum) != HSA_STATUS_SUCCESS)
        return std::nullopt;
    return maximum;
}

bool
pool_accepts(hsa_amd_memory_pool_t pool, size_t size)
{
    auto maximum = pool_allocation_max(pool);
    return maximum && size <= *maximum;
}

std::optional<size_t>
pool_size(hsa_amd_memory_pool_t pool)
{
    auto* ext = hsa::get_amd_ext_table();
    if(pool.handle == 0 || !ext || !ext->hsa_amd_memory_pool_get_info_fn) return std::nullopt;

    size_t total = 0;
    if(ext->hsa_amd_memory_pool_get_info_fn(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &total) !=
       HSA_STATUS_SUCCESS)
        return std::nullopt;
    return total;
}

std::optional<uint64_t>
gpu_available_bytes(hsa_agent_t agent)
{
    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_agent_get_info_fn) return std::nullopt;

    uint64_t   available = 0;
    const auto status    = core->hsa_agent_get_info_fn(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_AVAIL), &available);
    if(status != HSA_STATUS_SUCCESS) return std::nullopt;
    return available;
}

std::optional<uint64_t>
host_available_bytes()
{
    auto input = std::ifstream{"/proc/meminfo"};
    if(!input) return std::nullopt;

    auto     key       = std::string{};
    auto     unit      = std::string{};
    uint64_t value_kib = 0;
    while(input >> key >> value_kib >> unit)
    {
        if(key != "MemAvailable:") continue;
        if(unit != "kB" || value_kib > std::numeric_limits<uint64_t>::max() / 1024)
            return std::nullopt;
        return value_kib * 1024;
    }
    return std::nullopt;
}

size_t
gpu_plan_budget(const capture_context_t& ctx)
{
    if(ctx.mode == capture_mode::force_pinned || !ctx.gpu_backend_available) return 0;
    auto maximum = pool_allocation_max(ctx.gpu_pool);
    if(!maximum) return 0;

    // Hold at most half of the device's memory, and leave the application at least 5% of what is
    // free. The retained arena counts as available because a larger snapshot frees it first.
    auto budget = static_cast<uint64_t>(*maximum);
    if(auto total = pool_size(ctx.gpu_pool)) budget = std::min<uint64_t>(budget, *total / 2);
    if(auto available = gpu_available_bytes(ctx.agent))
    {
        const auto retained = static_cast<uint64_t>(retained_capacity(ctx.gpu_pool));
        budget              = std::min(budget, retained + (*available - *available / 20));
    }
    return static_cast<size_t>(budget);
}

size_t
host_plan_budget(const capture_context_t& ctx)
{
    if(ctx.mode == capture_mode::force_gpu) return 0;
    auto maximum = pool_allocation_max(ctx.pinned_pool);
    if(!maximum) return 0;
    // Pinned memory cannot be swapped out, so use at most half of what the host has available. The
    // retained arena counts as available, as on the GPU.
    if(auto available = host_available_bytes())
        return std::min(*maximum,
                        retained_capacity(ctx.pinned_pool) + static_cast<size_t>(*available / 2));
    return *maximum;
}

// Allocates one tier's backing from an HSA pool. The returned block releases itself according to
// its retain mode.
struct hsa_pool_allocator
{
    using handle_type = owned_block_t;

    storage_kind          kind       = storage_kind::none;
    retain_mode           retain     = retain_mode::per_replay;
    hsa_amd_memory_pool_t pool       = {};
    hsa_agent_t           agent      = {};
    hsa_agent_t           peer_agent = {};
    bool                  available  = false;

    std::optional<owned_block_t> allocate(size_t size) const
    {
        if(!available || pool.handle == 0 || size == 0) return std::nullopt;

        // A retained arena that is large enough consumes no fresh capacity. Consult it before
        // asking whether a new allocation of this size is currently possible.
        if(retain == retain_mode::across_replays)
        {
            if(auto retained = take_retained_block(pool, size); retained)
                return owned_block_t{retained, kind, pool, retain};
        }

        if(!pool_accepts(pool, size)) return std::nullopt;

        auto* ext = hsa::get_amd_ext_table();
        if(!ext || !ext->hsa_amd_memory_pool_allocate_fn || !ext->hsa_amd_agents_allow_access_fn ||
           !ext->hsa_amd_memory_pool_free_fn)
            return std::nullopt;

        void* ptr    = nullptr;
        auto  status = ext->hsa_amd_memory_pool_allocate_fn(pool, size, 0, &ptr);
        if(status != HSA_STATUS_SUCCESS || !ptr) return std::nullopt;

        if(peer_agent.handle != 0)
        {
            status = ext->hsa_amd_agents_allow_access_fn(1, &peer_agent, nullptr, ptr);
            if(status != HSA_STATUS_SUCCESS)
            {
                ext->hsa_amd_memory_pool_free_fn(ptr);
                return std::nullopt;
            }
        }

        status = ext->hsa_amd_agents_allow_access_fn(1, &agent, nullptr, ptr);
        if(status == HSA_STATUS_SUCCESS)
            return owned_block_t{block_t{size, ptr}, kind, pool, retain};

        ext->hsa_amd_memory_pool_free_fn(ptr);
        return std::nullopt;
    }
};

using hsa_tier =
    storage::arena<planning::whole_region, storage::bump_suballocator<hsa_pool_allocator>>;
using snapshot_storage = storage::fallback<hsa_tier, hsa_tier, storage::null_tier>;

snapshot_storage
make_snapshot_storage(const capture_context_t& ctx, storage_budgets_t budgets)
{
    const auto gpu    = hsa_pool_allocator{.kind      = storage_kind::gpu_local,
                                        .retain    = retain_mode::across_replays,
                                        .pool      = ctx.gpu_pool,
                                        .agent     = ctx.agent,
                                        .available = ctx.gpu_backend_available};
    const auto pinned = hsa_pool_allocator{.kind       = storage_kind::pinned_host,
                                           .retain     = retain_mode::across_replays,
                                           .pool       = ctx.pinned_pool,
                                           .agent      = ctx.agent,
                                           .peer_agent = ctx.cpu_agent,
                                           .available  = true};
    return snapshot_storage{
        hsa_tier{{}, {budgets.gpu_bytes, arena_alignment, gpu}},
        hsa_tier{{}, {budgets.host_bytes, arena_alignment, pinned}},
        storage::null_tier{},
    };
}

bool
region_is_live(const snapshot_region_t& region, const memory_tracker::tracked_map_t& inventory)
{
    if(region.liveness == liveness_kind::module_variable) return true;
    if(region.liveness == liveness_kind::retired) return false;
    auto itr = inventory.find(region.live_address);
    return itr != inventory.end() && itr->second.size >= region.size;
}

copy_region_t
capture_copy(const snapshot_t& snapshot, const snapshot_extent_t& extent)
{
    const auto& region  = snapshot.regions.at(extent.region_index);
    const auto& segment = snapshot.segments.at(extent.segment_index);
    auto*       backing = static_cast<std::byte*>(segment.backing.block().data);
    return copy_region_t{
        extent.size,
        static_cast<std::byte*>(region.live_address) + extent.region_offset,
        backing + extent.segment_offset,
    };
}

copy_region_t
restore_copy(const snapshot_t& snapshot, const snapshot_extent_t& extent)
{
    const auto& region  = snapshot.regions.at(extent.region_index);
    const auto& segment = snapshot.segments.at(extent.segment_index);
    auto*       backing = static_cast<std::byte*>(segment.backing.block().data);
    return copy_region_t{
        extent.size,
        backing + extent.segment_offset,
        static_cast<std::byte*>(region.live_address) + extent.region_offset,
    };
}
}  // namespace

owned_block_t::owned_block_t(block_t               block,
                             storage_kind          kind,
                             hsa_amd_memory_pool_t pool,
                             retain_mode           retain)
: m_block{block}
, m_kind{kind}
, m_pool{pool}
, m_retain{retain}
{}

owned_block_t::~owned_block_t() { reset(); }

owned_block_t::owned_block_t(owned_block_t&& rhs) noexcept
: m_block{std::exchange(rhs.m_block, block_t{})}
, m_kind{std::exchange(rhs.m_kind, storage_kind::none)}
, m_pool{std::exchange(rhs.m_pool, hsa_amd_memory_pool_t{.handle = 0})}
, m_retain{std::exchange(rhs.m_retain, retain_mode::per_replay)}
{}

owned_block_t&
owned_block_t::operator=(owned_block_t&& rhs) noexcept
{
    if(this == &rhs) return *this;
    reset();
    m_block  = std::exchange(rhs.m_block, block_t{});
    m_kind   = std::exchange(rhs.m_kind, storage_kind::none);
    m_pool   = std::exchange(rhs.m_pool, hsa_amd_memory_pool_t{.handle = 0});
    m_retain = std::exchange(rhs.m_retain, retain_mode::per_replay);
    return *this;
}

void
owned_block_t::reset() noexcept
{
    if(!m_block) return;
    if(m_retain == retain_mode::across_replays)
        retain_block(m_pool, m_block);
    else if(auto* ext = hsa::get_amd_ext_table(); ext && ext->hsa_amd_memory_pool_free_fn)
        ext->hsa_amd_memory_pool_free_fn(m_block.data);

    m_block  = {};
    m_kind   = storage_kind::none;
    m_pool   = hsa_amd_memory_pool_t{.handle = 0};
    m_retain = retain_mode::per_replay;
}

size_t
snapshot_t::footprint_bytes() const
{
    size_t total = 0;
    for(const auto& region : regions)
        if(region.liveness != liveness_kind::retired) total += region.size;
    return total;
}

storage_kind
snapshot_t::storage() const
{
    if(segments.empty()) return storage_kind::none;
    const auto first = segments.front().backing.kind();
    for(const auto& segment : segments)
        if(segment.backing.kind() != first) return storage_kind::mixed;
    return first;
}

snapshot_t
snap(hsa_agent_t agent)
{
    auto cache = agent::get_agent_cache(agent);
    if(!cache)
    {
        auto out   = snapshot_t{};
        out.status = capture_status::unavailable;
        return out;
    }

    return snap(capture_context_t{
        agent,
        cache->near_cpu(),
        cache->gpu_pool(),
        cache->cpu_pool(),
        capture_mode::force_pinned,
        false,
    });
}

snapshot_t
snap(const capture_context_t& ctx)
{
    return snap(ctx, storage_budgets_t{gpu_plan_budget(ctx), host_plan_budget(ctx)});
}

snapshot_t
snap(const capture_context_t& ctx, storage_budgets_t budgets)
{
    auto out = snapshot_t{};

    // Note: trackable allocations carrying HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG are recorded in
    // memory_tracker::unsupported_executable() and omitted from the main inventory. Declining
    // replay whenever that side inventory is non-empty is not viable -- the HIP runtime (and the
    // SDK's own AQL pools) routinely keep such allocations live -- so they remain an unsupported
    // omitted class for beta (documented in the public header). Direct-HSA apps that put ordinary
    // writable device data behind the flag observe the same omission.

    // Host memory pressure while building the inventory is the same condition the per-region
    // capture below reports through ok==false, and snap()'s contract is that the caller declines
    // replay and runs the dispatch once. Aborting here instead would kill the application over an
    // opt-in beta feature, and a large allocation inventory is exactly when it would happen.
    auto inventory = memory_tracker::alloc_map_t{};
    auto scan      = module_variable_scan_t{};
    try
    {
        inventory = memory_tracker::snap_inventory(ctx.agent);
        scan      = discover_module_variables(ctx.agent);
        out.regions.reserve(inventory.size() + scan.found.size());
    } catch(const std::bad_alloc&)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel-replay snapshot: out of memory reserving metadata; "
                                   "declining replay for this dispatch";
        out.status = capture_status::unavailable;
        return out;
    }

    // An executable or symbol HSA would not tell us about may hold a writable __device__ global. We
    // cannot restore what we did not capture, so the passes would not see identical inputs.
    if(scan.incomplete)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel-replay snapshot: could not enumerate module-scope "
                                   "variables for every loaded executable; declining replay for "
                                   "this dispatch";
        out.status = capture_status::invalid_plan;
        return out;
    }

    for(const auto& [ptr, size] : inventory)
    {
        if(size == 0) continue;
        out.regions.emplace_back(snapshot_region_t{size, ptr, liveness_kind::tracked_allocation});
    }

    for(const auto& var : scan.found)
    {
        if(var.size == 0) continue;
        out.regions.emplace_back(
            snapshot_region_t{var.size, var.gpu_addr, liveness_kind::module_variable});
    }

    if(out.regions.empty())
    {
        out.status = capture_status::complete;
        return out;
    }

    // Address order makes placement under memory pressure reproducible from run to run.
    std::sort(out.regions.begin(), out.regions.end(), [](const auto& lhs, const auto& rhs) {
        return std::less<>{}(lhs.live_address, rhs.live_address);
    });

    auto chain      = make_snapshot_storage(ctx, budgets);
    auto placements = std::optional<std::vector<planning::placement_t>>{};
    try
    {
        auto requests = std::vector<planning::request_t>{};
        requests.reserve(out.regions.size());
        for(size_t i = 0; i < out.regions.size(); ++i)
            requests.push_back(planning::request_t{i, 0, out.regions[i].size});

        out.segments.reserve(snapshot_storage::tier_count);
        placements = planning::plan_and_commit(chain, requests);
        if(placements)
        {
            chain.release_backings([&](size_t tier, size_t size, owned_block_t backing) {
                out.segments.push_back(storage_segment_t{size, tier, std::move(backing)});
            });
        }
    } catch(const std::bad_alloc&)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel-replay snapshot: out of memory planning snapshot "
                                   "storage; declining replay for this dispatch";
        out.regions.clear();
        out.segments.clear();
        out.status = capture_status::unavailable;
        return out;
    }

    if(!placements)
    {
        ROCP_WARNING << fmt::format(
            "kernel-replay snapshot: could not place {} bytes within {} GPU "
            "and {} pinned-host bytes on agent {}; declining replay",
            out.footprint_bytes(),
            budgets.gpu_bytes,
            budgets.host_bytes,
            ctx.agent.handle);
        out.regions.clear();
        out.status = capture_status::unavailable;
        return out;
    }

    try
    {
        auto segment_for_tier = std::array<std::optional<size_t>, snapshot_storage::tier_count>{};
        for(size_t i = 0; i < out.segments.size(); ++i)
            segment_for_tier[out.segments[i].tier] = i;

        out.extents.reserve(placements->size());
        for(const auto& placement : *placements)
        {
            const auto segment = segment_for_tier[placement.tier];
            if(!segment)
            {
                out.regions.clear();
                out.segments.clear();
                out.status = capture_status::invalid_plan;
                return out;
            }
            out.extents.push_back(snapshot_extent_t{placement.size,
                                                    placement.region_index,
                                                    placement.region_offset,
                                                    *segment,
                                                    placement.backing_offset});
        }
    } catch(const std::bad_alloc&)
    {
        out.regions.clear();
        out.segments.clear();
        out.extents.clear();
        out.status = capture_status::unavailable;
        return out;
    }

    size_t captured_regions = 0;
    for(size_t region_index = 0; region_index < out.regions.size(); ++region_index)
    {
        auto&      region         = out.regions[region_index];
        const auto capture_region = [&] {
            for(const auto& extent : out.extents)
            {
                if(extent.region_index != region_index) continue;
                const auto copy   = capture_copy(out, extent);
                const auto status = dma_copy(copy.dest, copy.source, copy.size);
                if(status != HSA_STATUS_SUCCESS) return status;
            }
            return HSA_STATUS_SUCCESS;
        };
        const auto st = region.liveness == liveness_kind::tracked_allocation
                            ? with_inventory_check(region.live_address, region.size, capture_region)
                            : std::optional<hsa_status_t>{capture_region()};

        if(!st)
        {
            ROCP_INFO << fmt::format("kernel-replay snapshot: region {} ({}B) was retired before "
                                     "capture, dropping from snapshot",
                                     region.live_address,
                                     region.size);
            region.liveness = liveness_kind::retired;
            continue;
        }
        if(*st != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format("kernel-replay snapshot: copy failed for region {} ({}B)",
                                        region.live_address,
                                        region.size);
            out.regions.clear();
            out.extents.clear();
            out.segments.clear();
            out.status = capture_status::copy_failed;
            return out;
        }
        ++captured_regions;
    }

    out.status = capture_status::complete;

    size_t gpu_bytes  = 0;
    size_t host_bytes = 0;
    for(const auto& segment : out.segments)
    {
        if(segment.backing.kind() == storage_kind::gpu_local)
            gpu_bytes += segment.size;
        else if(segment.backing.kind() == storage_kind::pinned_host)
            host_bytes += segment.size;
    }
    ROCP_INFO << fmt::format("kernel-replay snapshot: captured {} regions using {} GPU bytes and "
                             "{} pinned-host bytes for agent {}",
                             captured_regions,
                             gpu_bytes,
                             host_bytes,
                             ctx.agent.handle);
    return out;
}

bool
restore(const snapshot_t& snapshot)
{
    if(!snapshot.complete()) return false;
    if(snapshot.extents.empty()) return true;

    size_t     restored = 0;
    const auto status   = memory_tracker::inventory().rlock([&](const auto& inventory) {
        for(const auto& extent : snapshot.extents)
        {
            const auto& region = snapshot.regions.at(extent.region_index);
            if(!region_is_live(region, inventory))
            {
                ROCP_WARNING << fmt::format(
                    "kernel-replay restore: skipping region {} ({}B) that is no longer live",
                    region.live_address,
                    region.size);
                continue;
            }

            const auto copy = restore_copy(snapshot, extent);
            const auto st   = dma_copy(copy.dest, copy.source, copy.size);
            if(st != HSA_STATUS_SUCCESS) return st;
            ++restored;
        }
        return HSA_STATUS_SUCCESS;
    });

    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << fmt::format("kernel-replay restore: copy failed after {}/{} extents",
                                  restored,
                                  snapshot.extents.size());
        return false;
    }
    ROCP_INFO << fmt::format(
        "kernel-replay restore: restored {}/{} extents", restored, snapshot.extents.size());
    return true;
}

bool
restore(const snapshot_t& snapshot, const batch_copy_fn_t& batch_copy)
{
    if(!snapshot.complete()) return false;

    auto gpu_regions = std::vector<blit::copy_region_t>{};
    try
    {
        gpu_regions.reserve(snapshot.extents.size());
    } catch(const std::bad_alloc&)
    {
        return false;
    }
    size_t restored = 0;
    size_t skipped  = 0;

    const auto status = memory_tracker::inventory().rlock([&](const auto& map) {
        for(const auto& extent : snapshot.extents)
        {
            const auto& region = snapshot.regions.at(extent.region_index);
            if(!region_is_live(region, map))
            {
                ROCP_WARNING << fmt::format(
                    "kernel-replay restore: skipping region {} ({}B) that is no longer live",
                    region.live_address,
                    region.size);
                ++skipped;
                continue;
            }

            auto       copy = restore_copy(snapshot, extent);
            const auto kind = snapshot.segments.at(extent.segment_index).backing.kind();
            if(kind == storage_kind::gpu_local)
            {
                gpu_regions.emplace_back(copy);
            }
            else
            {
                auto copy_status = dma_copy(copy.dest, copy.source, copy.size);
                if(copy_status != HSA_STATUS_SUCCESS) return copy_status;
                ++restored;
            }
        }

        auto copy_status = batch_copy(gpu_regions);
        if(copy_status != HSA_STATUS_SUCCESS) return copy_status;
        restored += gpu_regions.size();
        return HSA_STATUS_SUCCESS;
    });

    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << fmt::format("kernel-replay restore: batch copy failed after {}/{} regions",
                                  restored,
                                  snapshot.extents.size());
        return false;
    }

    ROCP_INFO << fmt::format("kernel-replay restore: restored {}/{} regions ({} skipped)",
                             restored,
                             snapshot.extents.size(),
                             skipped);
    return true;
}
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
