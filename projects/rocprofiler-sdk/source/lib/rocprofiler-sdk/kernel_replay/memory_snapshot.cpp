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

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
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

// At most one reusable high-water arena per GPU pool. A larger returned arena replaces and frees
// the old one; smaller returns are freed. This bounds retained VRAM independently of inventory
// size churn.
using device_backing_cache_t = std::unordered_map<uint64_t, block_t>;

common::Synchronized<device_backing_cache_t>&
device_backing_cache()
{
    // Keep backing allocations for process lifetime. HSA may already be unavailable during static
    // destruction, so do not attempt to free them there.
    static auto* value = new common::Synchronized<device_backing_cache_t>{};
    return *value;
}

block_t
take_cached_device_copy(hsa_amd_memory_pool_t pool, size_t size) noexcept
{
    if(pool.handle == 0 || size == 0) return {};
    try
    {
        return device_backing_cache().wlock([&](auto& cache) -> block_t {
            auto pool_itr = cache.find(pool.handle);
            if(pool_itr == cache.end() || !pool_itr->second || pool_itr->second.size < size)
                return {};
            return std::exchange(pool_itr->second, block_t{});
        });
    } catch(...)
    {
        return {};
    }
}

size_t
cached_device_capacity(hsa_amd_memory_pool_t pool) noexcept
{
    if(pool.handle == 0) return 0;
    try
    {
        return device_backing_cache().rlock([&](const auto& cache) {
            auto itr = cache.find(pool.handle);
            return itr == cache.end() ? size_t{0} : itr->second.size;
        });
    } catch(...)
    {
        return 0;
    }
}

void
free_pool_block(block_t block) noexcept
{
    if(!block) return;
    if(auto* ext = hsa::get_amd_ext_table(); ext && ext->hsa_amd_memory_pool_free_fn)
        ext->hsa_amd_memory_pool_free_fn(block.data);
}

void
release_device_copy(hsa_amd_memory_pool_t pool, block_t block) noexcept
{
    if(pool.handle == 0 || !block)
    {
        free_pool_block(block);
        return;
    }

    // Cache insertion may allocate. Destruction must remain noexcept, so retain nothing if the map
    // cannot grow and release the arena directly.
    try
    {
        device_backing_cache().wlock([&](auto& cache) {
            auto& cached = cache[pool.handle];
            if(!cached || block.size > cached.size) std::swap(cached, block);
        });
    } catch(...)
    {}
    free_pool_block(block);
}

enum class allocation_error : uint8_t
{
    none,
    unavailable,
    out_of_resources,
    hard_failure,
};

struct allocation_result_t
{
    owned_block_t    value{};
    allocation_error error = allocation_error::none;

    explicit operator bool() const { return static_cast<bool>(value); }
};

constexpr size_t arena_alignment = 16;

std::optional<size_t>
checked_align_up(size_t value, size_t alignment)
{
    if(alignment == 0) return std::nullopt;
    const auto remainder = value % alignment;
    if(remainder == 0) return value;
    const auto padding = alignment - remainder;
    if(value > std::numeric_limits<size_t>::max() - padding) return std::nullopt;
    return value + padding;
}

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
    if(auto available = gpu_available_bytes(ctx.agent))
    {
        const auto fresh_budget = *available - (*available / 10);
        const auto cached       = static_cast<uint64_t>(cached_device_capacity(ctx.gpu_pool));
        // A single arena can reuse the cached block or allocate from fresh capacity, but it cannot
        // combine the two without first releasing and reallocating the cache.
        const auto realizable = std::max(fresh_budget, cached);
        return static_cast<size_t>(std::min<uint64_t>(static_cast<uint64_t>(*maximum), realizable));
    }
    return *maximum;
}

size_t
host_plan_budget(const capture_context_t& ctx)
{
    if(ctx.mode == capture_mode::force_gpu) return 0;
    auto maximum = pool_allocation_max(ctx.pinned_pool);
    if(!maximum) return 0;
    if(auto available = host_available_bytes())
        return std::min(*maximum, static_cast<size_t>(*available / 2));
    return *maximum;
}

planning::plan_t
build_storage_plan(const std::vector<snapshot_region_t>& regions,
                   size_t                                gpu_budget,
                   size_t                                host_budget)
{
    auto requests = std::vector<planning::request_t>{};
    try
    {
        requests.reserve(regions.size());
        for(size_t i = 0; i < regions.size(); ++i)
        {
            requests.emplace_back(planning::request_t{
                regions[i].size,
                i,
                reinterpret_cast<uintptr_t>(regions[i].live_address),
            });
        }
    } catch(const std::bad_alloc&)
    {
        auto result  = planning::plan_t{};
        result.error = planning::error_t{planning::error_code::metadata_failure};
        return result;
    }

    auto allocator = planning::snapshot_allocator{
        planning::gpu_allocator{gpu_budget},
        planning::pinned_allocator{host_budget},
        planning::null_allocator{},
    };
    return planning::build(std::move(requests), allocator, arena_alignment);
}

allocation_result_t
allocate_pool_block(hsa_amd_memory_pool_t pool,
                    hsa_agent_t           agent,
                    hsa_agent_t           pinned_agent,
                    size_t                size,
                    storage_kind          kind,
                    bool                  cacheable)
{
    if(size == 0) return allocation_result_t{};
    if(!pool_accepts(pool, size))
        return allocation_result_t{.error = allocation_error::unavailable};

    if(cacheable)
    {
        if(auto cached = take_cached_device_copy(pool, size); cached)
            return allocation_result_t{.value = owned_block_t{cached, kind, pool, true}};
    }

    auto* ext = hsa::get_amd_ext_table();
    if(!ext || !ext->hsa_amd_memory_pool_allocate_fn || !ext->hsa_amd_agents_allow_access_fn ||
       !ext->hsa_amd_memory_pool_free_fn)
        return allocation_result_t{.error = allocation_error::hard_failure};

    void* ptr    = nullptr;
    auto  status = ext->hsa_amd_memory_pool_allocate_fn(pool, size, 0, &ptr);
    if(status != HSA_STATUS_SUCCESS || !ptr)
        return allocation_result_t{.error = allocation_error::out_of_resources};

    if(kind == storage_kind::pinned_host && pinned_agent.handle != 0)
    {
        status = ext->hsa_amd_agents_allow_access_fn(1, &pinned_agent, nullptr, ptr);
        if(status != HSA_STATUS_SUCCESS)
        {
            ext->hsa_amd_memory_pool_free_fn(ptr);
            return allocation_result_t{.error = allocation_error::hard_failure};
        }
    }
    status = ext->hsa_amd_agents_allow_access_fn(1, &agent, nullptr, ptr);
    if(status == HSA_STATUS_SUCCESS)
        return allocation_result_t{.value =
                                       owned_block_t{block_t{size, ptr}, kind, pool, cacheable}};

    ext->hsa_amd_memory_pool_free_fn(ptr);
    return allocation_result_t{.error = allocation_error::hard_failure};
}

struct gpu_arena_allocator
{
    allocation_result_t operator()(size_t size, const capture_context_t& ctx) const
    {
        if(!ctx.gpu_backend_available || ctx.gpu_pool.handle == 0)
            return allocation_result_t{.error = allocation_error::unavailable};

        // Reusing retained GPU backing consumes no new VRAM, so check the cache before applying the
        // current-availability admission gate.
        if(auto cached = take_cached_device_copy(ctx.gpu_pool, size); cached)
            return allocation_result_t{
                .value = owned_block_t{cached, storage_kind::gpu_local, ctx.gpu_pool, true}};

        if(auto available = gpu_available_bytes(ctx.agent))
        {
            // Keep ten percent available for runtime scratch and allocations racing this advisory
            // query. The HSA allocation result below remains authoritative.
            const auto budget = *available - (*available / 10);
            if(size > budget) return allocation_result_t{.error = allocation_error::unavailable};
        }
        return allocate_pool_block(
            ctx.gpu_pool, ctx.agent, hsa_agent_t{.handle = 0}, size, storage_kind::gpu_local, true);
    }
};

struct pinned_host_arena_allocator
{
    allocation_result_t operator()(size_t size, const capture_context_t& ctx) const
    {
        if(ctx.pinned_pool.handle == 0)
            return allocation_result_t{.error = allocation_error::unavailable};

        if(auto available = host_available_bytes())
        {
            // Pinned storage is non-reclaimable. Never consume more than half of currently
            // available host memory in this MVP.
            if(size > (*available / 2))
                return allocation_result_t{.error = allocation_error::unavailable};
        }
        return allocate_pool_block(
            ctx.pinned_pool, ctx.agent, ctx.pinned_agent, size, storage_kind::pinned_host, false);
    }
};

struct materialization_t
{
    std::vector<storage_segment_t> segments{};
    allocation_error               error      = allocation_error::none;
    bool                           gpu_failed = false;

    bool complete() const { return error == allocation_error::none; }
};

materialization_t
materialize_plan(const planning::plan_t& plan, const capture_context_t& ctx)
{
    auto out = materialization_t{};
    try
    {
        out.segments.reserve(2);
    } catch(const std::bad_alloc&)
    {
        out.error = allocation_error::out_of_resources;
        return out;
    }

    if(plan.gpu_bytes > 0)
    {
        auto result = gpu_arena_allocator{}(plan.gpu_bytes, ctx);
        if(!result)
        {
            out.error      = result.error;
            out.gpu_failed = true;
            return out;
        }
        out.segments.emplace_back(storage_segment_t{plan.gpu_bytes, 0, std::move(result.value)});
    }

    if(plan.host_bytes > 0)
    {
        auto result = pinned_host_arena_allocator{}(plan.host_bytes, ctx);
        if(!result)
        {
            out.segments.clear();
            out.error = result.error;
            return out;
        }
        out.segments.emplace_back(storage_segment_t{plan.host_bytes, 0, std::move(result.value)});
    }
    return out;
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
                             bool                  cacheable)
: m_block{block}
, m_kind{kind}
, m_pool{pool}
, m_cacheable{cacheable}
{}

owned_block_t::~owned_block_t() { reset(); }

owned_block_t::owned_block_t(owned_block_t&& rhs) noexcept
: m_block{std::exchange(rhs.m_block, block_t{})}
, m_kind{std::exchange(rhs.m_kind, storage_kind::none)}
, m_pool{std::exchange(rhs.m_pool, hsa_amd_memory_pool_t{.handle = 0})}
, m_cacheable{std::exchange(rhs.m_cacheable, false)}
{}

owned_block_t&
owned_block_t::operator=(owned_block_t&& rhs) noexcept
{
    if(this == &rhs) return *this;
    reset();
    m_block     = std::exchange(rhs.m_block, block_t{});
    m_kind      = std::exchange(rhs.m_kind, storage_kind::none);
    m_pool      = std::exchange(rhs.m_pool, hsa_amd_memory_pool_t{.handle = 0});
    m_cacheable = std::exchange(rhs.m_cacheable, false);
    return *this;
}

void
owned_block_t::reset() noexcept
{
    if(!m_block) return;
    if(m_cacheable)
        release_device_copy(m_pool, m_block);
    else if(auto* ext = hsa::get_amd_ext_table(); ext && ext->hsa_amd_memory_pool_free_fn)
        ext->hsa_amd_memory_pool_free_fn(m_block.data);

    m_block     = {};
    m_kind      = storage_kind::none;
    m_pool      = hsa_amd_memory_pool_t{.handle = 0};
    m_cacheable = false;
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

std::optional<size_t>
assign_logical_offsets(std::vector<snapshot_region_t>& regions)
{
    size_t required = 0;
    for(auto& region : regions)
    {
        auto aligned = checked_align_up(required, arena_alignment);
        if(!aligned || region.size > std::numeric_limits<size_t>::max() - *aligned)
            return std::nullopt;
        region.logical_offset = *aligned;
        required              = *aligned + region.size;
    }
    return required;
}

std::optional<std::vector<snapshot_extent_t>>
map_regions_to_storage(const std::vector<snapshot_region_t>& regions,
                       const std::vector<storage_segment_t>& segments)
{
    auto out = std::vector<snapshot_extent_t>{};
    try
    {
        out.reserve(regions.size());
    } catch(const std::bad_alloc&)
    {
        return std::nullopt;
    }

    try
    {
        size_t previous_segment_end = 0;
        for(const auto& segment : segments)
        {
            if(segment.size > std::numeric_limits<size_t>::max() - segment.logical_offset)
                return std::nullopt;
            if(segment.logical_offset < previous_segment_end) return std::nullopt;
            if(segment.backing && segment.backing.block().size < segment.size) return std::nullopt;
            previous_segment_end = segment.logical_offset + segment.size;
        }

        for(size_t region_index = 0; region_index < regions.size(); ++region_index)
        {
            const auto& region = regions[region_index];
            if(region.size > std::numeric_limits<size_t>::max() - region.logical_offset)
                return std::nullopt;
            const auto region_end = region.logical_offset + region.size;
            auto       cursor     = region.logical_offset;

            for(size_t segment_index = 0; segment_index < segments.size(); ++segment_index)
            {
                const auto& segment     = segments[segment_index];
                const auto  segment_end = segment.logical_offset + segment.size;
                const auto  begin       = std::max(region.logical_offset, segment.logical_offset);
                const auto  end         = std::min(region_end, segment_end);
                if(begin >= end) continue;
                if(begin != cursor) return std::nullopt;

                out.emplace_back(snapshot_extent_t{
                    end - begin,
                    region_index,
                    begin - region.logical_offset,
                    segment_index,
                    begin - segment.logical_offset,
                });
                cursor = end;
            }
            if(cursor != region_end) return std::nullopt;
        }
    } catch(const std::bad_alloc&)
    {
        return std::nullopt;
    }
    return out;
}

snapshot_t
snap(hsa_agent_t agent)
{
    auto cache = agent::get_agent_cache(agent);
    if(!cache)
    {
        auto out   = snapshot_t{};
        out.agent  = agent;
        out.status = capture_status::unavailable;
        return out;
    }

    return snap(capture_context_t{
        agent,
        cache->get_rocp_agent()->id,
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
    auto out         = snapshot_t{};
    out.agent        = ctx.agent;
    out.pinned_agent = ctx.pinned_agent;
    out.agent_id     = ctx.agent_id;

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
        out.regions.emplace_back(
            snapshot_region_t{size, ptr, 0, liveness_kind::tracked_allocation});
    }

    for(const auto& var : scan.found)
    {
        if(var.size == 0) continue;
        out.regions.emplace_back(
            snapshot_region_t{var.size, var.gpu_addr, 0, liveness_kind::module_variable});
    }

    if(out.regions.empty())
    {
        out.status = capture_status::complete;
        return out;
    }

    auto plan = build_storage_plan(out.regions, budgets.gpu_bytes, budgets.host_bytes);
    if(!plan.complete())
    {
        ROCP_WARNING << fmt::format("kernel-replay snapshot: no placement for region {} requesting "
                                    "{} bytes on agent {}; declining replay",
                                    plan.error.region_index,
                                    plan.error.requested,
                                    ctx.agent.handle);
        out.regions.clear();
        out.status = plan.error.code == planning::error_code::overflow
                         ? capture_status::invalid_plan
                         : capture_status::unavailable;
        return out;
    }

    auto materialization = materialize_plan(plan, ctx);
    if(!materialization.complete() && materialization.gpu_failed &&
       ctx.mode == capture_mode::automatic)
    {
        // The capacity query raced or the single GPU arena could not be realized. No application
        // bytes have been captured, so rebuild the complete plan with pinned storage only.
        plan            = build_storage_plan(out.regions, 0, budgets.host_bytes);
        materialization = plan.complete() ? materialize_plan(plan, ctx) : materialization_t{};
        if(!plan.complete()) materialization.error = allocation_error::unavailable;
    }
    if(!materialization.complete())
    {
        ROCP_WARNING << fmt::format("kernel-replay snapshot: failed to materialize backing for "
                                    "agent {}; declining replay",
                                    ctx.agent.handle);
        out.regions.clear();
        out.status = capture_status::unavailable;
        return out;
    }
    out.segments = std::move(materialization.segments);

    try
    {
        out.extents.reserve(plan.placements.size());
        auto gpu_segment  = std::optional<size_t>{};
        auto host_segment = std::optional<size_t>{};
        for(size_t i = 0; i < out.segments.size(); ++i)
        {
            if(out.segments[i].backing.kind() == storage_kind::gpu_local)
                gpu_segment = i;
            else if(out.segments[i].backing.kind() == storage_kind::pinned_host)
                host_segment = i;
        }

        for(const auto& placement : plan.placements)
        {
            const auto segment_index =
                placement.tier == storage_kind::gpu_local ? gpu_segment : host_segment;
            if(!segment_index)
            {
                out.regions.clear();
                out.segments.clear();
                out.status = capture_status::invalid_plan;
                return out;
            }
            out.regions[placement.region_index].logical_offset = placement.backing_offset;
            out.extents.emplace_back(snapshot_extent_t{
                placement.size,
                placement.region_index,
                0,
                *segment_index,
                placement.backing_offset,
            });
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

    ROCP_INFO << fmt::format("kernel-replay snapshot: captured {} regions using {} GPU bytes and "
                             "{} pinned-host bytes for agent {}",
                             captured_regions,
                             plan.gpu_bytes,
                             plan.host_bytes,
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
