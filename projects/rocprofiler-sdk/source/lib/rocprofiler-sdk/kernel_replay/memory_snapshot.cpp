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
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <cstdint>
#include <new>
#include <optional>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace memory_snapshot
{
namespace
{
hsa_status_t
dma_copy(void* dst, const void* src, size_t n)
{
    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_memory_copy_fn) return HSA_STATUS_ERROR;
    return core->hsa_memory_copy_fn(dst, src, n);
}

// A module-scope variable (__device__ / __constant__ global) discovered in a loaded executable.
struct module_variable_t
{
    void*  gpu_addr = nullptr;
    size_t size     = 0;
};

// hsa_executable_iterate_agent_symbols callback: collect HSA_SYMBOL_KIND_VARIABLE symbols
// (device address + size) into the vector passed via `data`. The HSA callback cannot capture, so
// state is threaded through the void* argument.
hsa_status_t
collect_module_variable(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t symbol, void* data)
{
    auto* out  = static_cast<std::vector<module_variable_t>*>(data);
    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_executable_symbol_get_info_fn) return HSA_STATUS_SUCCESS;

    hsa_symbol_kind_t kind{};
    if(core->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) !=
           HSA_STATUS_SUCCESS ||
       kind != HSA_SYMBOL_KIND_VARIABLE)
        return HSA_STATUS_SUCCESS;

    uint64_t addr = 0;
    uint32_t size = 0;
    if(core->hsa_executable_symbol_get_info_fn(
           symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr) != HSA_STATUS_SUCCESS ||
       core->hsa_executable_symbol_get_info_fn(
           symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &size) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    // Skip empties; 1 GiB per-variable sanity cap.
    if(addr == 0 || size == 0 || size > (1ULL << 30)) return HSA_STATUS_SUCCESS;

    // HSA reports the variable's device address as an integer; converting to a pointer is required.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    out->push_back(module_variable_t{reinterpret_cast<void*>(addr), size});
    return HSA_STATUS_SUCCESS;
}

// Enumerate module-scope variables visible to `agent` across all loaded executables. They live in
// the executable's data segment -- not in the allocation tracker's inventory -- so a kernel that
// mutates a __device__ global would otherwise leak that mutation across replay passes. Must run at
// snap time (not executable-load time): constant memory may not be populated at load.
std::vector<module_variable_t>
discover_module_variables(hsa_agent_t agent)
{
    std::vector<module_variable_t> found;

    auto* core = hsa::get_core_table();
    if(!core || !core->hsa_executable_iterate_agent_symbols_fn) return found;

    code_object::iterate_loaded_code_objects([&](const code_object::hsa::code_object& co) {
        // Iterating for `agent` naturally scopes to executables loaded on this agent (others yield
        // no symbols), matching snap()'s per-agent contract.
        core->hsa_executable_iterate_agent_symbols_fn(
            co.hsa_executable, agent, collect_module_variable, &found);
    });
    return found;
}
}  // namespace

device_snapshot_t
snap(hsa_agent_t agent)
{
    const auto inventory = memory_tracker::snap_inventory(agent);

    device_snapshot_t out{};
    out.blocks.reserve(inventory.size());

    // Capture one region (device->host) into the snapshot. Returns false on any failure: a host
    // allocation failure under memory pressure (resize throws bad_alloc) or a failed DMA copy.
    // Either leaves the snapshot incomplete, so the caller must decline replay rather than restore
    // partial state.
    auto capture =
        [&](void* gpu_addr, size_t size, std::string_view what, bool from_tracker) -> bool {
        if(size == 0) return true;

        mem_block_t blk;
        blk.gpu_addr     = gpu_addr;
        blk.from_tracker = from_tracker;
        try
        {
            blk.host_copy.resize(size);
        } catch(const std::bad_alloc&)
        {
            ROCP_WARNING << fmt::format("kernel-replay snapshot: host allocation of {} bytes "
                                        "failed for {} {} (memory pressure)",
                                        size,
                                        what,
                                        gpu_addr);
            return false;
        }

        if(dma_copy(blk.host_copy.data(), gpu_addr, size) != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format(
                "kernel-replay snapshot: device->host copy failed for {} {} ({}B)",
                what,
                gpu_addr,
                size);
            return false;
        }

        out.blocks.push_back(std::move(blk));
        return true;
    };

    for(const auto& [ptr, size] : inventory)
    {
        if(!capture(ptr, size, "region", /*from_tracker=*/true))
        {
            out.ok = false;
            return out;
        }
    }

    // Module-scope variables (__device__ / __constant__ globals) live in the loaded executable's
    // data segment, not in the allocation tracker, so capture them here too. Restored via the same
    // per-block host->device copy as tracked allocations (see restore()).
    for(const auto& var : discover_module_variables(agent))
    {
        if(!capture(var.gpu_addr, var.size, "module variable", /*from_tracker=*/false))
        {
            out.ok = false;
            return out;
        }
    }

    ROCP_INFO << fmt::format("kernel-replay snapshot: captured {} regions (tracked allocations + "
                             "module variables) for agent {}",
                             out.blocks.size(),
                             agent.handle);
    return out;
}

size_t
restore(const device_snapshot_t& snapshot)
{
    size_t ok = 0;
    for(const auto& blk : snapshot.blocks)
    {
        hsa_status_t status = HSA_STATUS_SUCCESS;

        if(blk.from_tracker)
        {
            // Copy under the read lock so a concurrent free cannot race the check and the write.
            // returns nullopt when the region is no longer a live allocation of at least its
            // size, meaning it was freed or reallocated after snap. Skip it.
            const auto st = memory_tracker::inventory().rlock(
                [&](const memory_tracker::tracked_map_t& map) -> std::optional<hsa_status_t> {
                    auto itr = map.find(blk.gpu_addr);
                    if(itr == map.end() || itr->second.size < blk.host_copy.size())
                        return std::nullopt;
                    return dma_copy(blk.gpu_addr, blk.host_copy.data(), blk.host_copy.size());
                });

            if(!st)
            {
                ROCP_WARNING << fmt::format(
                    "kernel-replay restore: skipping region {} ({}B) that is no longer live",
                    blk.gpu_addr,
                    blk.host_copy.size());
                continue;
            }
            status = *st;
        }
        else
        {
            // Module variable: lives in the loaded executable, always present, so restore directly.
            status = dma_copy(blk.gpu_addr, blk.host_copy.data(), blk.host_copy.size());
        }

        if(status != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format(
                "kernel-replay restore: host->device copy failed for region {} ({}B)",
                blk.gpu_addr,
                blk.host_copy.size());
            continue;
        }
        ++ok;
    }

    ROCP_INFO << fmt::format(
        "kernel-replay restore: restored {}/{} regions", ok, snapshot.blocks.size());
    return ok;
}
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
