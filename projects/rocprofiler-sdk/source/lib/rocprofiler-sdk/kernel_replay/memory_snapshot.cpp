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

#include <algorithm>
#include <cstdint>
#include <new>
#include <optional>
#include <string_view>
#include <unordered_set>
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

/// @brief Run @p copy under the tracker read lock, but only while region_is_restorable() still
/// holds for [@p gpu_addr, +@p size), so a concurrent free (write lock) can't retire or replace the
/// allocation mid-copy. Direction is caller-supplied: snap reads device->host, restore writes
/// host->device.
///
/// @return @p copy's status, or @c std::nullopt if the region was freed, shrunk, or replaced since
/// it was recorded.
template <typename CopyFn>
std::optional<hsa_status_t>
with_inventory_check(void* gpu_addr, size_t size, uint64_t generation, CopyFn&& copy)
{
    return memory_tracker::inventory().rlock(
        [&](const memory_tracker::tracked_map_t& map) -> std::optional<hsa_status_t> {
            if(!region_is_restorable(map, gpu_addr, size, generation)) return std::nullopt;
            return copy();
        });
}

// A module-scope variable (__device__ / __constant__ global) discovered in a loaded executable.
struct module_variable_t
{
    void*            gpu_addr = nullptr;
    size_t           size     = 0;
    hsa_executable_t executable{};  // owner; its liveness is this address's liveness
};

// hsa_executable_iterate_agent_symbols callback: collect HSA_SYMBOL_KIND_VARIABLE symbols
// (device address + size) into the vector passed via `data`. The HSA callback cannot capture, so
// state is threaded through the void* argument.
hsa_status_t
collect_module_variable(hsa_executable_t executable,
                        hsa_agent_t,
                        hsa_executable_symbol_t symbol,
                        void*                   data)
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
    out->push_back(module_variable_t{reinterpret_cast<void*>(addr), size, executable});
    return HSA_STATUS_SUCCESS;
}

// Handles of the executables the SDK currently tracks as loaded. Taken once per restore so each
// module-scope region can be checked without a walk of its own. The code-object module drops an
// executable from this set when it is destroyed, which is what makes the check meaningful.
std::unordered_set<uint64_t>
loaded_executable_handles()
{
    auto out = std::unordered_set<uint64_t>{};
    code_object::iterate_loaded_code_objects(
        [&](const code_object::hsa::code_object& co) { out.emplace(co.hsa_executable.handle); });
    return out;
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

bool
region_is_restorable(const memory_tracker::tracked_map_t& inventory,
                     void*                                gpu_addr,
                     size_t                               size,
                     uint64_t                             generation)
{
    const auto itr = inventory.find(gpu_addr);
    if(itr == inventory.end()) return false;

    // A shrunk allocation is rejected rather than partially copied: the recorded length is what the
    // snapshot holds, and writing it into something smaller would run past the allocation.
    if(itr->second.size < size) return false;

    // A caller that supplied a generation is asserting an exact identity, so a mismatch is a reused
    // address and not the region that was captured.
    return generation == 0 || itr->second.generation == generation;
}

bool
module_region_is_restorable(const std::unordered_set<uint64_t>& loaded_executables,
                            hsa_executable_t                    executable)
{
    if(loaded_executables.empty()) return true;
    return loaded_executables.count(executable.handle) > 0;
}

device_snapshot_t
snap(hsa_agent_t agent)
{
    device_snapshot_t out{};

    // Note: trackable allocations carrying HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG are recorded in
    // memory_tracker::unsupported_executable() and omitted from the main inventory. Declining
    // replay whenever that side inventory is non-empty is not viable -- the HIP runtime (and the
    // SDK's own AQL pools) routinely keep such allocations live -- so they remain an unsupported
    // omitted class for beta (documented in the public header). Direct-HSA apps that put ordinary
    // writable device data behind the flag observe the same omission.

    auto inventory   = memory_tracker::tracked_map_t{};
    auto module_vars = std::vector<module_variable_t>{};
    try
    {
        inventory   = memory_tracker::snap_inventory(agent);
        module_vars = discover_module_variables(agent);
        out.blocks.reserve(inventory.size() + module_vars.size());
    } catch(const std::bad_alloc&)
    {
        // Same policy as a per-region allocation failure below: report an incomplete capture so the
        // caller declines replay and runs the dispatch once. Aborting the process here would kill a
        // long job over transient host memory pressure, which is a worse outcome than not profiling
        // one dispatch -- and the caller already has a correct path for ok == false.
        ROCP_WARNING
            << "kernel-replay snapshot: out of memory reserving metadata; declining replay";
        out.ok = false;
        return out;
    }

    /// @brief Capture one region (device->host) into the snapshot.
    /// @param generation Tracker generation expected for @p gpu_addr (0 for module variables).
    /// @retval false The snapshot is incomplete because a host allocation or the DMA copy failed.
    /// The caller must decline replay rather than restore partial state.
    /// @retval true The region was captured, or was dropped because it had been freed, shrunk, or
    /// replaced by a different allocation at the same address since snap_inventory(). Dropping a
    /// freed region is harmless -- restore() runs the same check, so it could not have been
    /// restored anyway. Dropping a *replaced* region means the new allocation is not reverted
    /// between passes, so a kernel writing to it sees accumulated values; that is a narrow
    /// soundness gap, but it is strictly better than restoring the previous allocation's bytes over
    /// live data, and the window between snap_inventory() and this copy is microseconds.
    auto capture = [&](void*            gpu_addr,
                       size_t           size,
                       uint64_t         generation,
                       std::string_view what,
                       bool             from_tracker,
                       hsa_executable_t executable) -> bool {
        if(size == 0) return true;

        mem_block_t blk;
        blk.gpu_addr     = gpu_addr;
        blk.from_tracker = from_tracker;
        blk.generation   = generation;
        blk.executable   = executable;
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

        // snap_inventory() released the tracker lock before returning. A host thread calling
        // hsa_amd_memory_pool_free / hsa_memory_free can retire this allocation while we
        // read it, because the alloc/free wrappers are not covered by the per-agent replay lock.
        // The with_inventory_check helper re-checks liveness and copies under the read lock, the
        // same guard restore() applies to the write direction. A nullopt result means the region
        // was freed or shrunk after snap_inventory(), so we drop it rather than read retired device
        // memory. Module variables live in the loaded executable, not the tracker. They are always
        // present, so we copy them directly.
        const auto st =
            from_tracker
                ? with_inventory_check(
                      gpu_addr,
                      size,
                      generation,
                      [&] { return dma_copy(blk.host_copy.data(), gpu_addr, size); })
                : std::optional<hsa_status_t>{dma_copy(blk.host_copy.data(), gpu_addr, size)};

        if(!st)
        {
            ROCP_INFO << fmt::format("kernel-replay snapshot: {} {} ({}B) was retired or replaced "
                                     "before capture, dropping from snapshot",
                                     what,
                                     gpu_addr,
                                     size);
            return true;
        }

        if(*st != HSA_STATUS_SUCCESS)
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

    for(const auto& [ptr, info] : inventory)
    {
        if(!capture(ptr, info.size, info.generation, "region", /*from_tracker=*/true, {}))
        {
            out.ok = false;
            return out;
        }
    }

    // Module-scope variables (__device__ / __constant__ globals) live in the loaded executable's
    // data segment, not in the allocation tracker, so capture them here too. Restored via the same
    // per-block host->device copy as tracked allocations (see restore()).
    //
    // The liveness re-check is the read-direction counterpart of the one in restore(): an unload
    // between discover_module_variables() and the copy would have us read a segment the loader has
    // already released. Dropping such a variable is not a capture failure -- restore() would refuse
    // it for the same reason -- so replay proceeds without reverting that one global.
    const auto loaded_at_snap =
        module_vars.empty() ? std::unordered_set<uint64_t>{} : loaded_executable_handles();
    for(const auto& var : module_vars)
    {
        if(!module_region_is_restorable(loaded_at_snap, var.executable))
        {
            ROCP_INFO << fmt::format("kernel-replay snapshot: module variable {} ({}B) lost its "
                                     "executable before capture, dropping from snapshot",
                                     var.gpu_addr,
                                     var.size);
            continue;
        }
        if(!capture(
               var.gpu_addr, var.size, /*generation=*/0, "module variable", false, var.executable))
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

bool
restore(const device_snapshot_t& snapshot)
{
    // Enumerated once, and only when the snapshot actually holds module-scope regions, so a
    // snapshot of tracked allocations alone does not pay for an executable walk.
    const bool has_module_regions =
        std::any_of(snapshot.blocks.begin(), snapshot.blocks.end(), [](const mem_block_t& blk) {
            return !blk.from_tracker;
        });
    const auto loaded_executables =
        has_module_regions ? loaded_executable_handles() : std::unordered_set<uint64_t>{};

    size_t restored = 0;
    for(const auto& blk : snapshot.blocks)
    {
        hsa_status_t status = HSA_STATUS_SUCCESS;

        if(blk.from_tracker)
        {
            // Re-check identity and copy under the read lock so a concurrent free cannot race the
            // check and the write (see with_inventory_check). A nullopt result means the address no
            // longer names the allocation we captured: it was freed, shrunk, or freed and
            // reallocated at the same base (the generation stamp is what detects the last case).
            // Skipping is benign -- not a restore failure -- and is the only safe action, because
            // writing our bytes into a different allocation would corrupt live application data.
            const auto st =
                with_inventory_check(blk.gpu_addr, blk.host_copy.size(), blk.generation, [&] {
                    return dma_copy(blk.gpu_addr, blk.host_copy.data(), blk.host_copy.size());
                });

            if(!st)
            {
                ROCP_WARNING << fmt::format(
                    "kernel-replay restore: skipping region {} ({}B); it is no longer the "
                    "allocation captured at snap time (freed, shrunk, or the address was reused). "
                    "A kernel writing to it will not see identical inputs on later passes",
                    blk.gpu_addr,
                    blk.host_copy.size());
                continue;
            }
            status = *st;
        }
        else
        {
            // Module variable: it lives in its executable's data segment, so the executable being
            // loaded is what makes the address valid. The replay window serializes dispatches on
            // the agent, not code-object loading, so another thread can unload one between snap and
            // restore -- writing then would fault on memory the loader has already returned.
            if(!module_region_is_restorable(loaded_executables, blk.executable))
            {
                ROCP_WARNING << fmt::format(
                    "kernel-replay restore: skipping module variable {} ({}B); the executable that "
                    "owns it was unloaded since snap time. A kernel reading that global will not "
                    "see identical inputs on later passes",
                    blk.gpu_addr,
                    blk.host_copy.size());
                continue;
            }
            status = dma_copy(blk.gpu_addr, blk.host_copy.data(), blk.host_copy.size());
        }

        if(status != HSA_STATUS_SUCCESS)
        {
            // A live region failed to copy. Further passes would observe mutated state, and the
            // final pass (which deliberately skips restore) would leave that corruption visible to
            // the application. Abort: a partial restore cannot be undone.
            ROCP_ERROR << fmt::format(
                "kernel-replay restore: host->device copy failed for region {} ({}B); aborting "
                "restore after {}/{} regions",
                blk.gpu_addr,
                blk.host_copy.size(),
                restored,
                snapshot.blocks.size());
            return false;
        }
        ++restored;
    }

    ROCP_INFO << fmt::format(
        "kernel-replay restore: restored {}/{} regions", restored, snapshot.blocks.size());
    return true;
}

snapshot_footprint_t
estimate_footprint(hsa_agent_t agent)
{
    auto out = snapshot_footprint_t{};

    const auto tracked = memory_tracker::tracked_footprint(agent);
    out.bytes          = tracked.bytes;
    out.regions        = tracked.regions;

    // Module variables are enumerated rather than summed from the tracker, so they cost one
    // executable walk. That is the same walk snap() does and is cheap relative to any copy.
    for(const auto& var : discover_module_variables(agent))
    {
        out.bytes += var.size;
        ++out.regions;
    }
    return out;
}
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
