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
}  // namespace

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

    // Host memory pressure while building the inventory is the same condition the per-region
    // capture below reports through ok==false, and snap()'s contract is that the caller declines
    // replay and runs the dispatch once. Aborting here instead would kill the application over an
    // opt-in beta feature, and a large allocation inventory is exactly when it would happen.
    auto inventory = memory_tracker::alloc_map_t{};
    auto scan      = module_variable_scan_t{};
    try
    {
        inventory = memory_tracker::snap_inventory(agent);
        scan      = discover_module_variables(agent);

        out.blocks.reserve(inventory.size() + scan.found.size());
    } catch(const std::bad_alloc&)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel-replay snapshot: out of memory reserving metadata; "
                                   "declining replay for this dispatch";
        out.ok = false;
        return out;
    }

    // An executable or symbol HSA would not tell us about may hold a writable __device__ global. We
    // cannot restore what we did not capture, so the passes would not see identical inputs.
    if(scan.incomplete)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel-replay snapshot: could not enumerate module-scope "
                                   "variables for every loaded executable; declining replay for "
                                   "this dispatch";
        out.ok = false;
        return out;
    }

    const auto& module_vars = scan.found;

    /// @brief Capture one region (device->host) into the snapshot.
    /// @retval false The snapshot is incomplete because a host allocation or the DMA copy failed.
    /// The caller must decline replay rather than restore partial state.
    /// @retval true The region was captured. For tracker regions it may instead have been dropped
    /// after being freed or shrunk since snap_inventory(). That drop is safe because restore() runs
    /// the same liveness check, so a region gone now could not have been restored anyway.
    auto capture =
        [&](void* gpu_addr, size_t size, std::string_view what, bool from_tracker) -> bool {
        if(size == 0) return true;

        auto blk         = mem_block_t{};
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
                      [&] { return dma_copy(blk.host_copy.data(), gpu_addr, size); })
                : std::optional<hsa_status_t>{dma_copy(blk.host_copy.data(), gpu_addr, size)};

        if(!st)
        {
            ROCP_INFO << fmt::format("kernel-replay snapshot: {} {} ({}B) was retired before "
                                     "capture, dropping from snapshot",
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
    for(const auto& var : module_vars)
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

bool
restore(const device_snapshot_t& snapshot)
{
    size_t restored = 0;
    for(const auto& blk : snapshot.blocks)
    {
        hsa_status_t status = HSA_STATUS_SUCCESS;

        if(blk.from_tracker)
        {
            // Re-check liveness and copy under the read lock so a concurrent free cannot race the
            // check and the write (see with_inventory_check). A nullopt result means the region is
            // no longer a live allocation of at least its size. It was freed or reallocated after
            // snap, so skip it (benign -- not a restore failure).
            const auto st = with_inventory_check(blk.gpu_addr, blk.host_copy.size(), [&] {
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
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
