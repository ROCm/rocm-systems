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
#include <unordered_map>
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

using device_backing_cache_t =
    std::unordered_map<uint64_t, std::unordered_map<size_t, std::vector<void*>>>;

common::Synchronized<device_backing_cache_t>&
device_backing_cache()
{
    // Keep backing allocations for process lifetime. HSA may already be unavailable during static
    // destruction, so do not attempt to free them there.
    static auto* value = new common::Synchronized<device_backing_cache_t>{};
    return *value;
}

void
release_device_copy(hsa_amd_memory_pool_t pool, size_t size, void* ptr)
{
    if(pool.handle == 0 || size == 0 || !ptr) return;
    device_backing_cache().wlock([&](auto& cache) { cache[pool.handle][size].emplace_back(ptr); });
}

bool
allocate_device_copy(hsa_amd_memory_pool_t pool, hsa_agent_t agent, size_t size, void** ptr)
{
    if(pool.handle == 0 || !ptr) return false;

    *ptr = nullptr;
    device_backing_cache().wlock([&](auto& cache) {
        auto pool_itr = cache.find(pool.handle);
        if(pool_itr == cache.end()) return;
        auto size_itr = pool_itr->second.find(size);
        if(size_itr == pool_itr->second.end() || size_itr->second.empty()) return;
        *ptr = size_itr->second.back();
        size_itr->second.pop_back();
    });
    if(*ptr) return true;

    auto* ext = hsa::get_amd_ext_table();
    if(!ext || !ext->hsa_amd_memory_pool_allocate_fn || !ext->hsa_amd_agents_allow_access_fn ||
       !ext->hsa_amd_memory_pool_free_fn)
        return false;

    auto status = ext->hsa_amd_memory_pool_allocate_fn(pool, size, 0, ptr);
    if(status != HSA_STATUS_SUCCESS || !*ptr) return false;

    status = ext->hsa_amd_agents_allow_access_fn(1, &agent, nullptr, *ptr);
    if(status == HSA_STATUS_SUCCESS) return true;

    ext->hsa_amd_memory_pool_free_fn(*ptr);
    *ptr = nullptr;
    return false;
}
}  // namespace

mem_block_t::~mem_block_t()
{
    if(!device_copy) return;
    release_device_copy(device_pool, copy_size, device_copy);
}

mem_block_t::mem_block_t(mem_block_t&& rhs) noexcept
: gpu_addr{std::exchange(rhs.gpu_addr, nullptr)}
, device_copy{std::exchange(rhs.device_copy, nullptr)}
, device_pool{std::exchange(rhs.device_pool, hsa_amd_memory_pool_t{.handle = 0})}
, copy_size{std::exchange(rhs.copy_size, 0)}
, host_copy{std::move(rhs.host_copy)}
, from_tracker{std::exchange(rhs.from_tracker, false)}
{}

mem_block_t&
mem_block_t::operator=(mem_block_t&& rhs) noexcept
{
    if(this == &rhs) return *this;
    this->~mem_block_t();
    new(this) mem_block_t{std::move(rhs)};
    return *this;
}

device_snapshot_t
snap(hsa_agent_t agent)
{
    return snap(agent, hsa_amd_memory_pool_t{.handle = 0});
}

device_snapshot_t
snap(hsa_agent_t agent, hsa_amd_memory_pool_t gpu_pool)
{
    device_snapshot_t out{};

    // Note: trackable allocations carrying HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG are recorded in
    // memory_tracker::unsupported_executable() and omitted from the main inventory. Declining
    // replay whenever that side inventory is non-empty is not viable -- the HIP runtime (and the
    // SDK's own AQL pools) routinely keep such allocations live -- so they remain an unsupported
    // omitted class for beta (documented in the public header). Direct-HSA apps that put ordinary
    // writable device data behind the flag observe the same omission.

    const auto [inventory, module_vars] = [&]() {
        try
        {
            auto inv   = memory_tracker::snap_inventory(agent);
            auto mvars = discover_module_variables(agent);

            out.blocks.reserve(inv.size() + mvars.size());
            return std::pair{std::move(inv), std::move(mvars)};
        } catch(const std::bad_alloc&)
        {
            ROCP_FATAL << "kernel-replay snapshot: out of memory reserving metadata";
        }
    }();

    size_t device_backed = 0;
    size_t host_backed   = 0;

    /// @brief Capture one region into GPU-local backing when available, otherwise host memory.
    /// @retval false The snapshot is incomplete because backing allocation or the copy failed. The
    /// caller must decline replay rather than restore partial state.
    /// @retval true The region was captured. For tracker regions it may instead have been dropped
    /// after being freed or shrunk since snap_inventory(). That drop is safe because restore() runs
    /// the same liveness check, so a region gone now could not have been restored anyway.
    auto capture =
        [&](void* gpu_addr, size_t size, std::string_view what, bool from_tracker) -> bool {
        if(size == 0) return true;

        mem_block_t blk;
        blk.gpu_addr     = gpu_addr;
        blk.copy_size    = size;
        blk.from_tracker = from_tracker;

        if(allocate_device_copy(gpu_pool, agent, size, &blk.device_copy))
        {
            blk.device_pool = gpu_pool;
            ++device_backed;
        }
        else
        {
            try
            {
                blk.host_copy.resize(size);
                ++host_backed;
            } catch(const std::bad_alloc&)
            {
                ROCP_WARNING << fmt::format("kernel-replay snapshot: host allocation of {} "
                                            "bytes failed for {} {} (memory pressure)",
                                            size,
                                            what,
                                            gpu_addr);
                return false;
            }
        }
        auto* snapshot_addr =
            blk.device_copy ? blk.device_copy : static_cast<void*>(blk.host_copy.data());

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
                      gpu_addr, size, [&] { return dma_copy(snapshot_addr, gpu_addr, size); })
                : std::optional<hsa_status_t>{dma_copy(snapshot_addr, gpu_addr, size)};

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
                "kernel-replay snapshot: copy failed for {} {} ({}B)", what, gpu_addr, size);
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

    ROCP_INFO << fmt::format("kernel-replay snapshot: captured {} regions ({} GPU-local, {} host) "
                             "for agent {}",
                             out.blocks.size(),
                             device_backed,
                             host_backed,
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
            const auto st = with_inventory_check(blk.gpu_addr, blk.copy_size, [&] {
                return dma_copy(blk.gpu_addr, blk.saved_data(), blk.copy_size);
            });

            if(!st)
            {
                ROCP_WARNING << fmt::format(
                    "kernel-replay restore: skipping region {} ({}B) that is no longer live",
                    blk.gpu_addr,
                    blk.copy_size);
                continue;
            }
            status = *st;
        }
        else
        {
            // Module variable: lives in the loaded executable, always present, so restore directly.
            status = dma_copy(blk.gpu_addr, blk.saved_data(), blk.copy_size);
        }

        if(status != HSA_STATUS_SUCCESS)
        {
            // A live region failed to copy. Further passes would observe mutated state, and the
            // final pass (which deliberately skips restore) would leave that corruption visible to
            // the application. Abort: a partial restore cannot be undone.
            ROCP_ERROR << fmt::format(
                "kernel-replay restore: copy failed for region {} ({}B). Aborting "
                "restore after {}/{} regions",
                blk.gpu_addr,
                blk.copy_size,
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

bool
restore(const device_snapshot_t& snapshot, const batch_copy_fn_t& batch_copy)
{
    std::vector<blit::copy_region_t> device_regions;
    device_regions.reserve(snapshot.blocks.size());
    size_t restored = 0;
    size_t skipped  = 0;

    const auto status = memory_tracker::inventory().rlock([&](const auto& map) {
        for(const auto& blk : snapshot.blocks)
        {
            if(blk.from_tracker)
            {
                auto itr = map.find(blk.gpu_addr);
                if(itr == map.end() || itr->second.size < blk.copy_size)
                {
                    ROCP_WARNING << fmt::format(
                        "kernel-replay restore: skipping region {} ({}B) that is no longer live",
                        blk.gpu_addr,
                        blk.copy_size);
                    ++skipped;
                    continue;
                }
            }

            if(blk.device_copy)
            {
                device_regions.emplace_back(
                    blit::copy_region_t{blk.gpu_addr, blk.saved_data(), blk.copy_size});
            }
            else
            {
                auto copy_status = dma_copy(blk.gpu_addr, blk.saved_data(), blk.copy_size);
                if(copy_status != HSA_STATUS_SUCCESS) return copy_status;
                ++restored;
            }
        }

        if(!device_regions.empty())
        {
            auto copy_status = batch_copy(device_regions);
            if(copy_status != HSA_STATUS_SUCCESS) return copy_status;
            restored += device_regions.size();
        }
        return HSA_STATUS_SUCCESS;
    });

    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << fmt::format("kernel-replay restore: batch copy failed after {}/{} regions",
                                  restored,
                                  snapshot.blocks.size());
        return false;
    }

    ROCP_INFO << fmt::format("kernel-replay restore: restored {}/{} regions ({} skipped)",
                             restored,
                             snapshot.blocks.size(),
                             skipped);
    return true;
}
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
