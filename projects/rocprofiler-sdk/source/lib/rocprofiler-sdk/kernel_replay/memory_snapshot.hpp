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

#pragma once

#include "lib/rocprofiler-sdk/kernel_replay/blit-copy.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
// Save/restore of one agent's tracked device memory for kernel replay. A pure planning pass assigns
// each complete region to GPU-local or pinned-host storage, then capture materializes at most one
// arena per tier before reading application memory.
namespace memory_snapshot
{
enum class storage_kind : uint8_t
{
    none,
    gpu_local,
    pinned_host,
    mixed,
};

enum class liveness_kind : uint8_t
{
    tracked_allocation,
    module_variable,
    retired,
};

enum class capture_status : uint8_t
{
    complete,
    unavailable,
    invalid_plan,
    copy_failed,
};

enum class capture_mode : uint8_t
{
    automatic,
    force_gpu,
    force_pinned,
};

struct block_t
{
    size_t size = 0;
    void*  data = nullptr;

    explicit operator bool() const { return data != nullptr; }
};

// The only non-trivial storage object. GPU-local blocks return to the replay backing cache; pinned
// host blocks return directly to their HSA pool.
class owned_block_t
{
public:
    owned_block_t() = default;
    owned_block_t(block_t, storage_kind, hsa_amd_memory_pool_t, bool cacheable);
    ~owned_block_t();

    owned_block_t(const owned_block_t&) = delete;
    owned_block_t& operator=(const owned_block_t&) = delete;
    owned_block_t(owned_block_t&& rhs) noexcept;
    owned_block_t& operator=(owned_block_t&& rhs) noexcept;

    block_t      block() const { return m_block; }
    storage_kind kind() const { return m_kind; }
    explicit     operator bool() const { return static_cast<bool>(m_block); }

    void reset() noexcept;

private:
    block_t               m_block{};
    storage_kind          m_kind = storage_kind::none;
    hsa_amd_memory_pool_t m_pool{.handle = 0};
    bool                  m_cacheable = false;
};

struct snapshot_region_t
{
    size_t        size           = 0;
    void*         live_address   = nullptr;
    size_t        logical_offset = 0;
    liveness_kind liveness       = liveness_kind::tracked_allocation;
};

struct storage_segment_t
{
    size_t        size           = 0;
    size_t        logical_offset = 0;
    owned_block_t backing{};
};

struct snapshot_extent_t
{
    size_t size = 0;

    size_t region_index  = 0;
    size_t region_offset = 0;

    size_t segment_index  = 0;
    size_t segment_offset = 0;
};

struct capture_context_t
{
    hsa_agent_t            agent{};
    rocprofiler_agent_id_t agent_id{};
    hsa_agent_t            pinned_agent{};
    hsa_amd_memory_pool_t  gpu_pool{};
    hsa_amd_memory_pool_t  pinned_pool{};
    capture_mode           mode                  = capture_mode::automatic;
    bool                   gpu_backend_available = true;
};

struct storage_budgets_t
{
    size_t gpu_bytes  = 0;
    size_t host_bytes = 0;
};

// A captured set of device allocations for a single agent.
struct snapshot_t
{
    hsa_agent_t                    agent{};
    hsa_agent_t                    pinned_agent{};
    rocprofiler_agent_id_t         agent_id{};
    std::vector<snapshot_region_t> regions{};
    std::vector<storage_segment_t> segments{};
    std::vector<snapshot_extent_t> extents{};
    capture_status                 status{capture_status::unavailable};

    bool         complete() const { return status == capture_status::complete; }
    bool         empty() const { return regions.empty(); }
    size_t       footprint_bytes() const;
    storage_kind storage() const;
};

std::optional<size_t>
assign_logical_offsets(std::vector<snapshot_region_t>&);

std::optional<std::vector<snapshot_extent_t>>
map_regions_to_storage(const std::vector<snapshot_region_t>&,
                       const std::vector<storage_segment_t>&);

// Plan whole-region placement against queried GPU and host budgets, materialize at most one arena
// per tier, and capture only after every required arena exists.
snapshot_t
snap(const capture_context_t&);

snapshot_t
snap(const capture_context_t&, storage_budgets_t);

// Direct-test convenience: force the nearest CPU agent's pinned HSA pool.
snapshot_t
snap(hsa_agent_t agent);

// Copy each saved region back to its live device allocation. A region freed after snap is skipped.
// A failed copy returns false immediately because the snapshot is then only partially applied.
bool
restore(const snapshot_t& snapshot);

using batch_copy_fn_t = std::function<hsa_status_t(const std::vector<blit::copy_region_t>&)>;

// GPU-local storage restores through one batch blit callback. Pinned-host storage restores
// synchronously and invokes the callback with no copy regions so the caller submits the target.
bool
restore(const snapshot_t& snapshot, const batch_copy_fn_t& batch_copy);
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
