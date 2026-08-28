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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
// Minimal save/restore of device memory for kernel replay.
//
// snap(agent) copies every tracked device allocation owned by `agent` into snapshot backing and
// restore() copies it back. The replay path prefers GPU-local backing and falls back to host memory
// under device-memory pressure. This keeps each replay pass running against identical inputs,
// scoped to one agent so concurrent replays on other agents are unaffected.
namespace memory_snapshot
{
// Saved copy of a single device allocation.
struct mem_block_t
{
    mem_block_t() = default;
    ~mem_block_t();

    mem_block_t(const mem_block_t&) = delete;
    mem_block_t& operator=(const mem_block_t&) = delete;
    mem_block_t(mem_block_t&& rhs) noexcept;
    mem_block_t& operator=(mem_block_t&& rhs) noexcept;

    void*                 gpu_addr    = nullptr;  // live application allocation base pointer
    void*                 device_copy = nullptr;  // GPU-local snapshot backing when available
    hsa_amd_memory_pool_t device_pool{.handle = 0};
    size_t                copy_size = 0;
    std::vector<char>     host_copy;  // fallback backing when GPU-local allocation fails
    // true  = from the allocation tracker; re-check liveness before restoring (it can be freed).
    // false = module-scope variable in a loaded executable; always live, so restore
    // unconditionally.
    bool from_tracker = false;

    const void* saved_data() const
    {
        return device_copy ? device_copy : static_cast<const void*>(host_copy.data());
    }
};

// A captured set of device allocations for a single agent.
struct device_snapshot_t
{
    std::vector<mem_block_t> blocks;
    // false => capture was incomplete (backing-memory pressure, or a failed copy). The
    // snapshot must not be used to restore -- a partial restore corrupts application data -- so the
    // caller should decline replay and run the dispatch once instead.
    bool ok = true;

    bool empty() const { return blocks.empty(); }
};

// Copy every tracked allocation owned by `agent`, plus every module-scope variable
// (__device__ / __constant__ global) visible to `agent`, into snapshot backing. On success the
// returned snapshot has ok==true. A failed backing allocation or copy returns ok==false so the
// caller can decline replay rather than restore a partial snapshot.
device_snapshot_t
snap(hsa_agent_t agent);

// Prefer GPU-local snapshot backing from `gpu_pool`. A region falls back to host memory when its
// GPU-local allocation cannot be created. Internal allocations bypass the tracker, so a snapshot
// never captures its own backing.
device_snapshot_t
snap(hsa_agent_t agent, hsa_amd_memory_pool_t gpu_pool);

// Copy each saved region back to its live device allocation. A region freed after snap is skipped.
// A failed copy returns false immediately because the snapshot is then only partially applied.
bool
restore(const device_snapshot_t& snapshot);

using batch_copy_fn_t = std::function<hsa_status_t(const std::vector<blit::copy_region_t>&)>;

// Restore GPU-local blocks through a batch copy callback. Host-backed fallback blocks retain the
// synchronous ROCr path.
bool
restore(const device_snapshot_t& snapshot, const batch_copy_fn_t& batch_copy);
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
