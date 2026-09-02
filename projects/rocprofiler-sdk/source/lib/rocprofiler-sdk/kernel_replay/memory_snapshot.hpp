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

#include <hsa/hsa.h>

#include <cstddef>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
// Minimal save/restore of device memory for kernel replay.
//
// snap(agent) copies every tracked device allocation owned by `agent` into host memory; restore()
// copies it back. This keeps each replay pass running against identical inputs, scoped to one agent
// so concurrent replays on other agents are unaffected. The implementation is deliberately simple:
// a full copy of each region (no dirty-page diffing) of directly-allocated device memory.
namespace memory_snapshot
{
// Saved copy of a single device allocation.
struct mem_block_t
{
    void*             gpu_addr = nullptr;  // live device allocation base pointer
    std::vector<char> host_copy;           // pre-kernel contents held in host memory
    // true  = from the allocation tracker; re-check liveness before restoring (it can be freed).
    // false = module-scope variable in a loaded executable; always live, so restore
    // unconditionally.
    bool from_tracker = false;
};

// A captured set of device allocations (host-side copies) for a single agent.
struct device_snapshot_t
{
    std::vector<mem_block_t> blocks;
    // false => capture was incomplete: host memory pressure, a failed device->host copy, or HSA
    // would not enumerate the module-scope variables of every loaded executable. The snapshot must
    // not be used to restore -- a partial restore corrupts application data -- so the caller should
    // decline replay and run the dispatch once instead.
    bool ok = true;

    bool empty() const { return blocks.empty(); }
};

// Copy (device->host) every tracked allocation owned by `agent`, plus every module-scope variable
// (__device__ / __constant__ global) visible to `agent` in the loaded executables. On success the
// returned snapshot has ok==true. It returns early with ok==false, so the caller can decline replay
// rather than restore a partial snapshot, when a region cannot be captured (host memory pressure --
// the host buffer allocation fails -- or a failed copy) or when the module-scope variables of a
// loaded executable cannot be enumerated. Host memory pressure is never fatal: a dispatch that
// cannot be snapshotted runs once instead of aborting the application.
device_snapshot_t
snap(hsa_agent_t agent);

// Copy each saved region host->device. Returns true when every live region was restored.
// A region that was freed after snap is skipped (benign). A failed host->device DMA copy
// returns false immediately: the snapshot is then only partially applied, and the caller must
// abort replay rather than submit another pass over corrupted device memory.
bool
restore(const device_snapshot_t& snapshot);
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
