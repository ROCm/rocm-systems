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

#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
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
    // Tracker generation observed at snap time (see memory_tracker::alloc_info_t). restore()
    // refuses to write unless the address still carries this generation, which is what
    // distinguishes "the same allocation" from "a different allocation that reused the address".
    // Unused when from_tracker is false.
    uint64_t generation = 0;
    // Executable that owns this region, for module-scope variables only (from_tracker == false). A
    // module variable's address is valid only while its executable is loaded, and the replay window
    // gates dispatches rather than code-object loading, so another thread can unload one
    // mid-window.
    hsa_executable_t executable{};
};

// A captured set of device allocations (host-side copies) for a single agent.
struct device_snapshot_t
{
    std::vector<mem_block_t> blocks;
    // false => capture was incomplete (host memory pressure, or a failed device->host copy). The
    // snapshot must not be used to restore -- a partial restore corrupts application data -- so the
    // caller should decline replay and run the dispatch once instead.
    bool ok = true;

    bool empty() const { return blocks.empty(); }
};

// Copy (device->host) every tracked allocation owned by `agent`, plus every module-scope variable
// (__device__ / __constant__ global) visible to `agent` in the loaded executables. On success the
// returned snapshot has ok==true; if any region cannot be captured (host memory pressure -- the
// host buffer allocation fails -- or a failed copy) it returns early with ok==false so the caller
// can decline replay rather than restore a partial snapshot.
device_snapshot_t
snap(hsa_agent_t agent);

// Bytes and regions snap(agent) would capture, without copying anything. This is the tracked
// allocation footprint plus the module-scope variables, i.e. the host memory the snapshot will
// need and the volume that must cross the host link once per pass. The replay window uses it for
// admission control: a footprint the host cannot hold, or one whose projected copy time dwarfs the
// kernel being measured, is better declined up front than discovered by a std::bad_alloc partway
// through the capture or by a user watching an apparently hung job.
struct snapshot_footprint_t
{
    size_t bytes   = 0;
    size_t regions = 0;
};

snapshot_footprint_t
estimate_footprint(hsa_agent_t agent);

// Copy each saved region host->device. Returns true when every live region was restored.
// A region that was freed after snap is skipped (benign). A failed host->device DMA copy
// returns false immediately: the snapshot is then only partially applied, and the caller must
// abort replay rather than submit another pass over corrupted device memory.
bool
restore(const device_snapshot_t& snapshot);

// True when [`gpu_addr`, +`size`) in `inventory` still names the allocation that carried
// `generation`, and is therefore safe to copy.
//
// This is the predicate that keeps replay from corrupting the application, so it is named and
// exposed rather than left inline. The alloc/free wrappers deliberately sit outside the per-agent
// replay lock, so between snap() recording a region and restore() writing it back, another thread
// may free that allocation and receive the same base address for a new one -- with a caching or
// pooling allocator that is the expected outcome, not a rare race. A check on (base, size) alone
// cannot tell the two apart, and writing the old bytes over the new allocation would silently
// corrupt live data. The generation stamp is what distinguishes them.
//
// `generation == 0` means "any live allocation of at least `size` bytes", used for regions that are
// not tracker-owned (module-scope variables). No tracked allocation is ever stamped 0, so a tracked
// region can never fall back to the weaker check by accident.
bool
region_is_restorable(const memory_tracker::tracked_map_t& inventory,
                     void*                                gpu_addr,
                     size_t                               size,
                     uint64_t                             generation);

// The same question for a module-scope region, whose liveness is its executable's liveness rather
// than an entry in the tracker.
//
// `loaded` empty means "could not enumerate", not "nothing is loaded". The code-object module
// returns an empty set once it has shut down, and refusing every module variable then would quietly
// stop reverting globals between passes -- a silent accuracy loss in exchange for nothing. A
// snapshot holding module variables is itself proof that an executable was loaded, so an empty set
// is a failure to enumerate and the region is admitted; the copy still reports a fault if the
// address really is gone.
bool
module_region_is_restorable(const std::unordered_set<uint64_t>& loaded_executables,
                            hsa_executable_t                    executable);
}  // namespace memory_snapshot
}  // namespace kernel_replay
}  // namespace rocprofiler
