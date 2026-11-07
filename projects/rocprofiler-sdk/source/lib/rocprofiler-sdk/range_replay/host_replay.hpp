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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Host replay: range replay's alternate execution path, selected by the tool setting host_replay at
// CONFIG and the application opening the range with ROCPROFILER_RANGE_REPLAY_HOST_BEGIN.
//
// Packet replay records a range's dispatch packets and re-submits them, so it can only replay a
// range whose whole effect is those packets on one queue. Host replay records nothing: the
// application's code between begin and end is the program. end() restores the state the range
// started from and siglongjmps to the sigsetjmp the application took just before begin(), and the
// application executes the range again. Copies, memsets, several queues and agents, graph launches,
// balanced allocate/free pairs and host computation are therefore part of every pass.
//
// What has to hold for the passes to be repeatable, and how each part is enforced:
//  - Device memory on every GPU agent is snapshotted at begin and restored before every extra pass.
//    The tracked allocations must be the same set at each restore: a balanced allocate/free inside
//    the range is fine, anything else declines ALLOCATION_CHANGED_IN_RANGE.
//  - Host memory is rewound only where the application registered it
//    (rocprofiler_range_replay_host_track_memory). The divergence check reports a final pass that
//    ended in a different state from the application's own run.
//  - Other threads must leave the devices alone: their dispatches decline CONCURRENT_DISPATCH and
//    their device copies decline MEMORY_COPY_IN_RANGE, since a restore would race or erase them.
//  - A code object loaded during the application's run of the range (HIP loads a module on the
//    first launch of one of its kernels) is accepted with a warning: its module variables are not
//    in the entry snapshot, so they are not rewound.
//
// No exit snapshot is taken: the final pass re-ran the host code as well, so host and device state
// at the end are mutually consistent, and the application continues from them.

#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace range_replay
{
namespace host
{
// Outcome of one end() of a host-replay range. next_pass > 0: the caller must siglongjmp with that
// value once every C++ object in its frame is destroyed. next_pass == 0: the range is finished and
// status/divergence_count are what CLOSE reports.
struct step_result_t
{
    uint64_t                          next_pass        = 0;
    rocprofiler_range_replay_status_t status           = ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
    uint64_t                          divergence_count = 0;
};

// Called from begin() after CONFIG. Marks the range as host replay and, when the tool asked for
// extra passes, fences every GPU agent and captures its device memory plus the pending host
// regions.
void
enter_range(range_context_t& ctx, std::vector<pending_host_region_t>&& pending_regions);

// Register a host region with an open host-replay range. Captured now during the application's own
// run; ignored during re-executed passes.
void
track_region(range_context_t& ctx, void* ptr, size_t size);

// Called from end(): finish the pass that just ran and either set up the next one (restore, PASS
// enter) or conclude the range.
step_result_t
finish_pass(range_context_t& ctx);

// Capture and restore of registered host regions. GPU-free.
host_region_t
capture_region(void* ptr, size_t size);

void
restore_regions(const std::vector<host_region_t>& regions);
}  // namespace host
}  // namespace range_replay
}  // namespace rocprofiler
