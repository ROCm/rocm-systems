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

// Execution half of range replay: taking the range-entry snapshot, and re-running the recorded
// dispatches once the range closes.
//
// Two snapshots of the range's agent are involved, and the difference matters:
//  - the ENTRY snapshot, captured before the range's first dispatch runs, is what each replayed
//    pass is restored to, so every pass starts from the same inputs the application's own execution
//    started from;
//  - the EXIT snapshot, captured when the range closes, is restored after the last pass, so the
//    application resumes with exactly the state its own execution produced. Without it a range that
//    is not self-contained (a kernel reading memory the snapshot does not cover) would leave the
//    replay's output behind for the application to consume.
//
// Both snapshots are held in host memory at once, so peak host usage is roughly twice the device
// memory the snapshot covers, and a third transient snapshot is taken when divergence checking is
// enabled.

#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hsa/hsa_ext_amd.h>

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
class Queue;
}  // namespace hsa

namespace range_replay
{
// Capture the range-entry snapshot, if this range is bound and does not have one yet. Called from
// the queue path while recording the range's first submission, before that submission reaches the
// GPU: `writer` is the interceptor's packet writer, used to drain the queue so the snapshot sees
// settled memory. A capture failure declines the range (SNAPSHOT_FAILED).
void
ensure_entry_snapshot(range_context_t&                      ctx,
                      const hsa::Queue&                     queue,
                      hsa_amd_queue_intercept_packet_writer writer);

// Re-execute the recorded range. Returns the outcome to report at CLOSE, and writes the number of
// divergent snapshot regions to `divergence_count` (0 unless divergence checking is enabled).
// Blocks the calling thread; holds the range's agent replay lock for the whole window.
rocprofiler_range_replay_status_t
execute_range(range_context_t& ctx, uint64_t& divergence_count);

// True when divergence checking is enabled (ROCPROF_RANGE_REPLAY_VERIFY). Reads the environment
// once.
bool
divergence_check_enabled();
}  // namespace range_replay
}  // namespace rocprofiler
