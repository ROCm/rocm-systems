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

// KFD dispatch-log reader thread: a background thread owning the dispatch-log
// data ring. It sets up the KFD session, then poll()s one control eventfd plus
// every live stream fd (with a 10 ms timeout as the sparse-tail / lost-interrupt
// watchdog), drains firmware records, pairs dispatch_start + eop, and deposits
// paired timings into the DispatchHub. wptr != rptr is the sole drain authority;
// poll wakes are only hints. Finalization uses nudge_reader() +
// wait_for_reader_quiesce() so the poll timeout never delays teardown.
//
// Started lazily by the first establish_session() -- which is reached only from a
// kernel-dispatch-tracing context arming, or from an intercepted dispatch -- so a
// process with no consumer of kernel-dispatch data never creates it. Stopped from
// shutdown_kfd_profiler(). Both are idempotent and safe when the dispatch-log is
// unavailable.

#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
// No-op if already running, and safe regardless of whether any GPU supports
// dispatch-log. On false the caller must not advertise the dispatch-log as
// available; the reader has already released everything it acquired.
//
// NOT thread-safe: callers must serialize. The only caller is establish_session(),
// which holds reader_state::setup_mu.
bool
start_kfd_reader();

// Signal the reader thread to stop and join it. Idempotent.
void
stop_kfd_reader();

// Non-constructing: true iff the reader_state singleton has been constructed in
// this process. Used to verify flag-off inertness. Never constructs it.
bool
kfd_reader_state_constructed();

// True when the reader can never serve a session (dead/stopping), as opposed to a
// single GPU failing to arm. N1's fence short-circuits to abandon on this: a
// reader that is gone will not quiesce within the budget. Non-constructing.
bool
reader_unavailable();

// Ensure a dispatch-log session exists for the given gpu_id. Returns true only
// when a live session belongs to THIS gpu_id; callers must otherwise leave the
// correlation key invalid and fall back to HSA.
bool
ensure_reader_session(uint32_t gpu_id);

// Arm the ring before any queue exists: firmware only records once the stream is
// open, so arming on first dispatch loses the earliest dispatches. Unlike
// ensure_reader_session(), a merely-too-early failure does not latch the GPU off.
bool
arm_reader_session_early(uint32_t gpu_id);

// Break the reader out of its poll so it copies now. Safe from any thread.
void
nudge_reader();

// The two-stage fence wait: Stage 1 waits drain_epoch to advance by 2
// (advanced only when every session's overflow is drained), proving every record
// present in the ring at the call was copied AND published into the pipe; Stage 2
// polls pipe.empty(), proving every such batch was popped and fully processed.
// Bounded by the deadline; returns false on timeout (the caller abandons) and an
// immediate success when the reader never started.
bool
wait_for_reader_quiesce(uint64_t timeout_ns = 100'000'000);
}  // namespace kfd
}  // namespace rocprofiler
