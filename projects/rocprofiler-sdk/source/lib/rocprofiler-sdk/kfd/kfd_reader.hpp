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
// data ring. It sets up the KFD session, drains firmware records, pairs
// dispatch_start + eop, and deposits paired timings into the ResultsMap.
//
// Started from init_kfd_profiler(), stopped from shutdown_kfd_profiler(); both
// idempotent and safe when the dispatch-log is unavailable.
// probe + GPU discovery succeed; stop_kfd_reader() from shutdown_kfd_profiler().
// Both are idempotent and safe to call when KFD dispatch-log is unavailable.

#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
// No-op if already running, and safe regardless of whether any GPU supports
// dispatch-log. On false the caller must not advertise the dispatch-log as
// available; the reader has already released everything it acquired.
bool
start_kfd_reader();

// Signal the reader thread to stop and join it. Idempotent.
void
stop_kfd_reader();

// Ensure a dispatch-log session exists for the given gpu_id. Returns true only
// when a live session belongs to THIS gpu_id; callers must otherwise leave the
// correlation key invalid and fall back to HSA.
bool
ensure_reader_session(uint32_t gpu_id);

// Arm the ring before any queue exists: firmware only records once the buffer is
// registered, so arming on first dispatch loses the earliest dispatches. Unlike
// ensure_reader_session(), a merely-too-early failure does not latch the GPU off.
bool
arm_reader_session_early(uint32_t gpu_id);

// Retained starts are processor-thread-owned, so a destroying thread posts a
// request rather than touching them; results have their own lock and go now.
void
request_reader_slot_purge(uint32_t gpu_id, uint32_t doorbell_slot);

// Break the reader out of its poll so it copies now. Safe from any thread.
void
nudge_reader();

// Block until a full drain cycle has completed, so records already held became
// handoffs. Bounded: returns false on timeout rather than hanging finalization.
bool
wait_for_reader_drain_barrier(uint64_t timeout_ns = 100'000'000);
}  // namespace kfd
}  // namespace rocprofiler
