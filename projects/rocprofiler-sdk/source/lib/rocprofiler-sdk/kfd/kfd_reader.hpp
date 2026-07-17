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

// KFD dispatch-log reader thread.
//
// A dedicated background thread (modeled on kfd.cpp's poll_kfd_t bg_thread) that
// owns the dispatch-log data ring: it sets up the KFD session (register buffer /
// open stream / mmap), drains firmware records, pairs dispatch_start + eop, binds
// doorbells into the DoorbellMap, and deposits paired timings into the ResultsMap
// keyed by correlation_key.
//
// Lifecycle: start_kfd_reader() is called from init_kfd_profiler() once the ABID
// probe + GPU discovery succeed; stop_kfd_reader() from shutdown_kfd_profiler().
// Both are idempotent and safe to call when KFD dispatch-log is unavailable.

#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
// Start the reader thread. No-op if already running. Safe to call regardless of
// whether any GPU supports dispatch-log (it simply idles if there is nothing to
// read).
void
start_kfd_reader();

// Signal the reader thread to stop and join it. Idempotent.
void
stop_kfd_reader();

// Ensure a dispatch-log session exists for the given gpu_id. Called from the
// queue-creation path (queue_controller.cpp), which guarantees the SDK's HSA
// agent cache is populated and the device is acquired -- the preconditions the
// session's HSA allocation needs. Idempotent: sets up at most one session per
// gpu_id. No-op if KFD dispatch-log is unavailable or the GPU is unsupported.
void
ensure_reader_session(uint32_t gpu_id);
}  // namespace kfd
}  // namespace rocprofiler
