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

#include <cstdint>

// KFD dispatch-log profiler: startup probe + GPU support discovery.
//
// This is the entry gate for the KFD dispatch-log timestamp source. At startup
// init_kfd_profiler() probes the running kernel for the profiler ioctl ABI and
// discovers which GPUs expose the dispatch-log sysfs descriptor. Every failure
// is silent and complete: the entire KFD path is skipped and all dispatches
// fall back to hsa_amd_profiling_get_dispatch_time(). Nothing here ever changes
// the completion-signal lifecycle.
//
// Lifecycle: init_kfd_profiler() is called from queue_interposition::
// interposition_init(), i.e. only when inline interposition -- the sole path that
// produces a correlation key -- is selected; shutdown_kfd_profiler() from
// kfd::finalize(). State is owned by the translation unit (kfd_profiler.cpp), not
// exposed as bare globals.

namespace rocprofiler
{
namespace kfd
{
// Run the startup probe: env opt-out, open /dev/kfd, profiler VERSION ioctl,
// ABI version check, then GPU discovery. The outcome is published through
// kfd_dispatch_log_available(). Safe to call more than once (idempotent).
// Never throws.
void
init_kfd_profiler();

// Stop the reader and reset discovery state. Idempotent.
void
shutdown_kfd_profiler();

// Latch the dispatch-log path off. Async-signal-safe (one atomic store), so it
// is callable from the pthread_atfork child handler.
void
disable_kfd_dispatch_log();

// Gates ensure_reader_session() and the queue-destroy doorbell retirement. True
// only after the ABI probe passed, >=1 supported GPU was discovered, and
// start_kfd_reader() succeeded; false again in a forked child.
bool
kfd_dispatch_log_available();

// True when the given KFD gpu_id exposes the dispatch_log_format sysfs node
// (gfx9.4.3 / 9.4.4 / 9.5.0 / gfx12.0.1). Unsupported GPUs fall back to HSA
// without affecting other GPUs.
bool
gpu_supports_dispatch_log(uint32_t gpu_id);

// Arm the dispatch-log ring(s), called when a context that traces kernel dispatch
// starts.
//
// Deliberately NOT done at startup: installing the HSA table says nothing about
// whether anyone wants kernel traces, and arming registers a GTT buffer and opens
// a firmware stream. Configuration time is still early enough that the ring is
// live before the first dispatch, and a tool that configures tracing later simply
// arms later -- sessions are per-GPU and independent of queues. The first-dispatch
// fallback in ensure_reader_session() still covers anything in between.
//
// Idempotent and safe to call from any context start.
void
arm_dispatch_log_sessions();
}  // namespace kfd
}  // namespace rocprofiler
