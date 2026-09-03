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
#include <unordered_set>

// KFD dispatch-log profiler: startup probe + GPU support discovery. Every
// failure is silent and complete -- the KFD path is skipped and dispatches fall
// back to hsa_amd_profiling_get_dispatch_time(). Never touches signal lifecycle.

namespace rocprofiler
{
namespace kfd
{
// Walk the KFD topology under @nodes_path and return the gpu_ids of nodes that
// expose the stream-ABI dispatch-log contract (dispatch_log_stream_format).
// Pure filesystem scan, parameterized on the root for testability. CPU-only
// nodes (gpu_id == 0) and nodes exposing only the legacy GFX9 dispatch_log_format
// are excluded -- the stream reader speaks stream ABI v3 and must not treat the
// retired per-record format selector as support.
std::unordered_set<uint32_t>
discover_stream_dispatch_log_gpus(const char* nodes_path);

// Env opt-out, open /dev/kfd, profiler VERSION ioctl, ABI check, GPU discovery.
// Idempotent, never throws. Starts no reader thread; it arms the ring eagerly only
// when a kernel-dispatch context is already registered (F15), otherwise a process
// with no consumer of kernel-dispatch data gains no SDK-internal thread and the
// ring is armed lazily on first dispatch. Outcome via kfd_dispatch_log_supported().
void
init_kfd_profiler();

// Stop the reader and reset discovery state. Idempotent.
void
shutdown_kfd_profiler();

// Async-signal-safe (atomic stores only, no lock or allocation), so the atfork
// child handler can call it.
void
disable_kfd_dispatch_log();

// True only once the reader thread and its fork handler exist; false again in a
// forked child. Use this for anything whose safety depends on the reader running.
bool
kfd_dispatch_log_available();

// True when the ABI probe passed and >=1 supported GPU was found. Capability
// only: the reader may not have been started yet.
bool
kfd_dispatch_log_supported();

// Publish that the reader thread is running. Called by the reader once it has
// started, which is the only thing that makes kfd_dispatch_log_available() true.
void
note_kfd_reader_started();

// True for GPUs exposing the stream ABI (gfx12.0.0 / 12.0.1); the legacy GFX9
// dispatch_log_format path is a separate reader. Unsupported GPUs fall back to HSA.
bool
gpu_supports_dispatch_log(uint32_t gpu_id);

// Called when a context tracing kernel dispatch starts, not at startup: arming
// opens a KFD-owned dispatch-log stream, and installing the HSA table says
// nothing about whether anyone wants kernel traces. Idempotent.
void
arm_dispatch_log_sessions();

// Non-constructing: true iff the profiler_state singleton has been constructed in
// this process. Used to verify flag-off inertness (the feature must construct no
// state). Never constructs it.
bool
kfd_profiler_state_constructed();
}  // namespace kfd
}  // namespace rocprofiler
