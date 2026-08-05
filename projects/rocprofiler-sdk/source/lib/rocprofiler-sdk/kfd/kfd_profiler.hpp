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

// KFD dispatch-log profiler: startup probe + GPU support discovery. Every
// failure is silent and complete -- the KFD path is skipped and dispatches fall
// back to hsa_amd_profiling_get_dispatch_time(). Never touches signal lifecycle.

namespace rocprofiler
{
namespace kfd
{
// Env opt-out, open /dev/kfd, profiler VERSION ioctl, ABI check, GPU discovery.
// Idempotent, never throws. Outcome published via kfd_dispatch_log_available().
void
init_kfd_profiler();

// Stop the reader and reset discovery state. Idempotent.
void
shutdown_kfd_profiler();

// Async-signal-safe (one atomic store), so the atfork child handler can call it.
void
disable_kfd_dispatch_log();

// True only after the ABI probe passed, >=1 supported GPU was found, and the
// reader started; false again in a forked child.
bool
kfd_dispatch_log_available();

// gfx9.4.3 / 9.4.4 / 9.5.0 / gfx12.0.1. Unsupported GPUs fall back to HSA.
bool
gpu_supports_dispatch_log(uint32_t gpu_id);

// Called when a context tracing kernel dispatch starts, not at startup: arming
// registers a GTT buffer and opens a firmware stream, and installing the HSA
// table says nothing about whether anyone wants kernel traces. Idempotent.
void
arm_dispatch_log_sessions();
}  // namespace kfd
}  // namespace rocprofiler
