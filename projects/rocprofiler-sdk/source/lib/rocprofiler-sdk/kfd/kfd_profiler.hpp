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
#include <vector>

// KFD dispatch-log profiler: startup probe + GPU support discovery.
//
// This is the entry gate for the KFD dispatch-log timestamp source. At startup
// init_kfd_profiler() probes the running kernel for the profiler ioctl ABI and
// discovers which GPUs expose the dispatch-log sysfs descriptor. Every failure
// is silent and complete: the entire KFD path is skipped and all dispatches
// fall back to hsa_amd_profiling_get_dispatch_time(). Nothing here ever changes
// the completion-signal lifecycle.
//
// Lifecycle: init_kfd_profiler() is called from kfd::init(); shutdown_kfd_profiler()
// from kfd::finalize(). State is owned by the translation unit (kfd_profiler.cpp),
// not exposed as bare globals.

namespace rocprofiler
{
namespace kfd
{
// Run the startup probe: env opt-out, open /dev/kfd, profiler VERSION ioctl,
// ABI version check, then GPU discovery. Returns true only when the ABI probe
// succeeds AND at least one GPU exposes dispatch-log. Safe to call more than
// once (idempotent). Never throws.
bool
init_kfd_profiler();

// Release the probe fd and reset discovery state. Idempotent.
void
shutdown_kfd_profiler();

// True when the ABI probe passed and >=1 supported GPU was discovered. Checked
// by later stages (WriteInterceptor capture, get_dispatch_time) to gate the
// KFD path. Returns false until init_kfd_profiler() has succeeded.
bool
kfd_dispatch_log_available();

// True when the given KFD gpu_id exposes the dispatch_log_format sysfs node
// (gfx9.4.3 / 9.4.4 / 9.5.0 / gfx12.0.1). Unsupported GPUs fall back to HSA
// without affecting other GPUs.
bool
gpu_supports_dispatch_log(uint32_t gpu_id);

// Profiler ABI version reported by the kernel (0 if the probe has not run or
// the ioctl is unsupported). Exposed for logging/diagnostics.
uint32_t
kfd_profiler_abi_version();

// The set of KFD gpu_ids discovered to support dispatch-log (have the
// dispatch_log_format sysfs node). Empty until init_kfd_profiler() succeeds.
std::vector<uint32_t>
supported_dispatch_log_gpus();
}  // namespace kfd
}  // namespace rocprofiler
