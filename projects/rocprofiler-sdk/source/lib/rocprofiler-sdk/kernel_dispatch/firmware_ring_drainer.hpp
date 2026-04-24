// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#pragma once

// Standalone MEC firmware-assisted dispatch ring drainer.
//
// Drains 16-byte dispatch records written by the MEC firmware into a
// host-visible ring buffer per HSA queue, pairs START/END records by
// dispatch index, resolves kernel objects from the AQL packet ring, and
// emits ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH /
// ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH records via the standard
// tracing pipeline.
//
// This path bypasses HSA queue interception entirely. It is enabled when:
//   - The runtime exposes hsa_amd_queue_iterate and
//     hsa_amd_profiling_get_dispatch_records (resolved at runtime via dlsym
//     in lib/rocprofiler-sdk/hsa/dispatch_ring_buffer_support.{cpp,hpp})
//   - At least one registered context requests kernel-dispatch tracing
//     and is not also a PC-sampling context.
//
// See projects/rocprofiler-sdk/ai/KNOWN_ISSUES.md for the list of deferred
// issues affecting this path.

namespace rocprofiler
{
namespace kernel_dispatch
{
// Start the drainer thread. No-op if the firmware ring APIs are not
// available or if the drainer is already running.
void
start_firmware_dispatch_ring_drainer();

// Stop the drainer thread. No-op if the drainer was never started.
void
stop_firmware_dispatch_ring_drainer();
}  // namespace kernel_dispatch
}  // namespace rocprofiler
