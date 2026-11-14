// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <hsa/hsa.h>

#include <cstdint>
#include <optional>

namespace rocprofiler
{
namespace kernel_dispatch
{
using profiling_time = tracing::profiling_time;

// get the profiling time for a signal on an agent, if start time is less than baseline, correct to
// start at baseline. If kernel_id is provided, it will be included in error log message if there is
// an issue with
profiling_time
get_dispatch_time(hsa_agent_t             agent,
                  hsa_signal_t            signal,
                  rocprofiler_kernel_id_t kernel_id,
                  std::optional<uint64_t> baseline = {});

// What the host reads back from an SDK-owned completion signal in device memory.
struct device_signal_snapshot
{
    hsa_signal_value_t value      = 0;
    uint64_t           start_tick = 0;
    uint64_t           end_tick   = 0;
};

// Reads the value and raw timestamps of an SDK-owned device-memory signal with streaming loads of
// its one cache line, instead of the several serialized plain loads hsa_signal_load and
// hsa_amd_profiling_get_dispatch_time make. Call device_signal_read_fence() once between
// observing a batch complete and these reads.
device_signal_snapshot
read_device_signal(hsa_signal_t signal);

// get_dispatch_time for a signal read with read_device_signal. The first call cross-checks the
// result against hsa_amd_profiling_get_dispatch_time; on a mismatch, this and every later call
// return the latter.
profiling_time
get_device_signal_dispatch_time(hsa_agent_t                   agent,
                                hsa_signal_t                  signal,
                                const device_signal_snapshot& snapshot,
                                rocprofiler_kernel_id_t       kernel_id,
                                std::optional<uint64_t>       baseline = {});

// Streaming loads are weakly ordered, so they could run ahead of the load that saw the completion.
// They are also not snooped: a line an earlier read of the same signal left behind would carry
// that read's value, which is why callers check the snapshot's value before using its timestamps.
void
device_signal_read_fence();
}  // namespace kernel_dispatch
}  // namespace rocprofiler
