// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/agent.h>

#include <cstdint>

namespace rocprofiler
{
namespace counters
{
unsigned long
get_profiler_ioctl_request_for_version(uint32_t major_version, uint32_t minor_version);

// Returns the KFD profiler interface version reported by the running kernel (via the
// KFD_IOC_PROFILER_VERSION op), or 0 if that op is not available (e.g. no /dev/kfd, or a
// kernel predating the profiler ioctl entirely).
uint32_t
kfd_profiler_version();

// Whether the given KFD profiler interface version supports dispatch mode (see
// KFD_IOC_PROFILER_VERSION_NUM history in kfd_ioctl.h). Exposed separately from
// perfmon_clock_supported() so the version-comparison logic can be unit tested without a real
// /dev/kfd.
bool
perfmon_clock_supported_for_version(uint32_t version);

// Whether the running KFD supports dispatch mode, i.e. perfmon_clock_supported_for_version()
// on the version reported by kfd_profiler_version(). Used internally by
// counter_collection_dispatch_perfmon_enable() to gate the dispatch-mode ioctl below.
bool
perfmon_clock_supported();

bool
counter_collection_has_device_lock();

// Device counting isn't scoped to one dispatch, so another process's concurrent kernels would
// pollute the values without exclusivity. Requests the exclusive profiler_process lock
// (lock=1), which KFD gates on CAP_PERFMON regardless of KFD profiler interface version.
rocprofiler_status_t
counter_collection_device_lock(const rocprofiler_agent_t* agent, bool all_queues);

rocprofiler_status_t
counter_collection_device_unlock(const rocprofiler_agent_t* agent);

// Dispatch counting already serializes hardware execution to one dispatch at a time, so --
// unlike device counting above -- it needs no device lock to isolate its counters; there is
// really no "lock" involved here at all, which is why this is a separate call rather than a
// mode of counter_collection_device_lock(). Starting at KFD_IOC_PROFILER_VERSION_NUM 2 (see
// perfmon_clock_supported()) KFD supports a non-exclusive KFD_IOC_PROFILER_PMC request
// (lock=0, perfcount_enable=1) that just forces the perfmon clock on, without requiring
// CAP_PERFMON; a no-op success on older Linux kernels that lack this. KFD only clears the
// perfmon clock request when this process's KFD device context is torn down (e.g. process
// exit) -- there is no unprivileged "off" ioctl, so no matching disable call exists.
rocprofiler_status_t
counter_collection_dispatch_perfmon_enable(const rocprofiler_agent_t* agent);

bool
ptl_control_supported();

bool
use_device_lock_at_start();

rocprofiler_status_t
counter_collection_ptl_disable(const rocprofiler_agent_t* agent);

rocprofiler_status_t
counter_collection_ptl_enable(const rocprofiler_agent_t* agent);

}  // namespace counters
}  // namespace rocprofiler
