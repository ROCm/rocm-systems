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

#include "lib/rocprofiler-sdk/counters/ioctl.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"

#include <sys/ioctl.h>
#include <cerrno>

namespace rocprofiler
{
namespace counters
{
bool
counter_collection_has_device_lock()
{
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_VERSION;
    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PROFILER, &args);
    if(ret == 0)
    {
        return true;
    }
    return false;
}

rocprofiler_status_t
counter_collection_device_lock(const rocprofiler_agent_t* agent, bool all_queues)
{
    CHECK(agent);
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_PMC;
    args.pmc.gpu_id              = agent->gpu_id;
    args.pmc.lock                = 1;
    args.pmc.perfcount_enable    = all_queues ? 1 : 0;

    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PROFILER, &args);
    if(ret != 0)
    {
        switch(ret)
        {
            case -EBUSY:
                ROCP_WARNING << fmt::format(
                    "Device {} has a profiler attached to it. PMC Counters may be inaccurate.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
            case -EPERM:
                ROCP_WARNING << fmt::format(
                    "Device {} could not be locked for profiling due to lack of permissions "
                    "(capability SYS_PERFMON). PMC Counters may be inaccurate and System Counter "
                    "Collection will be degraded.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED;
            case -EINVAL:
                ROCP_WARNING << fmt::format(
                    "Driver/Kernel version does not support locking device {}. PMC Counters may be "
                    "inaccurate and System Counter Collection will be degraded.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
            default:
                ROCP_WARNING << fmt::format(
                    "Failed to lock device {}. PMC Counters may be inaccurate and System Counter "
                    "Collection will be degraded.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

// Not required now but may be useful in the future.
// rocprofiler_status_t
// counter_collection_device_unlock(const rocprofiler_agent_t* agent) {
//     CHECK(agent);
//     kfd_ioctl_profiler_args args = {};
//     args.op = KFD_IOC_PROFILER_PMC;
//     args.pmc.gpu_id = agent->gpu_id;
//     args.pmc.lock = 0;
//     args.pmc.perfcount_enable = 0;

//     int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PROFILER, &args);
//     if (ret != 0) {
//         switch (ret) {
//             case -EBUSY:
//             case -EPERM:
//                 ROCP_WARNING << fmt::format("Could not unlock the device {}", agent->id.handle);
//                 return ROCPROFILER_STATUS_ERROR;
//             case -EINVAL:
//                 return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
//             default:
//                 ROCP_WARNING << fmt::format("Could not unlock the device {}", agent->id.handle);
//                 return ROCPROFILER_STATUS_ERROR;
//         }
//     }

//     return ROCPROFILER_STATUS_SUCCESS;
// }
}  // namespace counters
}  // namespace rocprofiler
