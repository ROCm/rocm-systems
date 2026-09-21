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
#include "lib/common/environment.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"

#include <sys/ioctl.h>
#include <cerrno>

namespace rocprofiler
{
namespace counters
{
namespace
{
constexpr uint32_t mainline_profiler_ioctl_nr = 0x28;

unsigned long
get_mainline_profiler_ioctl_request()
{
    return AMDKFD_IOWR(mainline_profiler_ioctl_nr, struct kfd_ioctl_profiler_args);
}

unsigned long
get_profiler_ioctl_request()
{
    static auto request = []() {
        kfd_ioctl_get_version_args args = {};
        if(ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_GET_VERSION, &args) != 0)
        {
            auto err      = errno;
            auto fallback = static_cast<unsigned long>(AMDKFD_IOC_PROFILER);
            ROCP_INFO << fmt::format("KFD version query failed (error: {}); using fallback "
                                     "profiler ioctl request {:#x}.",
                                     strerror(err),
                                     fallback);
            return fallback;
        }

        auto selected =
            get_profiler_ioctl_request_for_version(args.major_version, args.minor_version);
        ROCP_INFO << fmt::format("KFD version {}.{} detected; using profiler ioctl request {:#x}.",
                                 args.major_version,
                                 args.minor_version,
                                 selected);
        return selected;
    }();

    return request;
}
}  // namespace

unsigned long
get_profiler_ioctl_request_for_version(uint32_t major_version, uint32_t minor_version)
{
    if(major_version > 1 || (major_version == 1 && minor_version >= 23))
    {
        return get_mainline_profiler_ioctl_request();
    }

    return AMDKFD_IOC_PROFILER;
}

bool
counter_collection_has_device_lock()
{
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_VERSION;
    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), get_profiler_ioctl_request(), &args);
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

    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), get_profiler_ioctl_request(), &args);
    if(ret != 0)
    {
        auto err = errno;
        switch(err)
        {
            case EBUSY:
                ROCP_WARNING << fmt::format(
                    "Device {} has a profiler attached to it. PMC Counters may be inaccurate.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
            case EPERM:
                ROCP_WARNING << fmt::format(
                    "Device {} could not be locked for profiling due to lack of permissions "
                    "(capability SYS_PERFMON). PMC Counters may be inaccurate and System Counter "
                    "Collection will be degraded.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED;
            case EINVAL:
                ROCP_WARNING << fmt::format(
                    "Driver/Kernel version does not support locking device {}. PMC Counters may be "
                    "inaccurate and System Counter Collection will be degraded.",
                    agent->id.handle);
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
            default:
                ROCP_WARNING << fmt::format(
                    "Failed to lock device {} (error: {}). PMC Counters may be "
                    "inaccurate and System Counter "
                    "Collection will be degraded.",
                    agent->id.handle,
                    strerror(err));
                return ROCPROFILER_STATUS_ERROR;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counter_collection_device_unlock(const rocprofiler_agent_t* agent)
{
    CHECK(agent);
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_PMC;
    args.pmc.gpu_id              = agent->gpu_id;
    args.pmc.lock                = 0;
    args.pmc.perfcount_enable    = 0;

    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), get_profiler_ioctl_request(), &args);
    if(ret != 0)
    {
        auto err = errno;
        ROCP_WARNING << fmt::format("Failed to unlock device {} (error: {}). Continuing anyway.",
                                    agent->id.handle,
                                    strerror(err));
        return ROCPROFILER_STATUS_ERROR;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

bool
ptl_control_supported()
{
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_VERSION;
    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), get_profiler_ioctl_request(), &args);
    return (ret == 0);
}

bool
use_device_lock_at_start()
{
    static bool value = rocprofiler::common::get_env("ROCPROFILER_DEVICE_LOCK_AT_START", false);
    return value;
}

rocprofiler_status_t
counter_collection_ptl_disable(const rocprofiler_agent_t* agent)
{
    CHECK(agent);
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_PTL_CONTROL;
    args.ptl.gpu_id              = agent->gpu_id;
    args.ptl.enable              = 0;

    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), get_profiler_ioctl_request(), &args);
    if(ret != 0)
    {
        auto err = errno;
        ROCP_INFO << fmt::format(
            "Failed to disable PTL on device {} (error: {}). Continuing anyway.",
            agent->id.handle,
            strerror(err));
        return ROCPROFILER_STATUS_ERROR;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counter_collection_ptl_enable(const rocprofiler_agent_t* agent)
{
    CHECK(agent);
    kfd_ioctl_profiler_args args = {};
    args.op                      = KFD_IOC_PROFILER_PTL_CONTROL;
    args.ptl.gpu_id              = agent->gpu_id;
    args.ptl.enable              = 1;

    int ret = ioctl(pc_sampling::ioctl::get_kfd_fd(), get_profiler_ioctl_request(), &args);
    if(ret != 0)
    {
        auto err = errno;
        ROCP_INFO << fmt::format(
            "Failed to enable PTL on device {} (error: {}). Continuing anyway.",
            agent->id.handle,
            strerror(err));
        return ROCPROFILER_STATUS_ERROR;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace counters
}  // namespace rocprofiler
