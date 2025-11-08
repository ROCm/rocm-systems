// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/counters/ioctl.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"
#include "rocprofiler-sdk/fwd.h"

#include <fmt/core.h>
#include <sys/ioctl.h>
#include <cerrno>
#include <cstring>

#include "ptl.hpp"

namespace rocprofiler
{
// For some GPUs, PTL needs to be disabled to collect some counters.
namespace ptl
{
namespace
{
namespace pc_sampling = rocprofiler::pc_sampling;

template <typename T>
std::pair<int, int>
check_ioctl(int fd, unsigned long request, T* arg)
{
    int ret = ::ioctl(fd, request, static_cast<void*>(arg));
    if(ret < 0)
    {
        return {ret, errno};
    }

    return {ret, 0};
}

int
enable_ptl(int gpu_id, int format1, int format2)
{
    ROCP_INFO << fmt::format(
        "PTL: Enabling PTL for GPU {:5} (format1={}, format2={})", gpu_id, format1, format2);

    kfd_ioctl_profiler_args args{};
    args.op = KFD_IOC_PROFILER_PTL;

    kfd_ioctl_ptl_settings ptl{};
    ptl.cmd          = KFD_IOCTL_PTL_OP_START;
    ptl.gpu_id       = gpu_id;
    ptl.ptl_state    = 1;
    ptl.pref_format1 = format1;
    ptl.pref_format2 = format2;

    args.ptl = ptl;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PROFILER, &args);

    ROCP_WARNING_IF(err != 0) << fmt::format(
        "PTL: Could not enable PTL for GPU {:5}: errno={} ({})\n", gpu_id, err, std::strerror(err));

    ROCP_INFO_IF(err == 0) << fmt::format(
        "PTL: PTL state set for GPU {:5} with format1={}, format2={}\n", gpu_id, format1, format2);

    return err;
}

int
disable_ptl(int gpu_id)
{
    ROCP_INFO << fmt::format("PTL: Disabling PTL for GPU {:5}", gpu_id);

    kfd_ioctl_profiler_args args{};
    args.op = KFD_IOC_PROFILER_PTL;

    kfd_ioctl_ptl_settings ptl{};
    ptl.cmd    = KFD_IOCTL_PTL_OP_STOP;
    ptl.gpu_id = gpu_id;
    args.ptl   = ptl;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PROFILER, &args);

    ROCP_WARNING_IF(err != 0) << fmt::format(
        "PTL: Disable PTL failed on GPU {:5}. Counter collection may "
        "be inaccurate or degraded: errno={} ({}): {}\n",
        gpu_id,
        err,
        std::strerror(err));

    return err;
}

constexpr uint32_t GPU_PCIE_DID = 0x74a2;

bool
enable_ptl(const rocprofiler_agent_t* agent, uint32_t format1, uint32_t format2)
{
    if(agent->device_id == GPU_PCIE_DID)
    {
        return enable_ptl(agent->gpu_id, format1, format2) == 0;
    }
    else
    {
        ROCP_INFO << "PTL: Ignoring PTL enable request for unsupported device";
        return false;
    }
}

bool
disable_ptl(const rocprofiler_agent_t* agent)
{
    if(agent->device_id == GPU_PCIE_DID)
    {
        return disable_ptl(agent->gpu_id) == 0;
    }
    else
    {
        ROCP_INFO << "PTL: Ignoring PTL disable request for unsupported device";
        return false;
    }
}

agent_ptl_state
get_agent_ptl_state(const rocprofiler_agent_t* rocp_agent)
{
    const auto& agent  = *rocp_agent;
    const auto  gpu_id = agent.gpu_id;

    kfd_ioctl_profiler_args args{};
    args.op  = KFD_IOC_PROFILER_PTL;
    args.ptl = {};

    auto& ptl  = args.ptl;
    ptl.cmd    = KFD_IOCTL_PTL_OP_QUERY_CAPABILITIES;
    ptl.gpu_id = gpu_id;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PROFILER, &args);

    ROCP_INFO << fmt::format(
        "PTL: Querying PTL state for GPU {:5} (node_id: {})", agent.gpu_id, agent.node_id);

    ROCP_WARNING_IF(ret != 0) << fmt::format(
        "PTL: Failed to query PTL state for GPU {:5}: errno={} ({})",
        gpu_id,
        err,
        std::strerror(err));

    return {
        .agent        = rocp_agent,
        .ptl_enabled  = ptl.ptl_state == 1,
        .pref_format1 = ptl.pref_format1,
        .pref_format2 = ptl.pref_format2,
    };
}

const std::vector<ptl::agent_ptl_state>*
get_ptl_cache()
{
    static auto*& _v = common::static_object<std::vector<ptl::agent_ptl_state>>::construct([]() {
        std::vector<ptl::agent_ptl_state> prev_states{};
        for(const auto& agent : agent::get_agents())
        {
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                prev_states.emplace_back(ptl::get_agent_ptl_state(agent));
            }
        }
        return prev_states;
    }());
    return _v;
}
}  // namespace

void
disable_ptl()
{
    for(const auto& ptl_agent : *get_ptl_cache())
    {
        if(ptl_agent.ptl_enabled) disable_ptl(ptl_agent.agent);
    }
}

void
restore_ptl()
{
    for(const auto& ptl_agent : *get_ptl_cache())
    {
        if(ptl_agent.ptl_enabled)
            enable_ptl(ptl_agent.agent, ptl_agent.pref_format1, ptl_agent.pref_format2);
    }
}

}  // namespace ptl
}  // namespace rocprofiler
