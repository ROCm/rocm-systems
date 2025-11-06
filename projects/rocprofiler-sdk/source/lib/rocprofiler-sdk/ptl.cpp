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
get_num_sample_infos(int gpu_id)
{
    kfd_ioctl_pc_sample_args args{};
    args.op              = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
    args.gpu_id          = gpu_id;
    args.num_sample_info = 0;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);
    if(err != 0)
    {
        ROCP_WARNING << fmt::format("PTL: Querying sample infos for GPU %d failed: errno=%d (%s)",
                                    gpu_id,
                                    err,
                                    std::strerror(err));
        return -1;
    }

    return args.num_sample_info;
}

int
create_pc_sampling_session(int gpu_id, int ptl_state, int format1, int format2)
{
    kfd_pc_sample_info info{};
    info.method   = KFD_IOCTL_PCS_METHOD_HOSTTRAP;
    info.type     = KFD_IOCTL_PCS_TYPE_TIME_US;
    info.interval = 1000;
    // info.ptl_state    = 0;
    // info.pref_format1 = 0;
    // info.pref_format2 = 0;
    info.ptl_state    = ptl_state;
    info.pref_format1 = format1;
    info.pref_format2 = format2;

    kfd_ioctl_pc_sample_args args{};
    args.op              = KFD_IOCTL_PCS_OP_CREATE;
    args.gpu_id          = gpu_id;
    args.num_sample_info = 1;
    args.sample_info_ptr = (uint64_t)(&info);
    args.trace_id        = -1;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    if(err != 0)
    {
        ROCP_WARNING << fmt::format(
            "PTL: PC Sampling session create for GPU %d failed: errno=%d (%s)",
            gpu_id,
            std::strerror(err));
        return -1;
    }

    ROCP_INFO << fmt::format("Created PC Sampling session with trace ID: %d", args.trace_id);

    return args.trace_id;
}

void
destroy_pc_sampling_session(int gpu_id, int trace_id)
{
    kfd_ioctl_pc_sample_args args{};
    args.op              = KFD_IOCTL_PCS_OP_DESTROY;
    args.gpu_id          = gpu_id;
    args.num_sample_info = 1;
    args.sample_info_ptr = 0;
    args.trace_id        = trace_id;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    ROCP_WARNING_IF(err != 0) << fmt::format(
        "PTL: Could not destroy PC Sampling session ID %d: errno=%d (%s)",
        args.trace_id,
        err,
        std::strerror(err));

    ROCP_INFO << fmt::format("PTL: Destroyed PC Sampling session with trace ID: %d", args.trace_id);
}

int
enable_ptl(int gpu_id, int format1, int format2)
{
    const auto trace_id = create_pc_sampling_session(gpu_id, 0, format1, format2);

    // could not create a session
    if(trace_id == -1) return -1;

    kfd_pc_sample_info info{};
    info.ptl_state    = 1;
    info.pref_format1 = format1;
    info.pref_format2 = format2;

    kfd_ioctl_pc_sample_args args{};
    args.op              = KFD_IOCTL_PCS_OP_START_PTL;
    args.gpu_id          = gpu_id;
    args.trace_id        = trace_id;
    args.num_sample_info = 1;
    args.sample_info_ptr = (uint64_t)(&info);

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    ROCP_WARNING_IF(err != 0) << fmt::format(
        "PTL: Could not enable PTL for GPU %d: errno=%d (%s)\n", gpu_id, err, std::strerror(err));

    ROCP_INFO_IF(err == 0) << fmt::format(
        "PTL: PTL state set for GPU %d with format1=%d, format2=%d\n", gpu_id, format1, format2);

    destroy_pc_sampling_session(gpu_id, trace_id);

    return err;
}

int
disable_ptl(int gpu_id)
{
    // Maybe KFD issue. Need to set this to 1 and then disable, create with 0 does not seem to work
    const auto trace_id = create_pc_sampling_session(gpu_id, 1, 0, 0);

    // could not create a session
    if(trace_id == -1) return -1;

    kfd_ioctl_pc_sample_args args{};
    args.op              = KFD_IOCTL_PCS_OP_STOP_PTL;
    args.gpu_id          = gpu_id;
    args.trace_id        = trace_id;
    args.num_sample_info = 0;
    args.sample_info_ptr = 0;

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    ROCP_WARNING_IF(err != 0) << fmt::format(
        "PTL: Disable PTL failed on GPU %d. Counter collection may "
        "be unavailable: errno=%d (%s): %s\n",
        gpu_id,
        err,
        std::strerror(err));

    destroy_pc_sampling_session(gpu_id, trace_id);

    return err;
}

static constexpr uint32_t GPU_PCIE_DID = 0x74a2;

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
    const auto gpu_id           = rocp_agent->gpu_id;
    const auto num_sample_infos = get_num_sample_infos(gpu_id);

    std::vector<kfd_pc_sample_info> samples(num_sample_infos);

    for(int i = 0; i < num_sample_infos; ++i)
    {
        samples[i].ptl_state    = UINT32_MAX;
        samples[i].pref_format1 = UINT32_MAX;
        samples[i].pref_format2 = UINT32_MAX;
    }

    kfd_ioctl_pc_sample_args args{};
    args.op              = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
    args.gpu_id          = gpu_id;
    args.num_sample_info = num_sample_infos;
    args.sample_info_ptr = (uint64_t)(samples.data());

    const auto [ret, err] =
        check_ioctl(pc_sampling::ioctl::get_kfd_fd(), AMDKFD_IOC_PC_SAMPLE, &args);

    ROCP_WARNING_IF(ret != 0) << fmt::format(
        "PTL: Failed to query PTL state for GPU %d: errno=%d (%s)",
        gpu_id,
        err,
        std::strerror(err));

    for(int i = 0; i < num_sample_infos; ++i)
    {
        ROCP_TRACE << fmt::format("PTL:  Sample [%d]:\n", i);
        ROCP_TRACE << fmt::format("PTL:      ptl_state:    %u\n", samples[i].ptl_state);
        ROCP_TRACE << fmt::format("PTL:      pref_format1: %u\n", samples[i].pref_format1);
        ROCP_TRACE << fmt::format("PTL:      pref_format2: %u\n", samples[i].pref_format2);

        ROCP_ERROR_IF(samples[i].ptl_state != samples[0].ptl_state)
            << fmt::format("PTL: ptl_state is not the same in all reported samples from KFD");
    }

    return {
        .agent        = rocp_agent,
        .ptl_enabled  = samples[0].ptl_state == 1,
        .pref_format1 = samples[0].pref_format1,
        .perf_format2 = samples[0].pref_format2,
    };
}

void
restore_ptl_state(const agent_ptl_state& prev_state)
{
    if(prev_state.agent == nullptr)
    {
        ROCP_ERROR << "PTL: Agent cannot be null";
    }

    // enable ptl if it was enabled before we disabled it
    if(prev_state.ptl_enabled)
    {
        enable_ptl(prev_state.agent, prev_state.pref_format1, prev_state.perf_format2);
    }
}

std::vector<ptl::agent_ptl_state>*
get_ptl_cache()
{
    static auto*& _v = common::static_object<std::vector<ptl::agent_ptl_state>>::construct([]() {
        std::vector<ptl::agent_ptl_state> prev_states{};
        for(const auto& agent : agent::get_agents())
        {
            prev_states.emplace_back(ptl::get_agent_ptl_state(agent));
        }
    });
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
