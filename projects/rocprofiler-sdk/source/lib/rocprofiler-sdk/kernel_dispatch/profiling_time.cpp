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

#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <string_view>

namespace rocprofiler
{
namespace kernel_dispatch
{
profiling_time
get_dispatch_time(hsa_agent_t             _hsa_agent,
                  hsa_signal_t            _signal,
                  rocprofiler_kernel_id_t _kernel_id,
                  std::optional<uint64_t> _baseline)  // NOLINT(performance-unnecessary-value-param)
{
    auto ts                   = common::timestamp_ns();
    auto dispatch_time        = hsa_amd_profiling_dispatch_time_t{};
    auto dispatch_time_status = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
        _hsa_agent, _signal, &dispatch_time);

    auto _profile_time =
        tracing::profiling_time{dispatch_time_status, dispatch_time.start, dispatch_time.end};

    if(_profile_time.status == HSA_STATUS_SUCCESS)
    {
        _profile_time = tracing::adjust_profiling_time(
            "dispatch",
            "hsa_amd_profiling_get_dispatch_time",
            _profile_time,
            tracing::profiling_time{
                HSA_STATUS_SUCCESS, _baseline.value_or(dispatch_time.start), ts});
    }
    else
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "hsa_amd_profiling_get_dispatch_time for kernel id={} on agent-{} returned status={} "
            ":: {}",
            _kernel_id,
            CHECK_NOTNULL(agent::get_rocprofiler_agent(_hsa_agent))->node_id,
            static_cast<int>(dispatch_time_status),
            hsa::get_hsa_status_string(dispatch_time_status));
    }

    return _profile_time;
}
profiling_time
get_dispatch_time_from_ticks(hsa_agent_t             _hsa_agent,
                             uint64_t                gpu_start_tick,
                             uint64_t                gpu_end_tick,
                             rocprofiler_kernel_id_t _kernel_id,
                             std::optional<uint64_t> _baseline)
{
    auto ts = common::timestamp_ns();

    if(gpu_start_tick == 0 || gpu_end_tick == 0)
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "get_dispatch_time_from_ticks for kernel id={} on agent-{}: zero timestamps "
            "(start={}, end={})",
            _kernel_id,
            CHECK_NOTNULL(agent::get_rocprofiler_agent(_hsa_agent))->node_id,
            gpu_start_tick,
            gpu_end_tick);
        return profiling_time{HSA_STATUS_ERROR_INVALID_ARGUMENT, 0, 0};
    }

    auto* ext_table  = hsa::get_amd_ext_table();
    uint64_t sys_start = 0;
    uint64_t sys_end   = 0;

    auto s1 = ext_table->hsa_amd_profiling_convert_tick_to_system_domain_fn(
        _hsa_agent, gpu_start_tick, &sys_start);
    auto s2 = ext_table->hsa_amd_profiling_convert_tick_to_system_domain_fn(
        _hsa_agent, gpu_end_tick, &sys_end);

    if(s1 != HSA_STATUS_SUCCESS || s2 != HSA_STATUS_SUCCESS)
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "tick-to-system conversion failed for kernel id={}: start_status={}, end_status={}",
            _kernel_id,
            static_cast<int>(s1),
            static_cast<int>(s2));
        return profiling_time{(s1 != HSA_STATUS_SUCCESS) ? s1 : s2, 0, 0};
    }

    auto _profile_time = tracing::profiling_time{HSA_STATUS_SUCCESS, sys_start, sys_end};

    _profile_time = tracing::adjust_profiling_time(
        "dispatch",
        "get_dispatch_time_from_ticks",
        _profile_time,
        tracing::profiling_time{HSA_STATUS_SUCCESS, _baseline.value_or(sys_start), ts});

    return _profile_time;
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
