// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

namespace rocprofiler
{
namespace tracing
{
hsa_amd_profiling_dispatch_time_t&
operator+=(hsa_amd_profiling_dispatch_time_t& lhs, uint64_t rhs)
{
    lhs.start += rhs;
    lhs.end += rhs;
    return lhs;
}

hsa_amd_profiling_dispatch_time_t&
operator-=(hsa_amd_profiling_dispatch_time_t& lhs, uint64_t rhs)
{
    lhs.start -= rhs;
    lhs.end -= rhs;
    return lhs;
}

hsa_amd_profiling_dispatch_time_t&
operator*=(hsa_amd_profiling_dispatch_time_t& lhs, uint64_t rhs)
{
    lhs.start *= rhs;
    lhs.end *= rhs;
    return lhs;
}

hsa_amd_profiling_async_copy_time_t&
operator+=(hsa_amd_profiling_async_copy_time_t& lhs, uint64_t rhs)
{
    lhs.start += rhs;
    lhs.end += rhs;
    return lhs;
}

hsa_amd_profiling_async_copy_time_t&
operator-=(hsa_amd_profiling_async_copy_time_t& lhs, uint64_t rhs)
{
    lhs.start -= rhs;
    lhs.end -= rhs;
    return lhs;
}

hsa_amd_profiling_async_copy_time_t&
operator*=(hsa_amd_profiling_async_copy_time_t& lhs, uint64_t rhs)
{
    lhs.start *= rhs;
    lhs.end *= rhs;
    return lhs;
}

profiling_time&
profiling_time::operator+=(uint64_t offset)
{
    start += offset;
    end += offset;
    return *this;
}

profiling_time&
profiling_time::operator-=(uint64_t offset)
{
    start -= offset;
    end -= offset;
    return *this;
}

profiling_time&
profiling_time::operator*=(uint64_t scale)
{
    start *= scale;
    end *= scale;
    return *this;
}
profiling_time
get_signal_profiling_time(
    hsa_agent_t             _hsa_agent,
    hsa_signal_t            _signal,
    rocprofiler_kernel_id_t _kernel_id,
    std::optional<uint64_t> _baseline)  // NOLINT(performance-unnecessary-value-param)
{
    auto ts                   = common::timestamp_ns();
    auto dispatch_time        = hsa_amd_profiling_dispatch_time_t{};
    auto dispatch_time_status = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
        _hsa_agent, _signal, &dispatch_time);

    auto _profile_time =
        profiling_time{dispatch_time_status, dispatch_time.start, dispatch_time.end};

    if(_profile_time.status == HSA_STATUS_SUCCESS)
    {
        _profile_time = adjust_profiling_time(
            "dispatch",
            "hsa_amd_profiling_get_dispatch_time",
            _profile_time,
            profiling_time{HSA_STATUS_SUCCESS, _baseline.value_or(dispatch_time.start), ts});
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
}  // namespace tracing
}  // namespace rocprofiler
