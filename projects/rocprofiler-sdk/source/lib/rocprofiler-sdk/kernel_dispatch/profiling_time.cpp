// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <string_view>

namespace rocprofiler
{
namespace kernel_dispatch
{
namespace
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
}  // namespace

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
get_dispatch_time(hsa_agent_t             _hsa_agent,
                  hsa_signal_t            _signal,
                  rocprofiler_kernel_id_t _kernel_id,
                  std::optional<uint64_t> _baseline)
{
    static auto sysclock_period = hsa::get_hsa_timestamp_period();

    auto ts                   = common::timestamp_ns();
    auto dispatch_time        = hsa_amd_profiling_dispatch_time_t{};
    auto dispatch_time_status = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
        _hsa_agent, _signal, &dispatch_time);

    if(dispatch_time_status == HSA_STATUS_SUCCESS)
    {
        // if we encounter this in CI, it will cause test to fail
        ROCP_CI_LOG_IF(ERROR, dispatch_time.end < dispatch_time.start)
            << "hsa_amd_profiling_get_dispatch_time for kernel_id=" << _kernel_id
            << " on rocprofiler_agent="
            << CHECK_NOTNULL(agent::get_rocprofiler_agent(_hsa_agent))->id.handle
            << " returned dispatch times where the end time (" << dispatch_time.end
            << ") was less than the start time (" << dispatch_time.start << ")";

        // normalize
        dispatch_time *= sysclock_period;

        // below is a hack for clock skew issues:
        // the timestamp of this handler for the kernel dispatch will always be after when the
        // kernel completed
        if(ts < dispatch_time.end) dispatch_time -= (dispatch_time.end - ts);

        // below is a hack for clock skew issues:
        // the timestamp of the packet rewriter for the kernel packet will always be before when the
        // kernel started
        if(_baseline && dispatch_time.start < *_baseline)
            dispatch_time += (*_baseline - dispatch_time.start);
    }
    else
    {
        ROCP_CI_LOG(ERROR) << "hsa_amd_profiling_get_dispatch_time for kernel id=" << _kernel_id
                           << " on rocprofiler_agent="
                           << CHECK_NOTNULL(agent::get_rocprofiler_agent(_hsa_agent))->id.handle
                           << " returned status=" << dispatch_time_status
                           << " :: " << hsa::get_hsa_status_string(dispatch_time_status);
    }

    return profiling_time{
        .status = dispatch_time_status, .start = dispatch_time.start, .end = dispatch_time.end};
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
