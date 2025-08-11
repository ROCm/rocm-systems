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

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <fmt/format.h>

#include <cstdint>

namespace rocprofiler
{
namespace tracing
{
hsa_amd_profiling_dispatch_time_t&
operator+=(hsa_amd_profiling_dispatch_time_t& lhs, uint64_t rhs);

hsa_amd_profiling_dispatch_time_t&
operator-=(hsa_amd_profiling_dispatch_time_t& lhs, uint64_t rhs);

hsa_amd_profiling_dispatch_time_t&
operator*=(hsa_amd_profiling_dispatch_time_t& lhs, uint64_t rhs);

hsa_amd_profiling_async_copy_time_t&
operator+=(hsa_amd_profiling_async_copy_time_t& lhs, uint64_t rhs);

hsa_amd_profiling_async_copy_time_t&
operator-=(hsa_amd_profiling_async_copy_time_t& lhs, uint64_t rhs);

hsa_amd_profiling_async_copy_time_t&
operator*=(hsa_amd_profiling_async_copy_time_t& lhs, uint64_t rhs);

#if !defined(ROCPROFILER_CI_STRICT_TIMESTAMPS)
#    define ROCPROFILER_CI_STRICT_TIMESTAMPS 0
#endif

struct profiling_time
{
    hsa_status_t status = HSA_STATUS_ERROR_INVALID_SIGNAL;
    uint64_t     start  = 0;
    uint64_t     end    = 0;

    profiling_time& operator+=(uint64_t offset);
    profiling_time& operator-=(uint64_t offset);
    profiling_time& operator*=(uint64_t scale);
};

inline profiling_time
adjust_profiling_time(std::string_view _label,
                      std::string_view _responsible,
                      profiling_time   _value,
                      profiling_time&& _bounds)
{
    static auto sysclock_period = hsa::get_hsa_timestamp_period();
    static auto normalize_env   = common::get_env("ROCPROFILER_CI_FREQ_SCALE_TIMESTAMPS", false);
    static auto strict_ts_env   = common::get_env(
        "ROCPROFILER_CI_STRICT_TIMESTAMPS", (ROCPROFILER_CI_STRICT_TIMESTAMPS > 0) ? true : false);

    // normalize
    if(ROCPROFILER_UNLIKELY(normalize_env)) _value *= sysclock_period;

    if(strict_ts_env)
    {
        ROCP_FATAL_IF(ROCPROFILER_UNLIKELY(_value.start > _value.end))
            << fmt::format("{} returned invalid {} time value: {} start time is greater than the "
                           "{} end time ({} > {}) :: difference={}",
                           _responsible,
                           _label,
                           _label,
                           _label,
                           _value.start,
                           _value.end,
                           (_value.start - _value.end));

        ROCP_FATAL_IF(ROCPROFILER_UNLIKELY(_value.start < _bounds.start))
            << fmt::format("{} returned invalid {} time value: {} start time is before the API "
                           "call enqueuing the operation on the CPU ({} < {}) :: difference={}",
                           _responsible,
                           _label,
                           _label,
                           _value.start,
                           _bounds.start,
                           (_bounds.start - _value.start));

        ROCP_FATAL_IF(ROCPROFILER_UNLIKELY(_value.end > _bounds.end))
            << fmt::format("{} returned invalid {} time value: {} end time is greater than the "
                           "current time on the CPU ({} > {}) :: difference={}",
                           _responsible,
                           _label,
                           _label,
                           _value.end,
                           _bounds.end,
                           (_value.end - _bounds.end));
    }

    if(_value.start > _value.end)
    {
        ROCP_ERROR << fmt::format(
            "{} returned {} times where the start time is after end time ({} > {}) :: "
            "difference={}. Swapping the values. Set the environment variable "
            "ROCPROFILER_CI_STRICT_TIMESTAMPS=1 to cause a failure instead",
            _responsible,
            _label,
            _value.start,
            _value.end,
            (_value.start - _value.end));

        std::swap(_value.start, _value.end);
    }

    // below are hacks for clock skew issues:
    //
    // the timestamp of this handler will always be after when the profiling time ended
    if(_bounds.end < _value.end) _value -= (_value.end - _bounds.end);

    // the timestamp of the enqueue will always be before when the profiling time started
    if(_value.start < _bounds.start) _value += (_bounds.start - _value.start);

    return _value;
}
}  // namespace tracing
}  // namespace rocprofiler
