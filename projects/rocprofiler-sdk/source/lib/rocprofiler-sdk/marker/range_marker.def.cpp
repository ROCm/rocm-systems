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

#include "lib/rocprofiler-sdk/marker/defines.hpp"
#include "lib/rocprofiler-sdk/marker/marker.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/marker/table_id.h>

namespace rocprofiler
{
namespace marker
{
template <>
struct roctx_domain_info<ROCPROFILER_MARKER_TABLE_ID_LAST>
{
    using args_type          = rocprofiler_marker_api_args_t;
    using retval_type        = rocprofiler_marker_api_retval_t;
    using callback_data_type = rocprofiler_callback_tracing_marker_api_data_t;
    using buffer_data_type   = rocprofiler_buffer_tracing_marker_api_record_t;
};

template <>
struct roctx_domain_info<ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange>
: roctx_domain_info<ROCPROFILER_MARKER_TABLE_ID_LAST>
{
    using enum_type                           = rocprofiler_marker_core_range_api_id_t;
    static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API;
    static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API;
    static constexpr auto none                = ROCPROFILER_MARKER_CORE_RANGE_API_ID_NONE;
    static constexpr auto last                = ROCPROFILER_MARKER_CORE_RANGE_API_ID_LAST;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_RANGE_API;
};
}  // namespace marker
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_MARKER_RANGE_MARKER_CPP_IMPL) &&                       \
    ROCPROFILER_LIB_ROCPROFILER_SDK_MARKER_RANGE_MARKER_CPP_IMPL == 1

// clang-format off
MARKER_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange, roctx_core_api_table_t)

MARKER_EVENT_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange, ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA, roctxMarkA, roctxMarkA_fn, message)
MARKER_RANGE_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange, ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA, roctxThreadRangeA, roctxRangePushA_fn, roctxRangePop_fn, message)
MARKER_RANGE_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange, ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA, roctxProcessRangeA, roctxRangeStartA_fn, roctxRangeStop_fn, message)
MARKER_EVENT_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange, ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxGetThreadId, roctxGetThreadId, roctxGetThreadId_fn, tid)
// clang-format on

#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/marker/range_marker.cpp"
#endif
