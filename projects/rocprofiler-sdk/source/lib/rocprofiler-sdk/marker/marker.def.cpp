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

#include "lib/rocprofiler-sdk/marker/defines.hpp"
#include "lib/rocprofiler-sdk/marker/marker.hpp"

namespace rocprofiler
{
namespace marker
{
template <>
struct roctx_domain_info<ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi>
{
    static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_MARKER_API;
    static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_MARKER_API;
    using args_type                           = rocprofiler_marker_api_args_t;
    using retval_type                         = rocprofiler_marker_api_retval_t;
    using callback_data_type                  = rocprofiler_callback_tracing_marker_api_data_t;
};
}  // namespace marker
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_MARKER_MARKER_CPP_IMPL) &&                                 \
    ROCPROFILER_LIB_ROCPROFILER_MARKER_MARKER_CPP_IMPL == 1

// clang-format off
MARKER_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, roctx_api_table_t)

MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxMarkA, roctxMarkA, roctxMarkA_fn, message)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxRangePushA, roctxRangePushA, roctxRangePushA_fn, message)
MARKER_API_INFO_DEFINITION_0(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxRangePop, roctxRangePop, roctxRangePop_fn)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxRangeStartA, roctxRangeStartA, roctxRangeStartA_fn, message)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxRangeStop, roctxRangeStop, roctxRangeStop_fn, id)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxProfilerPause, roctxProfilerPause, roctxProfilerPause_fn, tid)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxProfilerResume, roctxProfilerResume, roctxProfilerResume_fn, tid)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxNameOsThread, roctxNameOsThread, roctxNameOsThread_fn, name)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxNameHsaAgent, roctxNameHsaAgent, roctxNameHsaAgent_fn, name, agent)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxNameHipDevice, roctxNameHipDevice, roctxNameHipDevice_fn, name, device_id)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxNameHipStream, roctxNameHipStream, roctxNameHipStream_fn, name, stream)
MARKER_API_INFO_DEFINITION_V(ROCPROFILER_MARKER_API_TABLE_ID_RoctxApi, ROCPROFILER_MARKER_API_ID_roctxGetThreadId, roctxGetThreadId, roctxGetThreadId_fn, tid)
// clang-format on

#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/marker/marker.cpp"
#endif
