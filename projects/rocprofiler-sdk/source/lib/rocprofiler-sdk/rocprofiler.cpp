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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/utility.hpp"

namespace rocprofiler
{
namespace
{
#define ROCPROFILER_STATUS_STRING(CODE, MSG)                                                       \
    template <>                                                                                    \
    struct status_string<CODE>                                                                     \
    {                                                                                              \
        static constexpr auto name  = #CODE;                                                       \
        static constexpr auto value = MSG;                                                         \
    };

template <size_t Idx>
struct status_string;

ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_SUCCESS, "Success")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR, "General error")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND, "Context ID not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND, "Buffer ID not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND, "Kind ID not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND, "Operation ID not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND, "Thread ID not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR, "General context error")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND, "Agent ID not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND, "HW counter not found")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID,
                          "Context configuration is not valid")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED, "Context failed to start")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT,
                          "Context has a conflict with another context")
ROCPROFILER_STATUS_STRING(
    ROCPROFILER_STATUS_ERROR_BUFFER_BUSY,
    "Buffer operation failed because it is currently busy handling another request")
ROCPROFILER_STATUS_STRING(
    ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED,
    "Service configuration request would overwrite existing service configuration values")
ROCPROFILER_STATUS_STRING(
    ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED,
    "Configuration request occurred outside of valid rocprofiler configuration period")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED,
                          "API function is defined but not implemented")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI,
                          "Data structure provided by user has a incompatible binary interface "
                          "with this version of rocprofiler")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT,
                          "Function invoked with one or more invalid arguments")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT,
                          "Metric is not valid for the agent")

template <size_t Idx, size_t... Tail>
const char*
get_status_name(rocprofiler_status_t status, std::index_sequence<Idx, Tail...>)
{
    if(status == Idx) return status_string<Idx>::name;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_status_name(status, std::index_sequence<Tail...>{});
    return nullptr;
}

template <size_t Idx, size_t... Tail>
const char*
get_status_string(rocprofiler_status_t status, std::index_sequence<Idx, Tail...>)
{
    if(status == Idx) return status_string<Idx>::value;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_status_string(status, std::index_sequence<Tail...>{});
    return nullptr;
}
}  // namespace
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    if(major) *major = ROCPROFILER_VERSION_MAJOR;
    if(minor) *minor = ROCPROFILER_VERSION_MINOR;
    if(patch) *patch = ROCPROFILER_VERSION_PATCH;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_get_timestamp(rocprofiler_timestamp_t* ts)
{
    *ts = rocprofiler::common::timestamp_ns();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_get_thread_id(rocprofiler_thread_id_t* tid)
{
    *tid = rocprofiler::common::get_tid();
    return ROCPROFILER_STATUS_SUCCESS;
}

const char*
rocprofiler_get_status_name(rocprofiler_status_t status)
{
    return rocprofiler::get_status_name(status,
                                        std::make_index_sequence<ROCPROFILER_STATUS_LAST>{});
}

const char*
rocprofiler_get_status_string(rocprofiler_status_t status)
{
    return rocprofiler::get_status_string(status,
                                          std::make_index_sequence<ROCPROFILER_STATUS_LAST>{});
}
}
