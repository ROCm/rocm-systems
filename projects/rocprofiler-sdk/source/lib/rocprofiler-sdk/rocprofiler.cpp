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

#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

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
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_CONTEXT_ID_NOT_ZERO,
                          "Context ID should be initialized to zero")
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
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_FINALIZED,
                          "Invalid request because rocprofiler has finalized")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED,
                          "Function call requires that HSA is loaded")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND,
                          "Dimension is not found for counter")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND,
                          "Profile could not find counter for GPU")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED,
                          "AST could not be generated correctly")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND, "AST was not found")
ROCPROFILER_STATUS_STRING(
    ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD,
    "AQL Profiler was not able to find event coordinates for defined counters")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL,
                          "A service depends on a newer version of KFD (amdgpu kernel driver)")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES,
                          "The given resources are insufficient to complete operation")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND,
                          "Could not find counter profile")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_AGENT_DISPATCH_CONFLICT,
                          "Cannot have both an agent counter collection and a dispatch counter "
                          "in the same context")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_INTERNAL_NO_AGENT_CONTEXT,
                          "No context has agent profiling enabled, "
                          "error generally not returned to tools")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_SAMPLE_RATE_EXCEEDED,
                          "A sample is in progress and a new sample cannot be started")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE,
                          "No profile queue is available for this agent")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS,
                          "Counter set does not include any hardware counters")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH,
                          "Counter profile agent does not match the agent in the context")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE,
                          "The service is not available. Please refer to API functions that return "
                          "this status code for more information.")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT,
                          "Request exceeds the capabilities of the hardware to collect")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED,
                          "Agent HW architecture is not supported, no counter metrics found.")
ROCPROFILER_STATUS_STRING(ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED,
                          "Required permission (CAP_PERFMON) is not set, permission denied")

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
rocprofiler_get_version_triplet(rocprofiler_version_triplet_t* info)
{
    *info = {.major = ROCPROFILER_VERSION_MAJOR,
             .minor = ROCPROFILER_VERSION_MINOR,
             .patch = ROCPROFILER_VERSION_PATCH};
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
