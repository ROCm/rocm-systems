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

#pragma once

#include "config.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/output/domain_type.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/output_stream.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <amd_comgr/amd_comgr.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <glog/logging.h>

#include <cxxabi.h>
#include <semaphore.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            ROCP_FATAL << " :: [" << __FILE__ << ":" << __LINE__ << "]\n\t" << #result << "\n\n"   \
                       << msg << " failed with error code "                                        \
                       << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) << ": " << status_msg;       \
        }                                                                                          \
    }

constexpr size_t BUFFER_SIZE_BYTES = 4096;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 2);

using marker_message_map_t             = std::unordered_map<uint64_t, std::string>;
using tool_counter_info                = ::rocprofiler::tool::tool_counter_info;
using kernel_symbol_info               = ::rocprofiler::tool::kernel_symbol_info;
using host_function_info               = ::rocprofiler::tool::host_function_info;
using rocprofiler_kernel_symbol_info_t = ::rocprofiler::tool::rocprofiler_kernel_symbol_info_t;
using rocprofiler_host_kernel_symbol_data_t =
    ::rocprofiler::tool::rocprofiler_host_kernel_symbol_data_t;

enum tracing_marker_kind
{
    MARKER_API_CORE = 0,
    MARKER_API_CONTROL,
    MARKER_API_NAME,
    MARKER_API_LAST,
};

template <size_t CommonV>
struct marker_tracing_kind_conversion;

#define MAP_TRACING_KIND_CONVERSION(COMMON, CALLBACK, BUFFERED)                                    \
    template <>                                                                                    \
    struct marker_tracing_kind_conversion<COMMON>                                                  \
    {                                                                                              \
        static constexpr auto callback_value = CALLBACK;                                           \
        static constexpr auto buffered_value = BUFFERED;                                           \
                                                                                                   \
        bool operator==(rocprofiler_callback_tracing_kind_t val) const                             \
        {                                                                                          \
            return (callback_value == val);                                                        \
        }                                                                                          \
                                                                                                   \
        bool operator==(rocprofiler_buffer_tracing_kind_t val) const                               \
        {                                                                                          \
            return (buffered_value == val);                                                        \
        }                                                                                          \
                                                                                                   \
        auto convert(rocprofiler_callback_tracing_kind_t) const { return buffered_value; }         \
        auto convert(rocprofiler_buffer_tracing_kind_t) const { return callback_value; }           \
    };

MAP_TRACING_KIND_CONVERSION(MARKER_API_CORE,
                            ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API,
                            ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API)
MAP_TRACING_KIND_CONVERSION(MARKER_API_CONTROL,
                            ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                            ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API)
MAP_TRACING_KIND_CONVERSION(MARKER_API_NAME,
                            ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
                            ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API)
MAP_TRACING_KIND_CONVERSION(MARKER_API_LAST,
                            ROCPROFILER_CALLBACK_TRACING_LAST,
                            ROCPROFILER_BUFFER_TRACING_LAST)

template <typename TracingKindT, size_t Idx, size_t... Tail>
auto
convert_marker_tracing_kind(TracingKindT val, std::index_sequence<Idx, Tail...>)
{
    if(marker_tracing_kind_conversion<Idx>{} == val)
    {
        return marker_tracing_kind_conversion<Idx>{}.convert(val);
    }

    if constexpr(sizeof...(Tail) > 0)
        return convert_marker_tracing_kind(val, std::index_sequence<Tail...>{});

    return marker_tracing_kind_conversion<MARKER_API_LAST>{}.convert(val);
}

template <typename TracingKindT>
auto
convert_marker_tracing_kind(TracingKindT val)
{
    return convert_marker_tracing_kind(val, std::make_index_sequence<MARKER_API_LAST>{});
}

// RAII wrapper for semaphore to cleanup and
// sync worker processes in finalization process
struct SemaphoreGuard
{
    sem_t*      sem = nullptr;
    std::string name;

    SemaphoreGuard(const std::string& sem_name)
    : name(sem_name)
    {}

    ~SemaphoreGuard()
    {
        if(sem != nullptr)
        {
            if(sem_close(sem) == -1)
            {
                ROCP_WARNING << fmt::format("Failed to close semaphore in");
            }

            if(sem_unlink(name.c_str()) == -1)
            {
                ROCP_WARNING << fmt::format("Failed to unlink semaphore or it is already unlinked");
            }
        }
    }

    bool open_or_create()
    {
        // Try to create new semaphore
        sem = sem_open(name.c_str(), O_CREAT | O_EXCL, 0666, 0);
        if(sem != SEM_FAILED) return true;

        // If exists, open existing
        if(errno == EEXIST)
        {
            sem = sem_open(name.c_str(), 0);
            return (sem != SEM_FAILED);
        }

        return false;
    }
};