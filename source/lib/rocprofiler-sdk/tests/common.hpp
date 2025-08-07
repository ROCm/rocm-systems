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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/defines.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

#define ROCPROFILER_CALL_EXPECT(ARG, MSG, STATUS)                                                  \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, STATUS) << MSG << " (" << _status << "="                                \
                                   << rocprofiler_get_status_string(_status) << ") :: " << #ARG;   \
    }

namespace
{
struct callback_data
{
    rocprofiler_client_id_t*      client_id             = nullptr;
    rocprofiler_client_finalize_t client_fini_func      = nullptr;
    rocprofiler_context_id_t      client_ctx            = {0};
    rocprofiler_buffer_id_t       client_buffer         = {};
    rocprofiler_callback_thread_t client_thread         = {};
    uint64_t                      client_workflow_count = {};
    uint64_t                      client_callback_count = {};
    uint64_t                      client_elapsed        = {};
    int64_t                       current_depth         = 0;
    int64_t                       max_depth             = 0;
};

struct callback_data_ext
{
    using callback_count_data_t = std::pair<uint64_t, uint64_t>;
    using callback_count_map_t  = std::map<std::string_view, callback_count_data_t>;

    rocprofiler_client_id_t*      client_id             = nullptr;
    rocprofiler_client_finalize_t client_fini_func      = nullptr;
    rocprofiler_context_id_t      client_hsa_ctx        = {0};
    rocprofiler_context_id_t      client_hip_ctx        = {};
    rocprofiler_buffer_id_t       client_buffer         = {};
    rocprofiler_callback_thread_t client_thread         = {};
    uint64_t                      client_workflow_count = {};
    uint64_t                      client_elapsed        = {};
    int64_t                       current_depth         = 0;
    int64_t                       max_depth             = 0;
    callback_count_map_t          client_callback_count = {};
};

struct agent_data
{
    uint64_t                       agent_count = 0;
    std::vector<hsa_device_type_t> agents      = {};
};

using callback_kind_names_t = std::map<rocprofiler_callback_tracing_kind_t, std::string_view>;
using callback_kind_operation_names_t =
    std::map<rocprofiler_callback_tracing_kind_t, std::map<uint32_t, std::string_view>>;

struct callback_name_info
{
    callback_kind_names_t           kind_names      = {};
    callback_kind_operation_names_t operation_names = {};
};

inline auto
get_callback_tracing_names()
{
    static const auto supported_kinds = std::unordered_set<rocprofiler_callback_tracing_kind_t>{
        ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
        ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
        ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
        ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
        ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
        ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
        ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
        ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
    };

    auto cb_name_info = callback_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_callback_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t     operation,
                                               void*                               data_v) {
        auto* name_info_v = static_cast<callback_name_info*>(data_v);

        if(supported_kinds.count(kindv) > 0)
        {
            const char* name = nullptr;
            ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                 kindv, operation, &name, nullptr),
                             "query callback tracing kind operation name");
            EXPECT_TRUE(name != nullptr) << "kind=" << kindv << ", operation=" << operation;
            if(name) name_info_v->operation_names[kindv][operation] = name;
        }
        return 0;
    };

    //
    //  callback for each callback kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<callback_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query callback tracing kind operation name");
        EXPECT_TRUE(name != nullptr) << "kind=" << kind;
        if(name) name_info_v->kind_names[kind] = name;

        if(supported_kinds.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating callback tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb,
                                                                static_cast<void*>(&cb_name_info)),
                     "iterating callback tracing kinds");

    return cb_name_info;
}

using buffer_kind_names_t = std::map<rocprofiler_buffer_tracing_kind_t, std::string_view>;
using buffer_kind_operation_names_t =
    std::map<rocprofiler_buffer_tracing_kind_t, std::map<uint32_t, std::string_view>>;

struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};
};

inline buffer_name_info
get_buffer_tracing_names()
{
    static const auto supported_kinds = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
        ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
        ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
        ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
        ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API,
    };

    auto cb_name_info = buffer_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_buffer_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t   operation,
                                               void*                             data_v) {
        auto* name_info_v = static_cast<buffer_name_info*>(data_v);

        if(supported_kinds.count(kindv) > 0)
        {
            const char* name = nullptr;
            ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                 kindv, operation, &name, nullptr),
                             "query buffer tracing kind operation name");
            EXPECT_TRUE(name != nullptr) << "kind=" << kindv << ", operation=" << operation;
            if(name) name_info_v->operation_names[kindv][operation] = name;
        }
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<buffer_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr),
                         "query buffer tracing kind operation name");
        EXPECT_TRUE(name != nullptr) << "kind=" << kind;
        if(name) name_info_v->kind_names[kind] = name;

        if(supported_kinds.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating buffer tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb,
                                                              static_cast<void*>(&cb_name_info)),
                     "iterating buffer tracing kinds");

    return cb_name_info;
}
}  // namespace
