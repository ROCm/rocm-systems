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

#include "helper.hpp"
#include "config.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <glog/logging.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

rocprofiler_tool_buffer_name_info_t
get_buffer_id_names()
{
    static auto supported = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
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
        ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
    };

    auto cb_name_info = rocprofiler_tool_buffer_name_info_t{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_buffer_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<rocprofiler_tool_buffer_name_info_t*>(data_v);

            if(supported.count(kindv) > 0)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query buffer failed");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }

            return 0;
        };

    //
    //  callback for each kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<rocprofiler_tool_buffer_name_info_t*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr),
                         "query buffer failed");

        if(name) name_info_v->kind_names[kind] = name;

        if(supported.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "query buffer failed");
        }

        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb,
                                                              static_cast<void*>(&cb_name_info)),
                     "iterate_buffer failed");

    return cb_name_info;
}

rocprofiler_tool_callback_name_info_t
get_callback_id_names()
{
    static auto supported = std::unordered_set<rocprofiler_callback_tracing_kind_t>{
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

    auto cb_name_info = rocprofiler_tool_callback_name_info_t{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_callback_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<rocprofiler_tool_callback_name_info_t*>(data_v);

            if(supported.count(kindv) > 0)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query callback failed");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }

            return 0;
        };

    //
    //  callback for each kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<rocprofiler_tool_callback_name_info_t*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query callback failed");

        if(name) name_info_v->kind_names[kind] = name;

        if(supported.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "query callback failed");
        }

        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb,
                                                                static_cast<void*>(&cb_name_info)),
                     "iterate_callback failed");

    return cb_name_info;
}
