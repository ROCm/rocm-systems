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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file tests/tools/json-tool.cpp
 *
 * @brief Test rocprofiler tool
 */

#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/serialization.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <unistd.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace client
{
namespace
{
struct source_location
{
    std::string function = {};
    std::string file     = {};
    uint32_t    line     = 0;
    std::string context  = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("function", function));
        ar(cereal::make_nvp("file", file));
        ar(cereal::make_nvp("line", line));
        ar(cereal::make_nvp("context", context));
    }
};

using call_stack_t        = std::vector<source_location>;
using buffer_kind_names_t = std::map<rocprofiler_buffer_tracing_kind_t, std::string>;
using buffer_kind_operation_names_t =
    std::map<rocprofiler_buffer_tracing_kind_t, std::map<uint32_t, std::string>>;

using callback_kind_names_t = std::map<rocprofiler_callback_tracing_kind_t, std::string>;
using callback_kind_operation_names_t =
    std::map<rocprofiler_callback_tracing_kind_t, std::map<uint32_t, std::string>>;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

struct callback_name_info
{
    callback_kind_names_t           kind_names      = {};
    callback_kind_operation_names_t operation_names = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("kind_names", kind_names));
        ar(cereal::make_nvp("operation_names", operation_names));
    }
};

struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("kind_names", kind_names));
        ar(cereal::make_nvp("operation_names", operation_names));
    }
};

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;

callback_name_info
get_callback_tracing_names()
{
    static const auto supported = std::unordered_set<rocprofiler_callback_tracing_kind_t>{
        ROCPROFILER_CALLBACK_TRACING_HSA_API,
        ROCPROFILER_CALLBACK_TRACING_MARKER_API,
        ROCPROFILER_CALLBACK_TRACING_HIP_API};

    auto cb_name_info = callback_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_callback_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<callback_name_info*>(data_v);

            if(supported.count(kindv) > 0)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query buffer tracing kind operation name");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }
            return 0;
        };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<callback_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query buffer tracing kind operation name");
        if(name) name_info_v->kind_names[kind] = name;

        if(supported.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating buffer tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb,
                                                                static_cast<void*>(&cb_name_info)),
                     "iterating buffer tracing kinds");

    return cb_name_info;
}

buffer_name_info
get_buffer_tracing_names()
{
    static const auto supported = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
        ROCPROFILER_BUFFER_TRACING_HSA_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_API,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY};

    auto cb_name_info = buffer_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_buffer_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<buffer_name_info*>(data_v);

            if(supported.count(kindv) > 0)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query buffer tracing kind operation name");
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
        if(name) name_info_v->kind_names[kind] = name;

        if(supported.count(kind) > 0)
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

using callback_payload_t =
    std::variant<rocprofiler_callback_tracing_code_object_load_data_t,
                 rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t,
                 rocprofiler_callback_tracing_hsa_api_data_t,
                 rocprofiler_callback_tracing_marker_api_data_t>;

using callback_arg_array_t = std::vector<std::pair<std::string, std::string>>;

template <typename ArchiveT>
void
serialize_args(ArchiveT& ar, const callback_arg_array_t& data)
{
    ar.setNextName("args");
    ar.startNode();
    for(const auto& itr : data)
        ar(cereal::make_nvp(itr.first, itr.second));
    ar.finishNode();
}

int
save_args(rocprofiler_callback_tracing_kind_t,
          uint32_t,
          uint32_t,
          const char* arg_name,
          const char* arg_value_str,
          const void* const,
          void* data)
{
    auto* argvec = static_cast<callback_arg_array_t*>(data);
    argvec->emplace_back(arg_name, arg_value_str);
    return 0;
}

struct code_object_callback_record_t
{
    uint64_t                                             timestamp = 0;
    rocprofiler_callback_tracing_record_t                record    = {};
    rocprofiler_callback_tracing_code_object_load_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        ar(cereal::make_nvp("record", record));
        ar(cereal::make_nvp("payload", payload));
    }
};

struct kernel_symbol_callback_record_t
{
    uint64_t                                                               timestamp = 0;
    rocprofiler_callback_tracing_record_t                                  record    = {};
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        ar(cereal::make_nvp("record", record));
        ar(cereal::make_nvp("payload", payload));
    }
};

struct hsa_api_callback_record_t
{
    uint64_t                                    timestamp = 0;
    rocprofiler_callback_tracing_record_t       record    = {};
    rocprofiler_callback_tracing_hsa_api_data_t payload   = {};
    callback_arg_array_t                        args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        ar(cereal::make_nvp("record", record));
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct hip_api_callback_record_t
{
    uint64_t                                    timestamp = 0;
    rocprofiler_callback_tracing_record_t       record    = {};
    rocprofiler_callback_tracing_hip_api_data_t payload   = {};
    callback_arg_array_t                        args      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        ar(cereal::make_nvp("record", record));
        ar(cereal::make_nvp("payload", payload));
        serialize_args(ar, args);
    }
};

struct marker_api_callback_record_t
{
    uint64_t                                       timestamp = 0;
    rocprofiler_callback_tracing_record_t          record    = {};
    rocprofiler_callback_tracing_marker_api_data_t payload   = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        ar(cereal::make_nvp("record", record));
        ar(cereal::make_nvp("payload", payload));
    }
};

auto code_object_records   = std::deque<code_object_callback_record_t>{};
auto kernel_symbol_records = std::deque<kernel_symbol_callback_record_t>{};
auto hsa_api_cb_records    = std::deque<hsa_api_callback_record_t>{};
auto marker_api_cb_records = std::deque<marker_api_callback_record_t>{};
auto hip_api_cb_records    = std::deque<hip_api_callback_record_t>{};

rocprofiler_thread_id_t
push_external_correlation();

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* /*user_data*/,
                      void* /*callback_data*/)
{
    static auto _mutex = std::mutex{};
    auto        _lk    = std::unique_lock<std::mutex>{_mutex};
    auto        ts     = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts), "get timestamp");

    static thread_local auto _once = std::once_flag{};
    std::call_once(_once, [&record]() {
        // account for the fact that we are not wrapping pthread_create so the
        // first external correlation id on a thread wont have updated value
        record.correlation_id.external.value = push_external_correlation();
    });

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    {
        if(record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD)
        {
            auto data_v =
                *static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
            code_object_records.emplace_back(code_object_callback_record_t{ts, record, data_v});
        }
        else if(record.operation ==
                ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
        {
            auto data_v = *static_cast<kernel_symbol_data_t*>(record.payload);
            kernel_symbol_records.emplace_back(kernel_symbol_callback_record_t{ts, record, data_v});
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_hsa_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        rocprofiler_iterate_callback_tracing_kind_operation_args(record, save_args, &args);
        hsa_api_cb_records.emplace_back(
            hsa_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_hip_api_data_t*>(record.payload);
        auto  args = callback_arg_array_t{};
        rocprofiler_iterate_callback_tracing_kind_operation_args(record, save_args, &args);
        hip_api_cb_records.emplace_back(
            hip_api_callback_record_t{ts, record, *data, std::move(args)});
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_API)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
        marker_api_cb_records.emplace_back(marker_api_callback_record_t{ts, record, *data});
    }
    else
    {
        throw std::runtime_error{"unsupported callback kind"};
    }
}

auto hsa_api_bf_records      = std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>{};
auto marker_api_bf_records   = std::deque<rocprofiler_buffer_tracing_marker_api_record_t>{};
auto hip_api_bf_records      = std::deque<rocprofiler_buffer_tracing_hip_api_record_t>{};
auto kernel_dispatch_records = std::deque<rocprofiler_buffer_tracing_kernel_dispatch_record_t>{};
auto memory_copy_records     = std::deque<rocprofiler_buffer_tracing_memory_copy_record_t>{};

void
tool_tracing_buffered(rocprofiler_context_id_t /*context*/,
                      rocprofiler_buffer_id_t /*buffer_id*/,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
    // std::cerr << "[" << getpid() << "][" << __FUNCTION__ << "] buffer flush callback for "
    //           << num_headers << " records...\n"
    //           << std::flush;

    static auto _mutex = std::mutex{};
    auto        _lk    = std::unique_lock<std::mutex>{_mutex};

    assert(user_data != nullptr);
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

    if(num_headers == 0)
        throw std::runtime_error{
            "rocprofiler invoked a buffer callback with no headers. this should never happen"};
    else if(headers == nullptr)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                                 "array of headers. this should never happen"};

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(header->hash !=
                rocprofiler_record_header_compute_hash(header->category, header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);

                hsa_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_marker_api_record_t*>(header->payload);

                marker_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_API)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

                hip_api_bf_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            {
                auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                    header->payload);

                kernel_dispatch_records.emplace_back(*record);
            }
            else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            {
                auto* record =
                    static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

                memory_copy_records.emplace_back(*record);
            }
            else
            {
                throw std::runtime_error{
                    "unexpected rocprofiler_record_header_t tracing category kind"};
            }
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }
}

void
thread_precreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread about to be created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

void
thread_postcreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread was created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

bool
is_active(rocprofiler_context_id_t ctx)
{
    int  status = 0;
    auto errc   = rocprofiler_context_is_active(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

void
start();

void
stop();

void
flush();

// contexts
rocprofiler_context_id_t hsa_api_callback_ctx    = {};
rocprofiler_context_id_t hip_api_callback_ctx    = {};
rocprofiler_context_id_t marker_api_callback_ctx = {};
rocprofiler_context_id_t code_object_ctx         = {};
rocprofiler_context_id_t hsa_api_buffered_ctx    = {};
rocprofiler_context_id_t hip_api_buffered_ctx    = {};
rocprofiler_context_id_t marker_api_buffered_ctx = {};
rocprofiler_context_id_t kernel_dispatch_ctx     = {};
rocprofiler_context_id_t memory_copy_ctx         = {};
// buffers
rocprofiler_buffer_id_t hsa_api_buffered_buffer    = {};
rocprofiler_buffer_id_t hip_api_buffered_buffer    = {};
rocprofiler_buffer_id_t marker_api_buffered_buffer = {};
rocprofiler_buffer_id_t kernel_dispatch_buffer     = {};
rocprofiler_buffer_id_t memory_copy_buffer         = {};

auto contexts = std::unordered_map<std::string_view, rocprofiler_context_id_t*>{
    {"HSA_API_CALLBACK", &hsa_api_callback_ctx},
    {"HIP_API_CALLBACK", &hip_api_callback_ctx},
    {"MARKER_API_CALLBACK", &marker_api_callback_ctx},
    {"CODE_OBJECT", &code_object_ctx},
    {"HSA_API_BUFFERED", &hsa_api_buffered_ctx},
    {"HIP_API_BUFFERED", &hip_api_buffered_ctx},
    {"MARKER_API_BUFFERED", &marker_api_buffered_ctx},
    {"KERNEL_DISPATCH", &kernel_dispatch_ctx},
    {"MEMORY_COPY", &memory_copy_ctx}};

auto buffers = std::array<rocprofiler_buffer_id_t*, 5>{&hsa_api_buffered_buffer,
                                                       &hip_api_buffered_buffer,
                                                       &marker_api_buffered_buffer,
                                                       &kernel_dispatch_buffer,
                                                       &memory_copy_buffer};

auto agents = std::vector<rocprofiler_agent_t>{};

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    rocprofiler_available_agents_cb_t iterate_cb =
        [](const rocprofiler_agent_t** agents_arr, size_t num_agents, void* user_data) {
            auto* agents_v = static_cast<std::vector<rocprofiler_agent_t>*>(user_data);
            for(size_t i = 0; i < num_agents; ++i)
                agents_v->emplace_back(*agents_arr[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    client_fini_func = fini_func;

    for(auto itr : contexts)
    {
        ROCPROFILER_CALL(rocprofiler_create_context(itr.second), "context creation");
    }

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(hsa_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HSA_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "hsa api callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(hip_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "hip api callback tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(code_object_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(marker_api_callback_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "hsa api tracing service configure");

    constexpr auto buffer_size = 8192;
    constexpr auto watermark   = 7936;

    ROCPROFILER_CALL(rocprofiler_create_buffer(hsa_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &hsa_api_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(hip_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &hip_api_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(marker_api_buffered_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &marker_api_buffered_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(kernel_dispatch_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &kernel_dispatch_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_create_buffer(memory_copy_ctx,
                                               buffer_size,
                                               watermark,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_buffered,
                                               tool_data,
                                               &memory_copy_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(hsa_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_HSA_API,
                                                     nullptr,
                                                     0,
                                                     hsa_api_buffered_buffer),
        "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(hip_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_HIP_API,
                                                     nullptr,
                                                     0,
                                                     hip_api_buffered_buffer),
        "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(marker_api_buffered_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MARKER_API,
                                                     nullptr,
                                                     0,
                                                     marker_api_buffered_buffer),
        "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(kernel_dispatch_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                     nullptr,
                                                     0,
                                                     kernel_dispatch_buffer),
        "buffer tracing service for kernel dispatch configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(memory_copy_ctx,
                                                     ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                     nullptr,
                                                     0,
                                                     memory_copy_buffer),
        "buffer tracing service for memory copy configure");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "creating callback thread");

    for(auto* itr : buffers)
    {
        ROCPROFILER_CALL(rocprofiler_assign_callback_thread(*itr, client_thread),
                         "assignment of thread for buffer");
    }

    for(auto itr : contexts)
    {
        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(*itr.second, &valid_ctx),
                         "context validity check");
        if(valid_ctx == 0)
        {
            // notify rocprofiler that initialization failed
            // and all the contexts, buffers, etc. created
            // should be ignored
            return -1;
        }
    }

    // environment variable to select which contexts to collect
    auto* context_settings_env = getenv("ROCPROFILER_TOOL_CONTEXTS");
    if(context_settings_env != nullptr && !std::string_view{context_settings_env}.empty())
    {
        auto context_settings = std::string{context_settings_env};

        // ignore case
        for(auto& itr : context_settings)
            itr = toupper(itr);

        // if context is not in string, set the pointer to null in the contexts array
        auto options = std::stringstream{};
        for(auto& itr : contexts)
        {
            options << "\n\t- " << itr.first;
            auto pos = context_settings.find(itr.first);
            if(pos == std::string::npos)
                itr.second = nullptr;
            else
                context_settings.erase(pos, itr.first.length());
        }

        // detect if there are any invalid entries
        if(context_settings.find_first_not_of(" ,;:\t\n\r") != std::string::npos)
        {
            auto filename = std::string_view{__FILE__};
            auto msg      = std::stringstream{};
            msg << "[rocprofiler-sdk-json-tool][" << filename.substr(filename.find_last_of('/') + 1)
                << ":" << __LINE__ << "] invalid specification of ROCPROFILER_TOOL_CONTEXTS ('"
                << context_settings_env << "'). Valid choices are: " << options.str();
            throw std::runtime_error{msg.str()};
        }
    }

    start();

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    static std::atomic_flag _once = ATOMIC_FLAG_INIT;
    if(_once.test_and_set()) return;

    stop();
    flush();

    std::cerr << "[" << getpid() << "][" << __FUNCTION__
              << "] Finalizing... agents=" << agents.size()
              << ", code_object_callback_records=" << code_object_records.size()
              << ", kernel_symbol_callback_records=" << kernel_symbol_records.size()
              << ", hsa_api_callback_records=" << hsa_api_cb_records.size()
              << ", hip_api_callback_records=" << hip_api_cb_records.size()
              << ", marker_api_callback_records=" << marker_api_cb_records.size()
              << ", kernel_dispatch_records=" << kernel_dispatch_records.size()
              << ", memory_copy_records=" << memory_copy_records.size()
              << ", hsa_api_bf_records=" << hsa_api_bf_records.size()
              << ", hip_api_bf_records=" << hip_api_bf_records.size()
              << ", marker_api_bf_records=" << marker_api_bf_records.size() << " ...\n"
              << std::flush;

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    if(_call_stack)
    {
        _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});
    }

    auto ofname = std::string{"rocprofiler-tool-results.json"};
    if(auto* eofname = getenv("ROCPROFILER_TOOL_OUTPUT_FILE")) ofname = eofname;

    std::ostream* ofs     = nullptr;
    auto          cleanup = std::function<void(std::ostream*&)>{};
    if(ofname == "stdout")
        ofs = &std::cout;
    else if(ofname == "stderr")
        ofs = &std::cerr;
    else
    {
        ofs = new std::ofstream{ofname};
        if(ofs && *ofs)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] Outputting collected data to " << ofname << "...\n"
                      << std::flush;
            cleanup = [](std::ostream*& _os) { delete _os; };
        }
        else
        {
            std::cerr << "Error outputting to " << ofname << ". Redirecting to stderr...\n"
                      << std::flush;
            ofname = "stderr";
            ofs    = &std::cerr;
        }
    }

    {
        using JSONOutputArchive = cereal::MinimalJSONOutputArchive;

        constexpr auto json_prec          = 32;
        constexpr auto json_indent        = JSONOutputArchive::Options::IndentChar::space;
        auto           json_opts          = JSONOutputArchive::Options{json_prec, json_indent, 1};
        auto           json_ar            = JSONOutputArchive{*ofs, json_opts};
        auto           buffer_name_info   = get_buffer_tracing_names();
        auto           callback_name_info = get_callback_tracing_names();

        json_ar.setNextName("rocprofiler-sdk-json-tool");
        json_ar.startNode();

        json_ar(cereal::make_nvp("agents", agents));
        if(_call_stack) json_ar(cereal::make_nvp("call_stack", *_call_stack));

        json_ar.setNextName("callback_records");
        json_ar.startNode();
        try
        {
            json_ar(cereal::make_nvp("names", callback_name_info));
            json_ar(cereal::make_nvp("code_objects", code_object_records));
            json_ar(cereal::make_nvp("kernel_symbols", kernel_symbol_records));
            json_ar(cereal::make_nvp("hsa_api_traces", hsa_api_cb_records));
            json_ar(cereal::make_nvp("hip_api_traces", hip_api_cb_records));
            json_ar(cereal::make_nvp("marker_api_traces", marker_api_cb_records));
        } catch(std::exception& e)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] threw an exception: " << e.what() << "\n"
                      << std::flush;
        }
        json_ar.finishNode();

        json_ar.setNextName("buffer_records");
        json_ar.startNode();
        try
        {
            json_ar(cereal::make_nvp("names", buffer_name_info));
            json_ar(cereal::make_nvp("kernel_dispatches", kernel_dispatch_records));
            json_ar(cereal::make_nvp("memory_copies", memory_copy_records));
            json_ar(cereal::make_nvp("hsa_api_traces", hsa_api_bf_records));
            json_ar(cereal::make_nvp("hip_api_traces", hip_api_bf_records));
            json_ar(cereal::make_nvp("marker_api_traces", marker_api_bf_records));
        } catch(std::exception& e)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] threw an exception: " << e.what() << "\n"
                      << std::flush;
        }
        json_ar.finishNode();

        json_ar.finishNode();
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);

    std::cerr << "[" << getpid() << "][" << __FUNCTION__ << "] Finalization complete.\n"
              << std::flush;

    delete _call_stack;
}

void
start()
{
    for(auto itr : contexts)
    {
        if(itr.second && !is_active(*itr.second))
        {
            ROCPROFILER_CALL(rocprofiler_start_context(*itr.second), "context start");
        }
    }
}

void
stop()
{
    for(auto itr : contexts)
    {
        if(itr.second && is_active(*itr.second))
        {
            ROCPROFILER_CALL(rocprofiler_stop_context(*itr.second), "context stop");
        }
    }
}

void
flush()
{
    for(auto* itr : buffers)
    {
        if(!itr) continue;
        auto status = rocprofiler_flush_buffer(*itr);
        if(status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        {
            ROCPROFILER_CALL(status, "buffer flush");
        }
    }
}

rocprofiler_thread_id_t
push_external_correlation()
{
    auto tid = rocprofiler_thread_id_t{};
    ROCPROFILER_CALL(rocprofiler_get_thread_id(&tid), "get thread id");

    for(auto itr : contexts)
    {
        if(itr.second)
        {
            ROCPROFILER_CALL(rocprofiler_push_external_correlation_id(
                                 *itr.second, tid, rocprofiler_user_data_t{.value = tid}),
                             "push external correlation");
        }
    }
    return tid;
}
}  // namespace
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "rocprofiler-sdk-json-tool";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    auto* client_tool_data = new std::vector<client::source_location>{};

    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(
                         client::thread_precreate,
                         client::thread_postcreate,
                         ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
                             ROCPROFILER_MARKER_LIBRARY,
                         static_cast<void*>(client_tool_data)),
                     "registration for thread creation notifications");

    std::atexit([]() {
        if(client::client_fini_func) client::client_fini_func(*client::client_id);
    });
    std::at_quick_exit([]() {
        if(client::client_fini_func) client::client_fini_func(*client::client_id);
    });

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
