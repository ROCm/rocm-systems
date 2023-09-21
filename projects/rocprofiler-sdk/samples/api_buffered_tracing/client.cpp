// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/api_buffered_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler/buffer.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/internal_threading.h>
#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::cerr << #result << " failed with error code " << CHECKSTATUS << std::endl;        \
            throw std::runtime_error(#result " failure");                                          \
        }                                                                                          \
    }

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
};

using call_stack_t = std::vector<source_location>;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};
rocprofiler_buffer_id_t       client_buffer    = {};

void
print_call_stack(const call_stack_t& _call_stack)
{
    namespace fs = ::std::filesystem;

    size_t n = 0;
    for(const auto& itr : _call_stack)
    {
        std::clog << std::setw(2) << ++n << "/" << std::setw(2) << _call_stack.size() << " ";
        std::clog << "[" << fs::path{itr.file}.filename() << ":" << itr.line << "] "
                  << std::setw(20) << std::left << itr.function;
        if(!itr.context.empty()) std::clog << " :: " << itr.context;
        std::clog << "\n";
    }

    std::clog << std::flush;
}

void
store_buffer_id_names(call_stack_t* tool_data)
{
    //
    // buffered for each kind operation
    //
    static auto tracing_operation_names_cb = [](rocprofiler_service_buffer_tracing_kind_t /*kindv*/,
                                                uint32_t /*operation*/,
                                                const char* operation_name,
                                                void*       data_v) {
        static_cast<call_stack_t*>(data_v)->emplace_back(
            source_location{"rocprofiler_iterate_buffer_trace_kind_operation_names",
                            __FILE__,
                            __LINE__,
                            std::string{"    "} + std::string{operation_name}});
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_names_cb =
        [](rocprofiler_service_buffer_tracing_kind_t kind, const char* kind_name, void* data) {
            //  store the buffer kind name
            static_cast<call_stack_t*>(data)->emplace_back(
                source_location{"rocprofiler_iterate_buffer_trace_kind_names          ",
                                __FILE__,
                                __LINE__,
                                kind_name});

            // store the operation names for the HSA API
            if(kind == ROCPROFILER_SERVICE_BUFFER_TRACING_HSA_API)
            {
                rocprofiler_iterate_buffer_tracing_kind_operation_names(
                    kind, tracing_operation_names_cb, data);
            }

            return 0;
        };

    rocprofiler_iterate_buffer_tracing_kind_names(tracing_kind_names_cb,
                                                  static_cast<void*>(tool_data));
}

void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
    assert(user_data != nullptr);

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
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_SERVICE_BUFFER_TRACING_HSA_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);
            auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle << ", cid=" << record->correlation_id.id
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", drop_count=" << drop_count << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp;

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("start > end");

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }
}

void
thread_precreate(rocprofiler_internal_thread_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread about to be created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

void
thread_postcreate(rocprofiler_internal_thread_library_t lib, void* tool_data)
{
    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__,
                        __FILE__,
                        __LINE__,
                        std::string{"internal thread was created by rocprofiler (lib="} +
                            std::to_string(lib) + ")"});
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    store_buffer_id_names(static_cast<call_stack_t*>(tool_data));

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation failed");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_SERVICE_BUFFER_TRACING_HSA_API, nullptr, 0, client_buffer),
        "buffer tracing service failed to configure");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");

    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(client_buffer, client_thread),
                     "failed to assign thread for buffer");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "failure checking context validity");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
}
}  // namespace

void
setup()
{
    ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure),
                     "failed to force configuration");
}

void
shutdown()
{
    if(client_id)
    {
        auto status = ROCPROFILER_STATUS_SUCCESS;
        while((status = rocprofiler_flush_buffer(client_buffer)) ==
              ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        ROCPROFILER_CALL(status, "rocprofiler_flush_buffer failed");
        while((status = rocprofiler_flush_buffer(client_buffer)) ==
              ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        client_fini_func(*client_id);
    }
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
}

void
stop()
{
    ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop failed");
}
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
    id->name = "ExampleTool";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler v" << major << "." << minor << "." << patch << " ("
         << runtime_version << ")";

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
                     "failed to register for thread creation notifications");

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
