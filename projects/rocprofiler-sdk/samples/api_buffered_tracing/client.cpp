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
 * @file samples/api_buffered_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler/buffer.h>
#include <rocprofiler/callback_tracing.h>
#include <rocprofiler/external_correlation.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/internal_threading.h>
#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
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
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
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

using call_stack_t        = std::vector<source_location>;
using buffer_kind_names_t = std::map<rocprofiler_service_buffer_tracing_kind_t, const char*>;
using buffer_kind_operation_names_t =
    std::map<rocprofiler_service_buffer_tracing_kind_t, std::map<uint32_t, const char*>>;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};
};

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};
rocprofiler_buffer_id_t       client_buffer    = {};
buffer_name_info              client_name_info = {};
kernel_symbol_map_t           client_kernels   = {};

void
print_call_stack(const call_stack_t& _call_stack)
{
    namespace fs = ::std::filesystem;

    auto ofname = std::string{"api_buffered_trace.log"};
    if(auto* eofname = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE")) ofname = eofname;

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
            cleanup = [](std::ostream*& _os) { delete _os; };
        else
        {
            std::cerr << "Error outputting to " << ofname << ". Redirecting to stderr...\n";
            ofname = "stderr";
            ofs    = &std::cerr;
        }
    }

    std::cout << "Outputting collected data to " << ofname << "...\n" << std::flush;

    size_t n = 0;
    for(const auto& itr : _call_stack)
    {
        *ofs << std::left << std::setw(2) << ++n << "/" << std::setw(2) << _call_stack.size()
             << " [" << fs::path{itr.file}.filename() << ":" << itr.line << "] " << std::setw(20)
             << itr.function;
        if(!itr.context.empty()) *ofs << " :: " << itr.context;
        *ofs << "\n";
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);
}

buffer_name_info
get_buffer_tracing_names()
{
    auto cb_name_info = buffer_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_service_buffer_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<buffer_name_info*>(data_v);

            if(kindv == ROCPROFILER_BUFFER_TRACING_HSA_API)
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
    static auto tracing_kind_cb = [](rocprofiler_service_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<buffer_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr),
                         "query buffer tracing kind operation name");
        if(name) name_info_v->kind_names[kind] = name;

        if(kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
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

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CALL(flush_status, "buffer flush");
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation ==
                ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            client_kernels.emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            client_kernels.erase(data->kernel_id);
        }
    }

    (void) user_data;
    (void) callback_data;
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
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);
            auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.operation_names[record->kind][record->operation];

            if(record->start_timestamp > record->end_timestamp)
            {
                auto msg = std::stringstream{};
                msg << "hsa api: start > end (" << record->start_timestamp << " > "
                    << record->end_timestamp
                    << "). diff = " << (record->start_timestamp - record->end_timestamp);
                std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
                // throw std::runtime_error{msg.str()};
            }

            static_cast<call_stack_t*>(user_data)->emplace_back(
                source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

            auto info = std::stringstream{};

            info << "agent_id=" << record->agent_id.handle
                 << ", queue_id=" << record->queue_id.handle << ", kernel_id=" << record->kernel_id
                 << ", kernel=" << client_kernels.at(record->kernel_id).kernel_name
                 << ", context=" << context.handle << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp
                 << ", private_segment_size=" << record->private_segment_size
                 << ", group_segment_size=" << record->group_segment_size << ", workgroup_size=("
                 << record->workgroup_size.x << "," << record->workgroup_size.y << ","
                 << record->workgroup_size.z << "), grid_size=(" << record->grid_size.x << ","
                 << record->grid_size.y << "," << record->grid_size.z << ")";

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("kernel dispatch: start > end");

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

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    client_name_info = get_buffer_tracing_names();

    for(const auto& itr : client_name_info.operation_names)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << static_cast<int32_t>(itr.first) << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_buffer_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            client_name_info.kind_names.at(itr.first)});

        for(const auto& ditr : itr.second)
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << static_cast<int32_t>(ditr.first) << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_buffer_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{ditr.second}});
        }
    }

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                         client_ctx, ROCPROFILER_BUFFER_TRACING_HSA_API, nullptr, 0, client_buffer),
                     "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer),
        "buffer tracing service for kernel dispatch configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, client_buffer),
        "buffer tracing service for memory copy configure");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "creating callback thread");

    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(client_buffer, client_thread),
                     "assignment of thread for buffer");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "context validity check");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start");

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
    ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure), "force configuration");
}

void
shutdown()
{
    if(client_id)
    {
        ROCPROFILER_CALL(rocprofiler_flush_buffer(client_buffer), "buffer flush");
        client_fini_func(*client_id);
    }
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "context start");
}

void
identify(uint64_t val)
{
    auto _tid = rocprofiler_thread_id_t{};
    rocprofiler_get_thread_id(&_tid);
    rocprofiler_push_external_correlation_id(
        client_ctx, _tid, rocprofiler_user_data_t{.value = val});
}

void
stop()
{
    ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "context stop");
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
                     "registration for thread creation notifications");

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
