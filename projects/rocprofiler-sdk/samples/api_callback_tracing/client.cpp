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
 * @file samples/api_callback_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
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
store_callback_id_names(call_stack_t* tool_data)
{
    //
    // callback for each kind operation
    //
    static auto tracing_operation_names_cb =
        [](rocprofiler_service_callback_tracing_kind_t /*kindv*/,
           uint32_t /*operation*/,
           const char* operation_name,
           void*       data_v) {
            static_cast<call_stack_t*>(data_v)->emplace_back(
                source_location{"rocprofiler_iterate_callback_tracing_kind_operation_names",
                                __FILE__,
                                __LINE__,
                                std::string{"    "} + std::string{operation_name}});
            return 0;
        };

    //
    //  callback for each callback kind (i.e. domain)
    //
    static auto tracing_kind_names_cb = [](rocprofiler_service_callback_tracing_kind_t kind,
                                           const char*                                 kind_name,
                                           void*                                       data) {
        //  store the callback kind name
        static_cast<call_stack_t*>(data)->emplace_back(source_location{
            "rocprofiler_iterate_callback_tracing_kind_names     ", __FILE__, __LINE__, kind_name});

        // store the operation names for the HSA API
        if(kind == ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API)
        {
            rocprofiler_iterate_callback_tracing_kind_operation_names(
                kind, tracing_operation_names_cb, data);
        }

        return 0;
    };

    rocprofiler_iterate_callback_tracing_kind_names(tracing_kind_names_cb,
                                                    static_cast<void*>(tool_data));
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record, void* user_data)
{
    assert(user_data != nullptr);

    auto info = std::stringstream{};
    info << "tid=" << record.thread_id << ", cid=" << record.correlation_id.id
         << ", kind=" << record.kind << ", operation=" << record.operation
         << ", phase=" << record.phase;

    auto info_data_cb = [](rocprofiler_service_callback_tracing_kind_t,
                           uint32_t,
                           uint32_t          arg_num,
                           const char*       arg_name,
                           const char*       arg_value_str,
                           const void* const arg_value_addr,
                           void*             cb_data) -> int {
        auto& dss = *static_cast<std::stringstream*>(cb_data);
        dss << ((arg_num == 0) ? "(" : ", ");
        dss << arg_num << ": " << arg_name << "=" << arg_value_str;
        (void) arg_value_addr;
        return 0;
    };

    auto info_data = std::stringstream{};
    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_operation_args(
                         record, info_data_cb, static_cast<void*>(&info_data)),
                     "Failure iterating trace operation args");

    auto info_data_str = info_data.str();
    if(!info_data_str.empty()) info << " " << info_data_str << ")";

    static auto _mutex = std::mutex{};
    _mutex.lock();
    static_cast<call_stack_t*>(user_data)->emplace_back(
        source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
    _mutex.unlock();
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    static_cast<call_stack_t*>(tool_data)->emplace_back(
        source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    store_callback_id_names(static_cast<call_stack_t*>(tool_data));

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

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
{}

void
shutdown()
{
    if(client_id) client_fini_func(*client_id);
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
}

void
stop()
{
    int status = 0;
    ROCPROFILER_CALL(rocprofiler_is_initialized(&status), "failed to retrieve init status");
    if(status != 0)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop failed");
    }
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

    // demonstration of alternative way to get the version info
    {
        auto version_info = std::array<uint32_t, 3>{};
        ROCPROFILER_CALL(
            rocprofiler_get_version(&version_info.at(0), &version_info.at(1), &version_info.at(2)),
            "failed to get version info");

        if(std::array<uint32_t, 3>{major, minor, patch} != version_info)
        {
            throw std::runtime_error{"version info mismatch"};
        }
    }

    // data passed around all the callbacks
    auto* client_tool_data = new std::vector<client::source_location>{};

    // add first entry
    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
