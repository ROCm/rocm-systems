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
#include <ratio>
#include <string>
#include <string_view>
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

using call_stack_t          = std::vector<source_location>;
using callback_kind_names_t = std::map<rocprofiler_service_callback_tracing_kind_t, const char*>;
using callback_kind_operation_names_t =
    std::map<rocprofiler_service_callback_tracing_kind_t, std::map<uint32_t, const char*>>;

struct callback_name_info
{
    callback_kind_names_t           kind_names      = {};
    callback_kind_operation_names_t operation_names = {};
};

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};

void
print_call_stack(const call_stack_t& _call_stack)
{
    namespace fs = ::std::filesystem;

    auto ofname = std::string{"api_callback_trace.log"};
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
    *ofs << std::left;
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

callback_name_info
get_callback_id_names()
{
    auto cb_name_info = callback_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_service_callback_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<callback_name_info*>(data_v);

            if(kindv == ROCPROFILER_CALLBACK_TRACING_HSA_API)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query callback tracing kind operation name");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }
            return 0;
        };

    //
    //  callback for each callback kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_service_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<callback_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query callback tracing kind operation name");
        if(name) name_info_v->kind_names[kind] = name;

        if(kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
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

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 callback_data)
{
    assert(callback_data != nullptr);

    auto     now = std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t dt  = 0;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        user_data->value = now;
    else
        dt = (now - user_data->value);

    auto info = std::stringstream{};
    info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
         << record.correlation_id.internal << ", kind=" << record.kind
         << ", operation=" << std::setw(3) << record.operation << ", phase=" << record.phase
         << ", dt_nsec=" << std::setw(6) << dt;

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
    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                         record, info_data_cb, static_cast<void*>(&info_data)),
                     "Failure iterating trace operation args");

    auto info_data_str = info_data.str();
    if(!info_data_str.empty()) info << " " << info_data_str << ")";

    static auto _mutex = std::mutex{};
    _mutex.lock();
    static_cast<call_stack_t*>(callback_data)
        ->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
    _mutex.unlock();
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    callback_name_info name_info = get_callback_id_names();

    for(const auto& itr : name_info.operation_names)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << static_cast<int32_t>(itr.first) << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_callback_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            name_info.kind_names.at(itr.first)});

        for(const auto& ditr : itr.second)
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << static_cast<int32_t>(ditr.first) << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_callback_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{ditr.second}});
        }
    }

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HSA_API,
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
