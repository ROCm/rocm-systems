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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t status = (result);                                                    \
        if(status != ROCPROFILER_STATUS_SUCCESS)                                                   \
        {                                                                                          \
            std::cerr << "Error: " << msg << " failed with status " << status << std::endl;        \
            return -1;                                                                             \
        }                                                                                          \
    }

namespace
{
struct api_trace_data_t
{
    uint64_t    timestamp = 0;
    int32_t     kind      = 0;
    int32_t     operation = 0;
    int32_t     phase     = 0;
    std::string name      = {};

    template <typename ArchiveT>
    void serialize(ArchiveT& ar)
    {
        ar(cereal::make_nvp("timestamp", timestamp));
        ar(cereal::make_nvp("kind", kind));
        ar(cereal::make_nvp("operation", operation));
        ar(cereal::make_nvp("phase", phase));
        ar(cereal::make_nvp("name", name));
    }
};

std::mutex                    trace_mutex;
std::vector<api_trace_data_t> traces;
rocprofiler_context_id_t      client_ctx = {.handle = 0};

const char*
get_operation_name(rocprofiler_callback_tracing_kind_t kind, int32_t operation)
{
    const char* name = nullptr;
    rocprofiler_query_callback_tracing_kind_operation_name(kind, operation, &name, nullptr);
    return (name) ? name : "<unknown>";
}

// Callback for HIP API tracing
void
hip_api_callback(rocprofiler_callback_tracing_record_t record,
                 rocprofiler_user_data_t*              user_data,
                 void*                                 callback_data)
{
    (void) user_data;
    (void) callback_data;

    std::lock_guard<std::mutex> lock(trace_mutex);
    traces.push_back(api_trace_data_t{0,  // timestamp not available in callback tracing
                                      record.kind,
                                      record.operation,
                                      record.phase,
                                      get_operation_name(record.kind, record.operation)});
}

// Callback for ROCTx tracing
void
marker_api_callback(rocprofiler_callback_tracing_record_t record,
                    rocprofiler_user_data_t*              user_data,
                    void*                                 callback_data)
{
    (void) user_data;
    (void) callback_data;

    std::lock_guard<std::mutex> lock(trace_mutex);
    traces.push_back(api_trace_data_t{0,  // timestamp not available in callback tracing
                                      record.kind,
                                      record.operation,
                                      record.phase,
                                      get_operation_name(record.kind, record.operation)});
}

// Tool initialization callback
int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    (void) fini_func;
    (void) tool_data;

    // Create a profiling context
    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "create context");

    // Configure HIP API callback tracing
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         client_ctx,
                         ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                         nullptr,  // operations (nullptr = all operations)
                         0,        // operations count
                         hip_api_callback,
                         nullptr),  // callback data
                     "configure HIP API tracing");

    // Configure ROCTx/marker tracing
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         client_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                         nullptr,  // operations (nullptr = all operations)
                         0,        // operations count
                         marker_api_callback,
                         nullptr),  // callback data
                     "configure marker API tracing");

    // Start the context
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "start context");

    return 0;
}

void
write_json(const std::vector<api_trace_data_t>& _trace_data)
{
    auto ofname = std::string{"late-start-tracing.json"};
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

        constexpr auto json_prec   = 32;
        constexpr auto json_indent = JSONOutputArchive::Options::IndentChar::space;
        auto           json_opts   = JSONOutputArchive::Options{json_prec, json_indent, 1};
        auto           json_ar     = JSONOutputArchive{*ofs, json_opts};

        json_ar.setNextName("late-start-tracing");
        json_ar.startNode();
        try
        {
            json_ar(cereal::make_nvp("traces", _trace_data));

        } catch(std::exception& e)
        {
            std::cerr << "[" << getpid() << "][" << __FUNCTION__
                      << "] threw an exception: " << e.what() << "\n"
                      << std::flush;
        }
        json_ar.finishNode();
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);

    std::cout << "[late-start-tool] Wrote " << traces.size() << " traces to " << ofname
              << std::endl;
}

// Tool finalization callback - write JSON output
void
tool_fini(void* tool_data)
{
    (void) tool_data;

    write_json(traces);
}

// rocprofiler_configure function (in anonymous namespace)
rocprofiler_tool_configure_result_t*
configure(uint32_t                 version,
          const char*              runtime_version,
          uint32_t                 priority,
          rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;

    // Set client name
    client_id->name = "LateStartTool";

    // Allocate result structure (SDK will free this)
    auto* result =
        (rocprofiler_tool_configure_result_t*) malloc(sizeof(rocprofiler_tool_configure_result_t));

    result->size       = sizeof(rocprofiler_tool_configure_result_t);
    result->initialize = tool_init;  // Will be called to create context
    result->finalize   = tool_fini;  // Will be called on shutdown
    result->tool_data  = nullptr;

    return result;
}

}  // namespace

// Exported late_start_init() function - call this to trigger late-start
extern "C" {
void
late_start_init() ROCPROFILER_PUBLIC_API;

void
late_start_init()
{
    rocprofiler_status_t status = rocprofiler_force_configure(configure);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Error: rocprofiler_force_configure failed with status " << status
                  << std::endl;
    }
}
}
