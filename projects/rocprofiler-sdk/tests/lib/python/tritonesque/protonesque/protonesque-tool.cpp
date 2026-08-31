// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

//
// protonesque-tool.cpp -- A rocprofiler-sdk tool that captures HIP API traces
// and GPU execution timing. Designed to be loaded via ctypes from Python and
// controlled via exported start/stop functions. Supports concurrent operation
// with other tool libraries (e.g. kinetoesque).
//

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t status_v = (result);                                                  \
        if(status_v != ROCPROFILER_STATUS_SUCCESS)                                                 \
        {                                                                                          \
            const char* status_msg = rocprofiler_get_status_string(status_v);                      \
            fprintf(stderr,                                                                        \
                    "[protonesque] %s failed with error: %s\n",                                    \
                    msg,                                                                           \
                    (status_msg) ? status_msg : "unknown");                                        \
            return -1;                                                                             \
        }                                                                                          \
    }

namespace
{
struct trace_record_t
{
    uint64_t    start_timestamp = 0;
    uint64_t    end_timestamp   = 0;
    int32_t     kind            = 0;
    int32_t     operation       = 0;
    int32_t     phase           = 0;
    std::string name            = {};
};

std::mutex               trace_mutex   = {};
auto                     traces        = new std::vector<trace_record_t>{};
rocprofiler_context_id_t client_ctx    = {.handle = 0};
rocprofiler_client_id_t* client_id_ptr = nullptr;
std::atomic<bool>        is_active{false};
std::atomic<bool>        is_initialized{false};
std::string              output_filename = "protonesque-trace.json";

std::string
get_operation_name(rocprofiler_callback_tracing_kind_t kind, int32_t operation)
{
    const char* name = nullptr;
    rocprofiler_query_callback_tracing_kind_operation_name(kind, operation, &name, nullptr);
    return (name) ? name : "<unknown>";
}

void
hip_api_callback(rocprofiler_callback_tracing_record_t record,
                 rocprofiler_user_data_t*              user_data,
                 void*                                 callback_data)
{
    (void) user_data;
    (void) callback_data;

    if(!is_active.load(std::memory_order_relaxed)) return;

    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();

    std::lock_guard<std::mutex> lock(trace_mutex);
    if(!traces) return;
    traces->push_back(trace_record_t{
        (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) ? static_cast<uint64_t>(ts) : 0,
        (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) ? static_cast<uint64_t>(ts) : 0,
        record.kind,
        record.operation,
        record.phase,
        get_operation_name(record.kind, record.operation)});
}

void
write_json()
{
    std::lock_guard<std::mutex> lock(trace_mutex);

    auto ofname = output_filename;
    if(auto* env = getenv("PROTONESQUE_OUTPUT_FILE")) ofname = env;

    auto ofs = std::ofstream{ofname};
    if(!ofs)
    {
        fprintf(stderr, "[protonesque] Failed to open output file: %s\n", ofname.c_str());
        return;
    }

    if(!traces) return;

    fprintf(stderr,
            "[protonesque][%d] Writing %zu traces to %s\n",
            getpid(),
            traces->size(),
            ofname.c_str());

    ofs << "{\n  \"protonesque-traces\": [\n";
    for(size_t i = 0; i < traces->size(); ++i)
    {
        auto& t = (*traces)[i];
        ofs << "    {";
        ofs << "\"start_timestamp\": " << t.start_timestamp;
        ofs << ", \"end_timestamp\": " << t.end_timestamp;
        ofs << ", \"kind\": " << t.kind;
        ofs << ", \"operation\": " << t.operation;
        ofs << ", \"phase\": " << t.phase;
        ofs << ", \"name\": \"" << t.name << "\"";
        ofs << "}";
        if(i + 1 < traces->size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n}\n";
    ofs.flush();
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    (void) fini_func;
    (void) tool_data;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "create context");

    // Configure HIP runtime API callback tracing
    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                       nullptr,
                                                       0,
                                                       hip_api_callback,
                                                       nullptr),
        "configure HIP runtime API tracing");

    // Configure HIP compiler API callback tracing
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         client_ctx,
                         ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
                         nullptr,
                         0,
                         hip_api_callback,
                         nullptr),
                     "configure HIP compiler API tracing");

    // Start the context (but not actively collecting yet)
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "start context");

    is_initialized.store(true, std::memory_order_release);

    fprintf(stderr, "[protonesque] Tool initialized (context started)\n");
    return 0;
}

void
tool_fini(void* tool_data)
{
    (void) tool_data;

    is_active.store(false, std::memory_order_release);

    if(traces)
    {
        write_json();
        delete traces;
        traces = nullptr;
    }

    fprintf(stderr, "[protonesque] Tool finalized\n");
}

rocprofiler_tool_configure_result_t*
configure(uint32_t                 version,
          const char*              runtime_version,
          uint32_t                 priority,
          rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;

    client_id->name = "protonesque";
    client_id_ptr   = client_id;

    static auto* result = new rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, nullptr};

    return result;
}
}  // namespace

extern "C" {

/// Initialize protonesque: triggers rocprofiler_force_configure to register as a tool.
/// Returns 0 on success, non-zero on error.
ROCPROFILER_PUBLIC_API int
protonesque_init()
{
    if(is_initialized.load(std::memory_order_acquire))
    {
        fprintf(stderr, "[protonesque] Already initialized\n");
        return 0;
    }

    fprintf(stderr, "[protonesque] Initializing via force_configure...\n");

    rocprofiler_status_t status = rocprofiler_force_configure(configure);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        const char* status_msg = rocprofiler_get_status_string(status);
        fprintf(stderr,
                "[protonesque] rocprofiler_force_configure failed: %s\n",
                (status_msg) ? status_msg : "unknown");
        return -1;
    }

    return 0;
}

/// Start active collection of trace data.
ROCPROFILER_PUBLIC_API int
protonesque_start()
{
    if(!is_initialized.load(std::memory_order_acquire))
    {
        fprintf(stderr, "[protonesque] Not initialized, cannot start\n");
        return -1;
    }

    is_active.store(true, std::memory_order_release);
    fprintf(stderr, "[protonesque] Profiling started\n");
    return 0;
}

/// Stop active collection of trace data.
ROCPROFILER_PUBLIC_API int
protonesque_stop()
{
    is_active.store(false, std::memory_order_release);
    fprintf(stderr, "[protonesque] Profiling stopped\n");
    return 0;
}

/// Finalize and write output. After this, the tool cannot be restarted.
ROCPROFILER_PUBLIC_API int
protonesque_finalize()
{
    is_active.store(false, std::memory_order_release);
    write_json();
    fprintf(stderr, "[protonesque] Finalized (wrote output)\n");
    return 0;
}

/// Set the output filename for the JSON trace.
ROCPROFILER_PUBLIC_API void
protonesque_set_output_file(const char* filename)
{
    if(filename) output_filename = filename;
}

/// Get the number of trace records collected so far.
ROCPROFILER_PUBLIC_API int
protonesque_get_trace_count()
{
    std::lock_guard<std::mutex> lock(trace_mutex);
    return (traces) ? static_cast<int>(traces->size()) : 0;
}
}  // extern "C"
