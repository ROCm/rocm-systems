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

#include "client.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t status_ = (result);                                                   \
        if(status_ != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << " :: " << (msg) << " failed with "         \
                      << rocprofiler_get_status_string(status_) << "\n";                           \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    }

namespace client
{
namespace
{
// Matches SKIP_RETURN_CODE on the ctest entry.
constexpr int etw_permission_skip_code = 77;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};
rocprofiler_buffer_id_t       client_buffer    = {};
rocprofiler_thread_id_t       client_tid       = 0;

// Guards both containers: the buffer callback runs on a rocprofiler callback thread while
// expect_operation() runs on the application thread.
std::mutex               record_mutex       = {};
std::vector<std::string> observed_names     = {};
std::set<std::string>    expected_names     = {};
uint64_t                 record_count       = 0;
uint64_t                 setup_timestamp_ns = 0;

std::string
operation_name(rocprofiler_tracing_operation_t operation)
{
    const char* name     = nullptr;
    uint64_t    name_len = 0;

    ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                         ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, operation, &name, &name_len),
                     "operation name query");

    return (name) ? std::string{name, name_len} : std::string{};
}

void
tool_tracing_callback(rocprofiler_context_id_t /*context*/,
                      rocprofiler_buffer_id_t /*buffer_id*/,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void* /*user_data*/,
                      uint64_t /*drop_count*/)
{
    auto lk = std::unique_lock<std::mutex>{record_mutex};

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
           header->kind != ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            std::cerr << "unexpected record category " << header->category << " / kind "
                      << header->kind << "\n";
            std::exit(EXIT_FAILURE);
        }

        auto* record = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

        if(record->end_timestamp < record->start_timestamp ||
           record->start_timestamp < setup_timestamp_ns)
        {
            std::cerr << "implausible timestamps on " << operation_name(record->operation) << ": ["
                      << record->start_timestamp << ", " << record->end_timestamp << "]\n";
            std::exit(EXIT_FAILURE);
        }

        if(record->thread_id != client_tid)
        {
            std::cerr << "record thread id " << record->thread_id << " does not match the calling "
                      << "thread " << client_tid << "\n";
            std::exit(EXIT_FAILURE);
        }

        observed_names.emplace_back(operation_name(record->operation));
        ++record_count;
    }
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* /*tool_data*/)
{
    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_get_timestamp(&setup_timestamp_ns), "timestamp query");
    ROCPROFILER_CALL(rocprofiler_get_thread_id(&client_tid), "thread id query");
    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    constexpr auto buffer_size_bytes      = 8192;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

    ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               buffer_size_bytes,
                                               buffer_watermark_bytes,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               nullptr,
                                               &client_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, client_buffer),
        "buffer tracing service configure");

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "creating callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(client_buffer, client_thread),
                     "assignment of thread for buffer");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "context validity check");
    if(valid_ctx == 0) return -1;

    // Opening the ETW consumer session happens here, so an under-privileged run reports the
    // missing group membership instead of silently tracing nothing.
    auto status = rocprofiler_start_context(client_ctx);
    if(status == ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED)
    {
        std::cerr << "skipping: creating an ETW session requires membership in the "
                     "'Performance Log Users' group\n";
        std::exit(etw_permission_skip_code);
    }
    ROCPROFILER_CALL(status, "context start");

    return 0;
}

void
tool_fini(void* /*tool_data*/)
{
    auto lk = std::unique_lock<std::mutex>{record_mutex};

    std::cout << "[hip-api-etw-tracing] collected " << record_count << " HIP runtime API records\n";

    if(record_count == 0)
    {
        std::cerr << "no HIP runtime API records were collected; the rocm_hip_tlg provider is "
                     "only present in an amdhip64 built with -DHIP_TRACE_BACKEND=tracelogging\n";
        std::exit(EXIT_FAILURE);
    }

    auto observed = std::set<std::string>{observed_names.begin(), observed_names.end()};

    // The provider numbers operations with clr's HIP_API_ID_*, which is unrelated to
    // ROCPROFILER_HIP_RUNTIME_API_ID_*. Recovering the exact function name the application
    // called is what proves the op_name string bridged the two.
    for(const auto& itr : expected_names)
    {
        if(observed.count(itr) == 0)
        {
            std::cerr << "no record resolved back to '" << itr << "'\n";
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << "[hip-api-etw-tracing] resolved " << expected_names.size()
              << " expected operations out of " << observed.size() << " distinct operations\n";
}
}  // namespace

void
setup()
{
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure), "force configure");
    }
}

void
shutdown()
{
    if(client_id)
    {
        // Stop first: ETW delivery is asynchronous, and closing the session is what flushes
        // the provider's buffers and drains the decoding thread. Flushing before that would
        // race the tail of the trace.
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "context stop");
        ROCPROFILER_CALL(rocprofiler_flush_buffer(client_buffer), "buffer flush");
        client_fini_func(*client_id);
    }
}

void
expect_operation(const char* name)
{
    auto lk = std::unique_lock<std::mutex>{record_mutex};
    expected_names.emplace(name);
}
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version,
                      const char* /*runtime_version*/,
                      uint32_t /*priority*/,
                      rocprofiler_client_id_t* id)
{
    id->name          = "HipApiEtwTracing";
    client::client_id = id;

    std::clog << id->name << " is using rocprofiler-sdk v" << (version / 10000) << "."
              << ((version % 10000) / 100) << "." << (version % 100) << "\n";

    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            nullptr};

    return &cfg;
}
