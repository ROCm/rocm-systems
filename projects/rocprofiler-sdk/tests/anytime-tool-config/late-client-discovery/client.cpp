// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file tests/anytime-tool-config/late-client-discovery/client.cpp
 *
 * @brief rocprofiler-sdk tool which counts kernel dispatches for the late client
 * discovery test.
 *
 * This library exports rocprofiler_configure and never calls
 * rocprofiler_force_configure(). The test app dlopens it and does nothing else to
 * announce it, so whether it is configured at all depends entirely on rocprofiler-sdk
 * looking for clients again after the point it was loaded.
 */

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace
{
constexpr auto buffer_size_bytes      = 4096;
constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

rocprofiler_context_id_t client_context = {};
rocprofiler_buffer_id_t  client_buffer  = {};
std::atomic<uint64_t>    dispatch_count = {0};

void
buffered_tracing_callback(rocprofiler_context_id_t /* context */,
                          rocprofiler_buffer_id_t /* buffer */,
                          rocprofiler_record_header_t** headers,
                          size_t                        num_headers,
                          void* /* user_data */,
                          uint64_t /* drop_count */)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        const auto* header = headers[i];
        if(header != nullptr && header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            ++dispatch_count;
    }
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* tool_data)
{
    if(rocprofiler_create_context(&client_context) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: failed to create context\n";
        return -1;
    }

    if(rocprofiler_create_buffer(client_context,
                                 buffer_size_bytes,
                                 buffer_watermark_bytes,
                                 ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                 buffered_tracing_callback,
                                 tool_data,
                                 &client_buffer) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: failed to create buffer\n";
        return -1;
    }

    if(rocprofiler_configure_buffer_tracing_service(
           client_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer) !=
       ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: failed to configure kernel dispatch tracing\n";
        return -1;
    }

    auto client_thread = rocprofiler_callback_thread_t{};
    if(rocprofiler_create_callback_thread(&client_thread) != ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_assign_callback_thread(client_buffer, client_thread) !=
           ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: failed to set up the buffer's callback thread\n";
        return -1;
    }

    int valid_ctx = 0;
    if(rocprofiler_context_is_valid(client_context, &valid_ctx) != ROCPROFILER_STATUS_SUCCESS ||
       valid_ctx == 0)
    {
        std::cerr << "ERROR: context is not valid\n";
        return -1;
    }

    if(rocprofiler_start_context(client_context) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: failed to start context\n";
        return -1;
    }

    std::cout << "Tool configured and collecting kernel dispatches\n" << std::flush;
    return 0;
}

void
tool_fini(void* /* tool_data */)
{}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t /* priority */,
                      rocprofiler_client_id_t* id)
{
    id->name = "LateClientDiscoveryClient";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}

// Exported with default visibility on purpose: the tests build compiles with hidden
// visibility, and the test app resolves this through dlsym on its own dlopen handle.
// Flushes first so the count covers every dispatch the runtime has already produced.
extern "C" __attribute__((visibility("default"))) uint64_t
late_client_discovery_dispatch_count()
{
    if(client_buffer.handle != 0) rocprofiler_flush_buffer(client_buffer);
    return dispatch_count.load();
}
