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
 * @file tests/hip-buffered-dangling-arg/client.cpp
 *
 * @brief rocprofiler tool that buffers HIP API records and iterates their
 * arguments on flush.
 *
 * The `*_EXT` buffered HIP API domains store each call's arguments into the
 * buffer record. rocprofiler_iterate_buffer_tracing_record_args() converts each
 * argument to a string; for a const char* that means dereferencing the stored
 * pointer. Because the buffer is flushed on the rocprofiler callback thread
 * after the traced call returned, any const char* argument that was stored as
 * the caller's raw pointer (rather than interned) is dangling at this point.
 *
 * This tool therefore turns the fix's contract into an observable outcome: with
 * the argument interned, iteration succeeds and we report a PASS; without it,
 * the process segfaults during flush (caught by the ctest FAIL regex).
 */

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace
{
std::atomic<uint64_t> g_hip_records{0};
std::atomic<uint64_t> g_args_stringified{0};
std::atomic<uint64_t> g_busid_args_seen{0};

rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};
rocprofiler_buffer_id_t       client_buffer    = {};

// Invoked for each argument of a buffered HIP API record. Touching
// arg_value_str forces rocprofiler to have stringified the argument, which for
// the const char* kernel-name argument means dereferencing the stored pointer.
int
hip_api_args_callback(rocprofiler_buffer_tracing_kind_t /* kind */,
                      rocprofiler_tracing_operation_t /* operation */,
                      uint32_t /* arg_number */,
                      const void* const /* arg_value_addr */,
                      int32_t /* arg_indirection_count */,
                      const char* /* arg_type */,
                      const char* arg_name,
                      const char* arg_value_str,
                      void* /* data */)
{
    if(arg_value_str != nullptr)
    {
        // Read the whole string so the dereference is not optimized away.
        volatile size_t len = std::strlen(arg_value_str);
        (void) len;
        g_args_stringified.fetch_add(1, std::memory_order_relaxed);

        // The PCI bus-id handed to hipDeviceGetByPCIBusId is the argument this
        // test exists to exercise; count it so we can assert we actually hit it.
        if(arg_name != nullptr && std::strcmp(arg_name, "pciBusId") == 0)
        {
            g_busid_args_seen.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return 0;
}

// Buffer flush callback (delivered on the rocprofiler callback thread, after the
// traced HIP calls returned and their stack/temporary strings were destroyed).
void
hip_buffered_callback(rocprofiler_context_id_t /* context */,
                      rocprofiler_buffer_id_t /* buffer_id */,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void* /* data */,
                      uint64_t /* drop_count */)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header == nullptr) continue;

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           (header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT ||
            header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT))
        {
            g_hip_records.fetch_add(1, std::memory_order_relaxed);
            rocprofiler_iterate_buffer_tracing_record_args(*header, hip_api_args_callback, nullptr);
        }
    }
}

void
tool_fini(void* /* tool_data */)
{
    // Ensure everything is stringified before we report results.
    auto flush_status = rocprofiler_flush_buffer(client_buffer);
    if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY &&
       flush_status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: buffer flush failed at finalize\n";
        std::abort();
    }

    std::cout << "\n=== HIP Buffered Dangling-Arg Test Results ===\n";
    std::cout << "HIP API records processed:  " << g_hip_records.load() << "\n";
    std::cout << "Arguments stringified:      " << g_args_stringified.load() << "\n";
    std::cout << "pciBusId arguments seen:    " << g_busid_args_seen.load() << "\n";
    std::cout << "==============================================\n";

    if(g_hip_records.load() == 0)
    {
        std::cerr << "ERROR: No buffered HIP API records were traced!\n";
        std::abort();
    }

    if(g_busid_args_seen.load() == 0)
    {
        std::cerr << "ERROR: hipDeviceGetByPCIBusId pciBusId argument was never traced!\n";
        std::abort();
    }

    std::cout << "Test PASSED: buffered HIP API string arguments stringified safely!\n";
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_fini_func = fini_func;

    if(rocprofiler_create_context(&client_ctx) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create context\n";
        return -1;
    }

    constexpr auto buffer_size_bytes      = 8192;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

    if(rocprofiler_create_buffer(client_ctx,
                                 buffer_size_bytes,
                                 buffer_watermark_bytes,
                                 ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                 hip_buffered_callback,
                                 tool_data,
                                 &client_buffer) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create buffer\n";
        return -1;
    }

    for(auto kind : {ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
                     ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT})
    {
        if(rocprofiler_configure_buffer_tracing_service(
               client_ctx, kind, nullptr, 0, client_buffer) != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "Failed to configure HIP API buffer tracing\n";
            return -1;
        }
    }

    auto client_thread = rocprofiler_callback_thread_t{};
    if(rocprofiler_create_callback_thread(&client_thread) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create callback thread\n";
        return -1;
    }

    if(rocprofiler_assign_callback_thread(client_buffer, client_thread) !=
       ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to assign callback thread\n";
        return -1;
    }

    int valid_ctx = 0;
    if(rocprofiler_context_is_valid(client_ctx, &valid_ctx) != ROCPROFILER_STATUS_SUCCESS ||
       valid_ctx == 0)
    {
        std::cerr << "Context is not valid\n";
        return -1;
    }

    if(rocprofiler_start_context(client_ctx) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to start context\n";
        return -1;
    }

    std::cout << "HIP buffered dangling-arg tool initialized\n";
    return 0;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t /* priority */,
                      rocprofiler_client_id_t* id)
{
    id->name = "HIPBufferedDanglingArgClient";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}
