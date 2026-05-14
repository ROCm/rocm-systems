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
 * @file tests/bin/code-object-multi-threaded/client.cpp
 *
 * @brief ROCProfiler tool that tracks code object operations
 *
 * This tool uses rocprofiler-sdk callback tracing to monitor code object
 * load/unload operations during multi-threaded execution, verifying that
 * the thread-safety fixes work correctly.
 */

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace
{
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

// Thread-safe counters
std::atomic<uint64_t> g_code_object_load_count{0};
std::atomic<uint64_t> g_code_object_unload_count{0};
std::atomic<uint64_t> g_kernel_symbol_load_count{0};
std::atomic<uint64_t> g_kernel_symbol_unload_count{0};

// Track which threads invoke callbacks
std::mutex                             g_thread_mutex;
std::unordered_map<uint64_t, uint64_t> g_thread_callback_counts;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};

void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /* user_data */,
                         void* /* callback_data */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;

    // Track thread activity
    {
        std::lock_guard<std::mutex> lock(g_thread_mutex);
        g_thread_callback_counts[record.thread_id]++;
    }

    if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            g_code_object_load_count.fetch_add(1, std::memory_order_relaxed);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            g_code_object_unload_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    else if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            g_kernel_symbol_load_count.fetch_add(1, std::memory_order_relaxed);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            g_kernel_symbol_unload_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void
tool_fini(void* /* tool_data */)
{
    std::cout << "\n=== ROCProfiler Multi-Threaded Code Object Test Results ===\n";
    std::cout << "Code objects loaded:        " << g_code_object_load_count.load() << "\n";
    std::cout << "Code objects unloaded:      " << g_code_object_unload_count.load() << "\n";
    std::cout << "Kernel symbols loaded:      " << g_kernel_symbol_load_count.load() << "\n";
    std::cout << "Kernel symbols unloaded:    " << g_kernel_symbol_unload_count.load() << "\n";

    size_t num_threads_with_callbacks = 0;
    {
        std::lock_guard<std::mutex> lock(g_thread_mutex);
        num_threads_with_callbacks = g_thread_callback_counts.size();
        std::cout << "Threads that invoked callbacks: " << num_threads_with_callbacks << "\n";
    }
    std::cout << "===========================================================\n";

    // Verify we actually traced something
    if(g_code_object_load_count.load() == 0)
    {
        std::cerr << "ERROR: No code objects were traced!\n";
        std::abort();
    }

    if(g_kernel_symbol_load_count.load() == 0)
    {
        std::cerr << "ERROR: No kernel symbols were traced!\n";
        std::abort();
    }

    // Verify multi-threaded execution
    if(num_threads_with_callbacks < 2)
    {
        std::cerr << "ERROR: Expected callbacks from multiple threads, got "
                  << num_threads_with_callbacks << "\n";
        std::abort();
    }

    std::cout << "Test PASSED: Successfully traced multi-threaded code object loading!\n";
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_fini_func = fini_func;

    auto status = rocprofiler_create_context(&client_ctx);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create context\n";
        return -1;
    }

    status =
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       codeobj_tracing_callback,
                                                       tool_data);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to configure code object tracing\n";
        return -1;
    }

    int valid_ctx = 0;
    status        = rocprofiler_context_is_valid(client_ctx, &valid_ctx);
    if(status != ROCPROFILER_STATUS_SUCCESS || valid_ctx == 0)
    {
        std::cerr << "Context is not valid\n";
        return -1;
    }

    status = rocprofiler_start_context(client_ctx);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to start context\n";
        return -1;
    }

    std::cout << "ROCProfiler multi-threaded code object tool initialized\n";
    return 0;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t /* priority */,
                      rocprofiler_client_id_t* id)
{
    id->name  = "CodeObjectMultiThreadedClient";
    client_id = id;

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}
