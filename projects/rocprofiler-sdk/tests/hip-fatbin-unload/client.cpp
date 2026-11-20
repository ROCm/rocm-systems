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
 * @file tests/hip-fatbin-unload/client.cpp
 *
 * @brief ROCProfiler tool for the HIP fat binary unload test
 *
 * Loading this tool loads rocprofiler-sdk into the test app, which then handles every HSA
 * executable freeze after the library unload. The tool subscribes to code object tracing only to
 * check that the test app's kernel symbol is still reported.
 */

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace
{
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

constexpr auto app_kernel_name = "fatbin_unload_app_kernel";

rocprofiler_context_id_t client_context  = {};
std::atomic<uint64_t>    app_kernel_seen = {0};

void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /* user_data */,
                         void* /* callback_data */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    const auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if(data->kernel_name && std::strstr(data->kernel_name, app_kernel_name) != nullptr)
        ++app_kernel_seen;
}

void
tool_fini(void* /* tool_data */)
{
    if(app_kernel_seen.load() == 0)
    {
        std::cerr << "ERROR: No kernel symbol was traced for " << app_kernel_name << "\n";
        return;
    }

    std::cout << "Traced kernel symbol for " << app_kernel_name << "\n";
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* tool_data)
{
    if(rocprofiler_create_context(&client_context) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create context\n";
        return -1;
    }

    if(rocprofiler_configure_callback_tracing_service(client_context,
                                                      ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                      nullptr,
                                                      0,
                                                      codeobj_tracing_callback,
                                                      tool_data) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to configure code object tracing\n";
        return -1;
    }

    int valid_ctx = 0;
    if(rocprofiler_context_is_valid(client_context, &valid_ctx) != ROCPROFILER_STATUS_SUCCESS ||
       valid_ctx == 0)
    {
        std::cerr << "Context is not valid\n";
        return -1;
    }

    if(rocprofiler_start_context(client_context) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Failed to start context\n";
        return -1;
    }

    return 0;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t /* priority */,
                      rocprofiler_client_id_t* id)
{
    id->name = "HipFatbinUnloadClient";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}
