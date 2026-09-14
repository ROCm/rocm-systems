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
 * @file tests/code-object-host-symbols/client.cpp
 *
 * @brief ROCProfiler tool that checks host function id assignment
 *
 * A host function id identifies a symbol, not a notification of that symbol, so
 * every subscribed context must be handed the same id for a given symbol, and the
 * ids must stay unique. This tool subscribes two contexts to code object tracing
 * and verifies both properties.
 */

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <set>

namespace
{
using host_symbol_data_t =
    rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t;

// kernel id -> host function id, as observed by one context
using host_symbol_map_t = std::map<uint64_t, uint64_t>;

constexpr size_t num_contexts = 2;

std::mutex                                         symbols_mutex;
std::map<uint64_t, host_symbol_map_t>              symbols_by_context;
std::array<rocprofiler_context_id_t, num_contexts> client_contexts = {};

void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /* user_data */,
                         void* /* callback_data */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    const auto* data = static_cast<host_symbol_data_t*>(record.payload);

    auto  lock     = std::lock_guard<std::mutex>{symbols_mutex};
    auto& symbols  = symbols_by_context[record.context_id.handle];
    auto  existing = symbols.emplace(data->kernel_id, data->host_function_id);
    if(!existing.second && existing.first->second != data->host_function_id)
    {
        std::cerr << "ERROR: Kernel " << data->kernel_id << " was registered with host function id "
                  << existing.first->second << " and again with " << data->host_function_id
                  << " in the same context\n";
        std::abort();
    }
}

void
tool_fini(void* /* tool_data */)
{
    auto lock = std::lock_guard<std::mutex>{symbols_mutex};

    std::cout << "\n=== ROCProfiler Host Symbol Id Test Results ===\n";
    std::cout << "Contexts that observed host symbols: " << symbols_by_context.size() << "\n";

    if(symbols_by_context.size() != num_contexts)
    {
        std::cerr << "ERROR: Expected " << num_contexts << " contexts to observe host symbols, got "
                  << symbols_by_context.size() << "\n";
        std::abort();
    }

    const auto& reference = symbols_by_context.begin()->second;
    std::cout << "Host symbols per context:            " << reference.size() << "\n";

    if(reference.empty())
    {
        std::cerr << "ERROR: No host symbols were traced!\n";
        std::abort();
    }

    // every context must have been handed the same id for a given symbol
    for(const auto& [context_handle, symbols] : symbols_by_context)
    {
        if(symbols != reference)
        {
            std::cerr << "ERROR: Context " << context_handle
                      << " observed different host function ids than the first context. A host "
                         "function id must identify the symbol, not the notification.\n";
            std::abort();
        }
    }

    // ids must be unique per symbol; the public header promises uniqueness, not density, so
    // the largest id is only required to be at least the number of symbols
    auto unique_ids = std::set<uint64_t>{};
    auto largest_id = uint64_t{0};
    for(const auto& [kernel_id, host_function_id] : reference)
    {
        unique_ids.emplace(host_function_id);
        largest_id = std::max(largest_id, host_function_id);
    }

    std::cout << "Distinct host function ids:          " << unique_ids.size() << "\n";
    std::cout << "Largest host function id:            " << largest_id << "\n";
    std::cout << "===============================================\n";

    if(unique_ids.size() != reference.size())
    {
        std::cerr << "ERROR: Expected " << reference.size() << " distinct host function ids, got "
                  << unique_ids.size() << "\n";
        std::abort();
    }

    if(largest_id < reference.size())
    {
        std::cerr << "ERROR: Largest host function id is " << largest_id << " for "
                  << reference.size() << " symbols, so ids are not unique\n";
        std::abort();
    }

    std::cout << "Test PASSED: Successfully verified host function id assignment!\n";
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* tool_data)
{
    for(auto& context : client_contexts)
    {
        if(rocprofiler_create_context(&context) != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "Failed to create context\n";
            return -1;
        }

        if(rocprofiler_configure_callback_tracing_service(context,
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
        if(rocprofiler_context_is_valid(context, &valid_ctx) != ROCPROFILER_STATUS_SUCCESS ||
           valid_ctx == 0)
        {
            std::cerr << "Context is not valid\n";
            return -1;
        }

        if(rocprofiler_start_context(context) != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "Failed to start context\n";
            return -1;
        }
    }

    std::cout << "ROCProfiler host symbol id tool initialized with " << num_contexts
              << " contexts\n";
    return 0;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t /* priority */,
                      rocprofiler_client_id_t* id)
{
    id->name = "CodeObjectHostSymbolsClient";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}
