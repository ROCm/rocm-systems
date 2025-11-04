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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/process_attachment/tool.cpp
 *
 * @brief Process attachment tool implementation - can be dynamically loaded into running processes
 */

#include "tool.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/version.hpp>

#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/name_info.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <ratio>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <atomic>

namespace attachment_tool
{
namespace
{
using common::callback_name_info;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {0};

// Statistics tracking
std::atomic<uint64_t> kernel_dispatch_count{0};
std::atomic<uint64_t> hip_api_call_count{0};
std::atomic<uint64_t> memory_copy_count{0};

// Control flags
std::atomic<bool> kernel_tracing_enabled{true};
std::atomic<bool> hip_api_tracing_enabled{true};

std::mutex output_mutex;
std::ofstream output_file;


void
kernel_dispatch_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t*              user_data,
                         void*                                 callback_data_args)
{
    if(!kernel_tracing_enabled.load()) return;

    (void) user_data;
    (void) callback_data_args;

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH)
    {
        kernel_dispatch_count.fetch_add(1);
        
        std::lock_guard<std::mutex> lock(output_mutex);
        if(output_file.is_open())
        {
            auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            
            output_file << "[KERNEL] " << ts << " ns: ";
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                output_file << "ENTER ";
            else
                output_file << "EXIT  ";
            
            // For kernel dispatch callbacks, payload contains dispatch info
            auto* dispatch_info = static_cast<rocprofiler_kernel_dispatch_info_t*>(record.payload);
            if(dispatch_info)
            {
                output_file << "kernel_dispatch_" << dispatch_info->dispatch_id;
                output_file << "agent_id: " << dispatch_info->agent_id.handle << " ";
                output_file << "queue_id: " << dispatch_info->queue_id.handle << " ";   
                output_file << "kernel_id: " << dispatch_info->kernel_id << " ";

            }
            else
            {
                output_file << "<unknown_kernel>";
            }
            
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            {
                dispatch_info = static_cast<rocprofiler_kernel_dispatch_info_t*>(record.payload);
                if(dispatch_info)
                {
                    output_file << " [grid: " 
                              << dispatch_info->grid_size.x << "x"
                              << dispatch_info->grid_size.y << "x"
                              << dispatch_info->grid_size.z
                              << ", workgroup: "
                              << dispatch_info->workgroup_size.x << "x"
                              << dispatch_info->workgroup_size.y << "x"
                              << dispatch_info->workgroup_size.z << "]";
                }
            }
            output_file << std::endl;
        }
    }
}

void
hip_api_callback(rocprofiler_callback_tracing_record_t record,
                 rocprofiler_user_data_t*              user_data,
                 void*                                 callback_data_args)
{
    if(!hip_api_tracing_enabled.load()) return;

    (void) user_data;
    (void) callback_data_args;

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
    {
        hip_api_call_count.fetch_add(1);
        
        std::lock_guard<std::mutex> lock(output_mutex);
        if(output_file.is_open())
        {
            auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            
            output_file << "[HIP_API] " << ts << " ns: ";
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                output_file << "ENTER ";
            else
                output_file << "EXIT  ";
            
            // Get API function name
            const char* name = nullptr;
            if(record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API)
            {
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                    record.kind, record.operation, &name, nullptr),
                    "failed to get HIP runtime API name");
            }
            else if(record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API)
            {
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                    record.kind, record.operation, &name, nullptr),
                    "failed to get HIP compiler API name");
            }
            
            output_file << (name ? name : "<unknown_api>") << std::endl;
        }
    }
}

void
memory_copy_callback(rocprofiler_callback_tracing_record_t record,
                     rocprofiler_user_data_t*              user_data,
                     void*                                 callback_data_args)
{
    (void) user_data;
    (void) callback_data_args;

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY)
    {
        memory_copy_count.fetch_add(1);
        
        std::lock_guard<std::mutex> lock(output_mutex);
        if(output_file.is_open())
        {
            auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            
            output_file << "[MEMORY] " << ts << " ns: ";
            if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
                output_file << "ENTER ";
            else
                output_file << "EXIT  ";
            
            // For memory copy callbacks, payload contains memory copy info
            // Note: For basic functionality, we'll just log the operation type
            output_file << "Memory copy operation: " << record.operation << std::endl;
        }
    }
}

}  // namespace

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    (void) tool_data;
    
    client_fini_func = fini_func;
    
    // Open output file
    std::string output_filename = "process_attachment_trace_" + std::to_string(getpid()) + ".log";
    output_file.open(output_filename, std::ios::out | std::ios::trunc);
    
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if(output_file.is_open())
        {
            output_file << "=== Process Attachment Tool Started ===" << std::endl;
            output_file << "PID: " << getpid() << std::endl;
            output_file << "========================================" << std::endl;
        }
        else
        {
            std::cerr << "Failed to open output file: " << output_filename << std::endl;
        }
    }

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "failed to create context");

    // Register for kernel dispatch callbacks
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         client_ctx,
                         ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                         nullptr,
                         0,
                         kernel_dispatch_callback,
                         nullptr),
                     "failed to configure kernel dispatch tracing service");

    // Register for HIP API callbacks
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         client_ctx,
                         ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                         nullptr,
                         0,
                         hip_api_callback,
                         nullptr),
                     "failed to configure HIP runtime API tracing service");

    // Register for memory copy callbacks
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         client_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                         nullptr,
                         0,
                         memory_copy_callback,
                         nullptr),
                     "failed to configure memory copy tracing service");

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "failed to start context");

    std::cout << "[ATTACHMENT_TOOL] Process attachment tool initialized successfully!" << std::endl;
    std::cout << "[ATTACHMENT_TOOL] Output file: " << output_filename << std::endl;
    
    return 0;
}

void
tool_fini(void* tool_data)
{
    (void) tool_data;
    
    std::cout << "[ATTACHMENT_TOOL] Process attachment tool finalizing..." << std::endl;
    
    // Print final statistics
    get_statistics();
    
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if(output_file.is_open())
        {
            output_file << "=== Process Attachment Tool Finished ===" << std::endl;
            output_file << "Final Statistics:" << std::endl;
            output_file << "  Kernel dispatches: " << kernel_dispatch_count.load() << std::endl;
            output_file << "  HIP API calls: " << hip_api_call_count.load() << std::endl;
            output_file << "  Memory copies: " << memory_copy_count.load() << std::endl;
            output_file << "=========================================" << std::endl;
            output_file.close();
        }
    }

    if(client_ctx.handle > 0)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "failed to stop context");
    }
    
    std::cout << "[ATTACHMENT_TOOL] Process attachment tool finalized!" << std::endl;
}

void
get_statistics()
{
    std::cout << "[ATTACHMENT_TOOL] === Statistics ===" << std::endl;
    std::cout << "[ATTACHMENT_TOOL] Kernel dispatches: " << kernel_dispatch_count.load() << std::endl;
    std::cout << "[ATTACHMENT_TOOL] HIP API calls: " << hip_api_call_count.load() << std::endl;
    std::cout << "[ATTACHMENT_TOOL] Memory copies: " << memory_copy_count.load() << std::endl;
    std::cout << "[ATTACHMENT_TOOL] ==================" << std::endl;
}

void
set_kernel_tracing(bool enable)
{
    kernel_tracing_enabled.store(enable);
    std::cout << "[ATTACHMENT_TOOL] Kernel tracing " << (enable ? "enabled" : "disabled") << std::endl;
}

void
set_hip_api_tracing(bool enable)
{
    hip_api_tracing_enabled.store(enable);
    std::cout << "[ATTACHMENT_TOOL] HIP API tracing " << (enable ? "enabled" : "disabled") << std::endl;
}

int
tool_attach(rocprofiler_client_detach_t /*detach_func*/,
           rocprofiler_context_id_t*   context_ids,
           uint64_t                    context_ids_length,
           void*                       /*tool_data*/)
{
    std::cout << "[ATTACHMENT_TOOL] Tool attach called with " << context_ids_length << " contexts" << std::endl;
    
    // Start the provided contexts (these are contexts from the target process)
    for(uint64_t i = 0; i < context_ids_length; ++i)
    {
        if(context_ids[i].handle != 0)
        {
            std::cout << "[ATTACHMENT_TOOL] Starting context " << i << " (handle: " << context_ids[i].handle << ")" << std::endl;
            ROCPROFILER_CALL(rocprofiler_start_context(context_ids[i]), "failed to start attached context");
        }
    }
    
    std::cout << "[ATTACHMENT_TOOL] Process attachment completed successfully!" << std::endl;
    return 0;
}

void
tool_detach(void* /*tool_data*/)
{
    std::cout << "[ATTACHMENT_TOOL] Tool detach called" << std::endl;
    
    // Print final statistics before detaching
    get_statistics();
    
    // Stop our context if it's running
    if(client_ctx.handle > 0)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "failed to stop context during detach");
    }
    
    // Close output file during detach
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if(output_file.is_open())
        {
            output_file << "=== Process Attachment Tool Detached ===" << std::endl;
            output_file << "Final Statistics at Detach:" << std::endl;
            output_file << "  Kernel dispatches: " << kernel_dispatch_count.load() << std::endl;
            output_file << "  HIP API calls: " << hip_api_call_count.load() << std::endl;
            output_file << "  Memory copies: " << memory_copy_count.load() << std::endl;
            output_file << "========================================" << std::endl;
            output_file.close();
        }
    }
    
    std::cout << "[ATTACHMENT_TOOL] Process detachment completed successfully!" << std::endl;
}

}  // namespace attachment_tool


extern "C" {

/**
 * @brief ROCprofiler-SDK tool configuration function
 * This is called when the tool is loaded normally (not during attach)
 */
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority, 
                      rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;
    
    attachment_tool::client_id = client_id;
    client_id->name = "process-attachment-tool";
    
    // Create configure result with normal init/fini functions
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), 
        &attachment_tool::tool_init, 
        &attachment_tool::tool_fini, 
        nullptr};
    
    return &cfg;
}

/**
 * @brief ROCprofiler-SDK tool attach configuration function
 * This is called when the tool is loaded via process attachment
 */
rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t /*version*/,
                             const char* /*runtime_version*/,
                             uint32_t /*priority*/,
                             rocprofiler_client_id_t* /*client_id*/)
{
    // This function is called right after rocprofiler_configure with the same parameters.
    // The data returned is only used when attaching to a running process.
    
    // Create configure result with attach/detach functions
    static auto cfg = rocprofiler_tool_configure_attach_result_t{
        sizeof(rocprofiler_tool_configure_attach_result_t),
        &attachment_tool::tool_attach,
        &attachment_tool::tool_detach,
        nullptr};
    
    return &cfg;
}

}  // extern "C"
