// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ptrace_session.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"

#include <rocprofiler-sdk/defines.h>

#include <atomic>
#include <thread>

extern char** environ;

namespace common = ::rocprofiler::common;

namespace
{
std::unique_ptr<rocprofiler::attach::PTraceSession> ptrace_session;
std::thread                                         ptrace_thread;
std::atomic<bool>                                   finished_setup(false);
}  // namespace

ROCPROFILER_EXTERN_C_INIT
int
attach(uint32_t pid) ROCPROFILER_EXPORT;

int
detach() ROCPROFILER_EXPORT;
ROCPROFILER_EXTERN_C_FINI

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

namespace
{
// Helper function to allocate memory in target process and write data
bool
write_data_to_target(const std::string&          description,
                     const std::vector<uint8_t>& data,
                     void*&                      allocated_addr)
{
    // Allocate memory in target process
    if(!ptrace_session->simple_mmap(allocated_addr, data.size()))
    {
        ROCP_ERROR << "Failed to allocate memory for " << description << " in target process";
        return false;
    }
    ROCP_TRACE << "Allocated memory for " << description << " at " << allocated_addr;

    // Stop target process for writing
    if(!ptrace_session->stop())
    {
        ROCP_ERROR << "Failed to stop target process for " << description << " writing";
        return false;
    }

    // Write data to target process memory
    if(!ptrace_session->write(reinterpret_cast<size_t>(allocated_addr), data, data.size()))
    {
        ROCP_ERROR << "Failed to write " << description << " to target process";
        return false;
    }

    // Continue target process
    if(!ptrace_session->cont())
    {
        ROCP_ERROR << "Failed to continue target process after " << description << " writing";
        return false;
    }

    ROCP_TRACE << "Wrote " << description << " to target process";
    return true;
}

// Helper function to build environment buffer
std::vector<uint8_t>
build_environment_buffer()
{
    std::vector<uint8_t> environment_buffer(4);
    uint32_t             var_count = 0;

    char** invars = environ;
    for(; *invars; invars++)
    {
        const char* var = *invars;
        if(strncmp("ROCP", var, 4) != 0)
        {
            continue;
        }

        var_count++;
        ROCP_TRACE << "Adding to environment buffer: " << var;

        // Add variable name
        while(*var != '=')
        {
            environment_buffer.emplace_back(*var++);
        }
        environment_buffer.emplace_back(0);

        // Add variable value
        var++;
        while(*var != 0)
        {
            environment_buffer.emplace_back(*var++);
        }
        environment_buffer.emplace_back(0);
    }

    // Store count in first 4 bytes
    const uint8_t* var_count_bytes = reinterpret_cast<uint8_t*>(&var_count);
    std::copy(var_count_bytes, var_count_bytes + 4, environment_buffer.data());

    return environment_buffer;
}
}  // anonymous namespace

ROCPROFILER_EXTERN_C_INIT

void
handle_ptrace_operations(uint32_t pid)
{
    // Setup attachement for rocprofiler
    ROCP_TRACE << "Attachment library called for pid " << pid;
    ptrace_session = std::make_unique<rocprofiler::attach::PTraceSession>(pid);
    ROCP_TRACE << "Attempting attachment to pid " << pid;
    if(!ptrace_session->attach())
    {
        ROCP_ERROR << "Attachment failed to pid " << pid;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        finished_setup.store(true);
        return;
    }
    ROCP_TRACE << "Attachment success to pid " << pid;

    // Build and write environment buffer to target process
    auto  environment_buffer      = build_environment_buffer();
    void* environment_buffer_addr = nullptr;
    if(!write_data_to_target("environment buffer", environment_buffer, environment_buffer_addr))
    {
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Build and write tool library path to target process
    auto tool_lib_path_env =
        rocprofiler::common::get_env("ROCPROF_ATTACH_TOOL_LIBRARY", "librocprofiler-sdk-tool.so");
    const char* tool_lib_path = tool_lib_path_env.c_str();
    ROCP_TRACE << "Tool library path: " << tool_lib_path;

    size_t               tool_lib_path_len = strlen(tool_lib_path) + 1;
    std::vector<uint8_t> tool_lib_buffer(tool_lib_path, tool_lib_path + tool_lib_path_len);

    void* tool_lib_path_addr = nullptr;
    if(!write_data_to_target("tool library path", tool_lib_buffer, tool_lib_path_addr))
    {
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Execute the attach function with both parameters
    if(!ptrace_session->call_function("librocprofiler-register.so",
                                      "rocprofiler_register_attach",
                                      environment_buffer_addr,
                                      tool_lib_path_addr))
    {
        ROCP_ERROR << "Failed to call attach function in target process " << pid;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Clean up - free the tool library path memory in target process
    if(!ptrace_session->simple_munmap(tool_lib_path_addr, tool_lib_path_len))
    {
        ROCP_ERROR << "Failed to free tool library path memory in target process";
        // Continue anyway since the main operation succeeded
    }
    ROCP_TRACE << "Cleaned up tool library path memory in target process";

    // Allow main thread to continue
    finished_setup.store(true);
    if(!ptrace_session->handle_signals())
    {
        ROCP_ERROR << "Signal handling loop terminated unexepectedly for pid " << pid;
        // don't return, try to detach anyways
    }
    // Detach rocprofiler
    ROCP_TRACE << "Detaching rocprofiler from pid " << pid;
    if(!ptrace_session->call_function("librocprofiler-register.so", "rocprofiler_register_detach"))
    {
        ROCP_ERROR << "Failed to call detach function in target process";
        // don't return, try to detach anyways
    }
    ptrace_session->stop();
    ptrace_session->detach();
    ptrace_session.reset();
}

int
attach(uint32_t pid)
{
    initialize_logging();
    ptrace_thread = std::thread(handle_ptrace_operations, pid);
    // Wait for ptrace thread to finish setting up
    while(!finished_setup.load())
        std::this_thread::yield();

    auto status = ptrace_session->m_setup_status.load();
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        ROCP_ERROR << "ptrace session failed with error code " << ptrace_session->m_setup_status;
        ptrace_thread.join();
        finished_setup.store(false);
        return status;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
detach()
{
    ptrace_session->detach_ptrace_session();
    ptrace_thread.join();
    finished_setup.store(false);
    return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_EXTERN_C_FINI
