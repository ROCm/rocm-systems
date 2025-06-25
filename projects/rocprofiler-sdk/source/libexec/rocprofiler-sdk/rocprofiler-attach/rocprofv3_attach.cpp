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

extern char** environ;

namespace common = ::rocprofiler::common;

namespace
{
std::unique_ptr<rocprofiler::attach::PTraceSession> ptrace_session;
}

ROCPROFILER_EXTERN_C_INIT
void
attach(uint32_t pid) ROCPROFILER_EXPORT;

void
detach() ROCPROFILER_EXPORT;
ROCPROFILER_EXTERN_C_FINI

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

ROCPROFILER_EXTERN_C_INIT

void
attach(uint32_t pid)
{
    initialize_logging();
    ROCP_TRACE << "Attachment library called for pid " << pid;
    ptrace_session = std::make_unique<rocprofiler::attach::PTraceSession>(pid);
    ROCP_TRACE << "Attempting attachment to pid " << pid;
    if(!ptrace_session->attach())
    {
        ROCP_ERROR << "Attachment failed to pid " << pid;
        return;
    }
    ROCP_TRACE << "Attachment success to pid " << pid;

    // Environment_buffer is a null-character delimited list of name value pairs.
    // Each name and value is delimited separately.
    // The first 4 bytes contain an uint32_t count of pairs

    std::vector<uint8_t> environment_buffer(4);
    {
        uint32_t var_count = 0;
        char**   invars    = environ;
        for(; *invars; invars++)
        {
            const char* var = *invars;
            if(strncmp("ROCPROF", var, 7) != 0)
            {
                continue;
            }
            var_count++;
            ROCP_TRACE << "Adding to environment buffer: " << var;
            while(*var != '=')
            {
                environment_buffer.emplace_back(*var++);
            }
            environment_buffer.emplace_back(0);

            var++;
            while(*var)
            {
                environment_buffer.emplace_back(*var++);
            }
            environment_buffer.emplace_back(0);
        }

        const uint8_t* var_count_bytes = reinterpret_cast<uint8_t*>(&var_count);
        std::copy(var_count_bytes, var_count_bytes + 4, environment_buffer.data());
    }

    // Now, allocate a buffer to store the environment variables
    void* environment_buffer_addr = nullptr;
    if(!ptrace_session->simple_mmap(environment_buffer_addr, environment_buffer.size()))
    {
        ROCP_ERROR << "Failed to call mmap in target process";
        return;
    }
    ROCP_TRACE << "mmap'd in target process at " << environment_buffer_addr;

    // Write to that buffer
    if(!ptrace_session->stop())
    {
        ROCP_ERROR << "Failed to stop target process for environment buffer writing";
        return;
    }
    if(!ptrace_session->write(reinterpret_cast<size_t>(environment_buffer_addr),
                              environment_buffer,
                              environment_buffer.size()))
    {
        ROCP_ERROR << "Failed to write environment buffer in target process";
        return;
    }
    if(!ptrace_session->cont())
    {
        ROCP_ERROR << "Failed to continue target process for environment buffer writing";
        return;
    }
    ROCP_TRACE << "wrote environment buffer to target process";

    // Execute the attach function with the buffer addr as parameter
    if(!ptrace_session->call_function(
           "librocprofiler-register.so", "rocprofiler_register_attach", environment_buffer_addr))
    {
        ROCP_ERROR << "Failed to call attach function in target process " << pid;
        return;
    }
}

void
detach()
{
    if(!ptrace_session->call_function("librocprofiler-register.so", "rocprofiler_register_detach"))
    {
        ROCP_ERROR << "Failed to call detach function in target process";
        // don't return, try to detach anyways
    }
    ptrace_session->stop();
    ptrace_session->detach();
    ptrace_session.reset();
}

ROCPROFILER_EXTERN_C_FINI
