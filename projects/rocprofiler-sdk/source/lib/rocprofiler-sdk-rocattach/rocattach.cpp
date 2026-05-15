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

#include <rocprofiler-sdk-rocattach/defines.h>
#include <rocprofiler-sdk-rocattach/rocattach.h>
#include <rocprofiler-sdk-rocattach/types.h>

#include <map>
#include <mutex>
#include <unordered_map>

extern char** environ;

namespace rocprofiler
{
namespace rocattach
{
namespace
{
using session_t      = rocprofiler::rocattach::PTraceSession;
using session_list_t = std::map<int, session_t>;

#define ROCATTACH_STATUS_STRING(CODE, MSG)                                                         \
    template <>                                                                                    \
    struct status_string<CODE>                                                                     \
    {                                                                                              \
        static constexpr auto name  = #CODE;                                                       \
        static constexpr auto value = MSG;                                                         \
    };

template <size_t Idx>
struct status_string;

ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_SUCCESS, "Success")
ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_ERROR, "General error")
ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT, "Invalid function argument")
ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_ERROR_NOT_SUPPORTED,
                        "Attachment not supported on this platform")
ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_ERROR_PTRACE_ERROR, "General ptrace error")
ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_ERROR_PTRACE_OPERATION_NOT_PERMITTED,
                        "ptrace returned EPERM, operation not permitted")
ROCATTACH_STATUS_STRING(ROCATTACH_STATUS_ERROR_PTRACE_PROCESS_NOT_FOUND,
                        "ptrace returned ESRCH, no such process")

template <size_t Idx, size_t... Tail>
const char*
get_status_name(rocattach_status_t status, std::index_sequence<Idx, Tail...>)
{
    if(status == Idx) return status_string<Idx>::name;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_status_name(status, std::index_sequence<Tail...>{});
    return nullptr;
}

template <size_t Idx, size_t... Tail>
const char*
get_status_string(rocattach_status_t status, std::index_sequence<Idx, Tail...>)
{
    if(status == Idx) return status_string<Idx>::value;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_status_string(status, std::index_sequence<Tail...>{});
    return nullptr;
}

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCATTACH", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

session_list_t*
get_sessions()
{
    static auto*& session_list = rocprofiler::common::static_object<session_list_t>::construct();
    return session_list;
}

std::lock_guard<std::mutex>
get_sessions_lock_guard()
{
    static auto*& m = rocprofiler::common::static_object<std::mutex>::construct();
    return std::lock_guard(*CHECK_NOTNULL(m));
}

// Helper function to allocate memory in target process and write data
rocattach_status_t
write_data_to_target(session_t&                  session,
                     const std::string&          description,
                     const std::vector<uint8_t>& data,
                     void*&                      allocated_addr)
{
    // Allocate memory in target process
    auto status = ROCATTACH_STATUS_SUCCESS;
    status      = session.simple_mmap(allocated_addr, data.size());
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to allocate memory for " << description
                   << " in target process pid " << session.get_pid();
        return status;
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Allocated memory for " << description << " at "
               << allocated_addr << " in target process pid " << session.get_pid();

    // Write data to target process memory
    status = session.write(reinterpret_cast<size_t>(allocated_addr), data, data.size());
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to write " << description
                   << " to target process pid " << session.get_pid();
        return status;
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Wrote " << description << " to target process pid "
               << session.get_pid();
    return status;
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
            // only take envvars starting with ROCP
            continue;
        }

        var_count++;
        ROCP_TRACE << "[rocprofiler-sdk-rocattach] Adding to environment buffer: " << var;

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

rocattach_status_t
setup(int pid)
{
    // Setup attachement for rocprofiler
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attachment library rocattach_attach function called "
                  "for pid "
               << pid;

    auto*      sessions = CHECK_NOTNULL(get_sessions());
    session_t* session;
    {
        auto lg = get_sessions_lock_guard();
        if(sessions->count(pid) > 0)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_attach called for pid " << pid
                       << ", which already has an active "
                          "attachment session.";
            return ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT;
        }

        sessions->emplace(pid, pid);
        session = &(sessions->at(pid));
    }
    auto status = ROCATTACH_STATUS_SUCCESS;

    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attempting attachment to pid " << pid;
    status = session->attach();
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Attachment failed to pid " << pid
                   << " with status code " << status;
        return status;
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attachment success to pid " << pid;

    // Build and write environment buffer to target process
    auto  environment_buffer      = build_environment_buffer();
    void* environment_buffer_addr = nullptr;
    status                        = write_data_to_target(
        *session, "environment buffer", environment_buffer, environment_buffer_addr);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        return status;
    }

    // Build and write tool library path to target process
    auto tool_lib_path_env =
        rocprofiler::common::get_env("ROCPROF_ATTACH_TOOL_LIBRARY", "librocprofiler-sdk-tool.so");
    const char* tool_lib_path = tool_lib_path_env.c_str();
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Tool library path: " << tool_lib_path;

    size_t               tool_lib_path_len = strlen(tool_lib_path) + 1;
    std::vector<uint8_t> tool_lib_buffer(tool_lib_path, tool_lib_path + tool_lib_path_len);

    void* tool_lib_path_addr = nullptr;
    status =
        write_data_to_target(*session, "tool library path", tool_lib_buffer, tool_lib_path_addr);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        return status;
    }

    uint64_t retval = 0;
    // Execute the attach function with both parameters
    status = session->call_function("librocprofiler-register.so",
                                    "rocprofiler_register_attach",
                                    retval,
                                    environment_buffer_addr,
                                    tool_lib_path_addr);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR
            << "[rocprofiler-sdk-rocattach] Failed to call "
               "rocprofiler-register::rocprofiler_register_attach function in target process "
            << pid << ". status: " << status;
        return status;
    }
    else if(retval != 0)
    {
        ROCP_ERROR
            << "[rocprofiler-sdk-rocattach] rocprofiler-register::rocprofiler_register_attach "
               "function returned non-zero status in target process "
            << pid << ". return: " << retval;
        return ROCATTACH_STATUS_ERROR;
    }

    // Clean up - free the environment buffer and tool library path memory in target process
    status = session->simple_munmap(environment_buffer_addr, environment_buffer.size());
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to free environment buffer memory in "
                      "target process "
                   << pid << ", continuing...";
        // Continue anyway since the main operation succeeded
    }
    ROCP_TRACE
        << "[rocprofiler-sdk-rocattach] Cleaned up tool environment memory in target process "
        << pid;

    status = session->simple_munmap(tool_lib_path_addr, tool_lib_path_len);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to free tool library path memory in "
                      "target process "
                   << pid << ", continuing...";
        // Continue anyway since the main operation succeeded
    }
    ROCP_TRACE
        << "[rocprofiler-sdk-rocattach] Cleaned up tool library path memory in target process "
        << pid;
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
teardown(int pid)
{
    // Setup attachement for rocprofiler
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attachment library rocattach_detach function called "
                  "for pid "
               << pid;

    auto*      sessions = CHECK_NOTNULL(get_sessions());
    session_t* session;
    {
        auto lg = get_sessions_lock_guard();
        if(sessions->count(pid) == 0)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_detach called for pid " << pid
                       << ", which has no active "
                          "attachment session.";
            return ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT;
        }

        session = &(sessions->at(pid));
    }
    auto status = ROCATTACH_STATUS_SUCCESS;

    uint64_t retval = 0;
    // Execute the attach function with both parameters
    status =
        session->call_function("librocprofiler-register.so", "rocprofiler_register_detach", retval);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR
            << "[rocprofiler-sdk-rocattach] Failed to call "
               "rocprofiler-register::rocprofiler_register_detach function in target process "
            << pid << ". status: " << status;
        // continue to detach anyways
    }
    else if(retval != 0)
    {
        ROCP_ERROR
            << "[rocprofiler-sdk-rocattach] rocprofiler-register::rocprofiler_register_detach "
               "function returned non-zero status in target process "
            << pid << ". return: " << retval;
        // continue to detach anyways
    }

    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attempting detachment to pid " << pid;
    status = session->detach();
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Detachment failed from pid " << pid;
        return status;
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Detachment success from pid " << pid;

    {
        auto lg = get_sessions_lock_guard();
        sessions->erase(pid);
    }

    return ROCATTACH_STATUS_SUCCESS;
}

}  // namespace
}  // namespace rocattach
}  // namespace rocprofiler

ROCATTACH_EXTERN_C_INIT

rocattach_status_t
rocattach_attach(int pid)
{
    rocprofiler::rocattach::initialize_logging();

    if(!rocprofiler::rocattach::PTraceSession::is_supported())
    {
        ROCP_ERROR << "[rocprofiler-sdk-attach] rocattach is not supported on this platform.";
        return ROCATTACH_STATUS_ERROR_NOT_SUPPORTED;
    }

    auto status = rocprofiler::rocattach::setup(pid);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_attach failed with error code "
                   << status;
        return status;
    }
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
rocattach_detach(int pid)
{
    rocprofiler::rocattach::initialize_logging();
    if(pid != 0)
    {
        auto status = rocprofiler::rocattach::teardown(pid);
        if(status != ROCATTACH_STATUS_SUCCESS)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_detach failed with error code "
                       << status;
            return status;
        }
        return ROCATTACH_STATUS_SUCCESS;
    }
    else
    {
        ROCP_INFO << "[rocprofiler-sdk-rocattach] rocattach_detach received pid=0, detaching from "
                     "ALL sessions";
        std::vector<int> pids;
        {
            auto lg = rocprofiler::rocattach::get_sessions_lock_guard();
            for(auto& pair_itr : *(CHECK_NOTNULL(rocprofiler::rocattach::get_sessions())))
            {
                pids.emplace_back(pair_itr.first);
            }
        }

        for(int pid_itr : pids)
        {
            rocprofiler::rocattach::teardown(pid_itr);
        }
        return ROCATTACH_STATUS_SUCCESS;
    }
}

rocattach_status_t
rocattach_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    *CHECK_NOTNULL(major) = ROCATTACH_VERSION_MAJOR;
    *CHECK_NOTNULL(minor) = ROCATTACH_VERSION_MINOR;
    *CHECK_NOTNULL(patch) = ROCATTACH_VERSION_PATCH;
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
rocattach_get_version_triplet(rocattach_version_triplet_t* info)
{
    *CHECK_NOTNULL(info) = {.major = ROCATTACH_VERSION_MAJOR,
                            .minor = ROCATTACH_VERSION_MINOR,
                            .patch = ROCATTACH_VERSION_PATCH};
    return ROCATTACH_STATUS_SUCCESS;
}

const char*
rocattach_get_status_name(rocattach_status_t status)
{
    return rocprofiler::rocattach::get_status_name(
        status, std::make_index_sequence<ROCATTACH_STATUS_LAST>{});
}

const char*
rocattach_get_status_string(rocattach_status_t status)
{
    return rocprofiler::rocattach::get_status_string(
        status, std::make_index_sequence<ROCATTACH_STATUS_LAST>{});
}

ROCATTACH_EXTERN_C_FINI
