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

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>

extern char** environ;

namespace rocprofiler
{
namespace rocattach
{
namespace
{
using session_t = rocprofiler::rocattach::PTraceSession;

struct pid_entry_t
{
    explicit pid_entry_t(pid_t tid)
    : session(tid)
    {}
    session_t session;
    // Non-empty only for tree-attach root PIDs. Holds the ordered list of PIDs
    // successfully attached by rocattach_attach_tree() for this root. Consumed
    // (and cleared) by the corresponding rocattach_detach_tree() call.
    std::vector<pid_t> tree_pids;
};

using session_list_t = std::map<int, pid_entry_t>;

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

// Returns the TID of the 'rocp-bg-attach' thread if found, or -1 if not found
pid_t
resolve_attach_tid(pid_t pid)
{
    auto            task_dir = "/proc/" + std::to_string(pid) + "/task";
    std::error_code ec;
    for(const auto& entry : std::filesystem::directory_iterator(task_dir, ec))
    {
        if(!entry.is_directory()) continue;

        auto          comm_path = entry.path() / "comm";
        std::ifstream comm_file(comm_path);
        std::string   name;
        if(std::getline(comm_file, name) && name == "rocp-bg-attach")
        {
            pid_t tid = std::stoi(entry.path().filename().string());
            ROCP_INFO << "[rocprofiler-sdk-rocattach] Found background thread TID " << tid
                      << " for pid " << pid << " via /proc scan";
            return tid;
        }
    }

    if(ec)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to scan " << task_dir
                   << " for 'rocp-bg-attach' thread: " << ec.message();
    }
    else
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Could not find 'rocp-bg-attach' thread in "
                   << task_dir;
    }

    return -1;
}

rocattach_status_t
setup(int pid)
{
    // Setup attachment for rocprofiler
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

        auto target_tid = resolve_attach_tid(pid);

        if(target_tid < 0)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] Cannot attach to process " << pid
                       << ": 'rocp-bg-attach' thread not found. The target process does not "
                          "appear to have attach support enabled. Start the target with "
                          "ROCP_TOOL_ATTACH=1, or use a rocprofiler-register build configured "
                          "with ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON.";
            return ROCATTACH_STATUS_ERROR;
        }

        ROCP_INFO << "[rocprofiler-sdk-rocattach] Attaching to PID " << pid
                  << " via background thread TID " << target_tid;
        sessions->emplace(pid, target_tid);
        session = &(sessions->at(pid).session);
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

    // Build and write tool library path to target process. Do not honor the
    // user-controllable ROCPROF_ATTACH_TOOL_LIBRARY override in a secure-execution
    // context (setuid/setgid, file capabilities, etc.), so an unprivileged user
    // cannot cause a privileged attach helper to inject an arbitrary library.
    auto        tool_lib_path_env = rocprofiler::common::is_at_secure()
                                        ? std::string{"librocprofiler-sdk-tool.so"}
                                        : rocprofiler::common::get_env("ROCPROF_ATTACH_TOOL_LIBRARY",
                                                                "librocprofiler-sdk-tool.so");
    const char* tool_lib_path     = tool_lib_path_env.c_str();
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

// Collect all PIDs in the process tree rooted at root_pid (BFS via /proc).
// Returns root_pid plus all descendant PIDs. Enumerates children for every thread
// since children can be spawned via clone() from any thread.
std::vector<pid_t>
collect_process_tree(pid_t root_pid)
{
    std::vector<pid_t> result;
    std::queue<pid_t>  worklist;
    worklist.push(root_pid);

    while(!worklist.empty())
    {
        pid_t pid = worklist.front();
        worklist.pop();
        result.push_back(pid);

        auto            task_dir = "/proc/" + std::to_string(pid) + "/task";
        std::error_code ec;
        for(const auto& entry : std::filesystem::directory_iterator(task_dir, ec))
        {
            if(!entry.is_directory()) continue;
            auto          children_path = entry.path() / "children";
            std::ifstream children_file(children_path);
            pid_t         child_pid;
            while(children_file >> child_pid)
                worklist.push(child_pid);
        }
    }
    return result;
}

rocattach_status_t
teardown(int pid)
{
    // Setup attachment for rocprofiler
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

        session = &(sessions->at(pid).session);
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
rocattach_attach_tree(int root_pid)
{
    rocprofiler::rocattach::initialize_logging();

    if(!rocprofiler::rocattach::PTraceSession::is_supported())
    {
        ROCP_ERROR << "[rocprofiler-sdk-attach] rocattach is not supported on this platform.";
        return ROCATTACH_STATUS_ERROR_NOT_SUPPORTED;
    }

    auto pids = rocprofiler::rocattach::collect_process_tree(root_pid);
    ROCP_INFO << "[rocprofiler-sdk-rocattach] Found " << pids.size()
              << " process(es) in tree rooted at pid " << root_pid;

    std::vector<pid_t> attached_pids;
    auto               last_status = ROCATTACH_STATUS_SUCCESS;
    for(pid_t pid : pids)
    {
        auto status = rocprofiler::rocattach::setup(pid);
        if(status != ROCATTACH_STATUS_SUCCESS)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_attach_tree failed for pid " << pid
                       << " with error code " << status << ", continuing with remaining processes";
            last_status = status;
        }
        else
        {
            attached_pids.push_back(pid);
        }
    }

    {
        auto  lg       = rocprofiler::rocattach::get_sessions_lock_guard();
        auto* sessions = CHECK_NOTNULL(rocprofiler::rocattach::get_sessions());
        auto  it       = sessions->find(root_pid);
        if(it != sessions->end())
        {
            it->second.tree_pids = std::move(attached_pids);
        }
        else
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_attach_tree could not record tree "
                          "session for root pid "
                       << root_pid;
        }
    }

    return last_status;
}

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
rocattach_detach_tree(int root_pid)
{
    rocprofiler::rocattach::initialize_logging();

    // Retrieve the PID list recorded by rocattach_attach_tree() from the root's map entry.
    // Using the recorded list (rather than re-enumerating /proc) ensures we detach exactly
    // the processes that were attached, no more and no less.
    std::vector<pid_t> pids;
    {
        auto  lg       = rocprofiler::rocattach::get_sessions_lock_guard();
        auto* sessions = CHECK_NOTNULL(rocprofiler::rocattach::get_sessions());
        auto  it       = sessions->find(root_pid);
        if(it == sessions->end() || it->second.tree_pids.empty())
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_detach_tree called for root pid "
                       << root_pid << " which has no recorded tree attachment session.";
            return ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT;
        }
        pids = std::move(it->second.tree_pids);
        // If root_pid was not successfully attached it won't be in pids, meaning teardown()
        // won't erase its entry. Remove it here to avoid leaving an orphan in the map.
        if(pids.empty() || pids.front() != root_pid) sessions->erase(it);
    }

    ROCP_INFO << "[rocprofiler-sdk-rocattach] rocattach_detach_tree detaching " << pids.size()
              << " process(es) for root pid " << root_pid;

    auto last_status = ROCATTACH_STATUS_SUCCESS;
    for(pid_t pid : pids)
    {
        auto status = rocprofiler::rocattach::teardown(pid);
        if(status != ROCATTACH_STATUS_SUCCESS)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach_detach_tree failed for pid " << pid
                       << " with error code " << status << ", continuing with remaining processes";
            last_status = status;
        }
    }
    return last_status;
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
