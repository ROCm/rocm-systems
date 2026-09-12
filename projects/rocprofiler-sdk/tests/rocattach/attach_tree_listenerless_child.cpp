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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk-rocattach/rocattach.h>
#include <rocprofiler-sdk-rocattach/types.h>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace
{
namespace fs = std::filesystem;

enum class listener_scan_result
{
    found,
    not_found,
    error,
};

listener_scan_result
scan_for_attach_listener(pid_t pid)
{
    auto task_dir = fs::path{"/proc"} / std::to_string(pid) / "task";
    auto ec       = std::error_code{};
    auto itr      = fs::directory_iterator(task_dir, ec);
    auto end      = fs::directory_iterator{};
    while(!ec && itr != end)
    {
        auto entry    = *itr;
        auto entry_ec = std::error_code{};
        if(!entry.is_directory(entry_ec))
        {
            if(entry_ec == std::errc::no_such_file_or_directory)
            {
                itr.increment(ec);
                continue;
            }
            if(entry_ec) return listener_scan_result::error;
            itr.increment(ec);
            continue;
        }

        auto comm_path = entry.path() / "comm";
        errno          = 0;
        auto comm      = std::ifstream{comm_path};
        if(!comm.is_open())
        {
            auto exists_ec = std::error_code{};
            if((errno == ENOENT || !fs::exists(comm_path, exists_ec)) && !exists_ec)
            {
                itr.increment(ec);
                continue;
            }
            return listener_scan_result::error;
        }

        auto name = std::string{};
        if(!std::getline(comm, name))
        {
            auto exists_ec = std::error_code{};
            if(!fs::exists(comm_path, exists_ec) && !exists_ec)
            {
                itr.increment(ec);
                continue;
            }
            return listener_scan_result::error;
        }
        if(name == "rocp-bg-attach") return listener_scan_result::found;
        itr.increment(ec);
    }
    return (ec) ? listener_scan_result::error : listener_scan_result::not_found;
}

bool
wait_for_attach_listener(pid_t pid)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while(std::chrono::steady_clock::now() < deadline)
    {
        auto result = scan_for_attach_listener(pid);
        if(result == listener_scan_result::found) return true;
        if(result == listener_scan_result::error)
        {
            std::cerr << "Test FAILED: could not inspect root process tasks\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    return false;
}

struct child_layout
{
    pid_t listenerless = -1;
    pid_t listener     = -1;
};

std::optional<child_layout>
wait_for_child_layout(pid_t parent_pid)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while(std::chrono::steady_clock::now() < deadline)
    {
        auto children_path = fs::path{"/proc"} / std::to_string(parent_pid) / "task" /
                             std::to_string(parent_pid) / "children";
        auto children_file = std::ifstream{children_path};
        if(!children_file.is_open())
        {
            std::cerr << "Test FAILED: could not inspect child process list\n";
            return std::nullopt;
        }

        auto child_pids = std::vector<pid_t>{};
        auto child_pid  = pid_t{-1};
        while(children_file >> child_pid)
            child_pids.emplace_back(child_pid);

        if(child_pids.size() == 2)
        {
            auto layout = child_layout{};
            for(auto pid : child_pids)
            {
                auto result = scan_for_attach_listener(pid);
                if(result == listener_scan_result::found)
                    layout.listener = pid;
                else if(result == listener_scan_result::not_found)
                    layout.listenerless = pid;
                else
                {
                    std::cerr << "Test FAILED: could not inspect child process tasks\n";
                    return std::nullopt;
                }
            }
            if(layout.listener > 0 && layout.listenerless > 0) return layout;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    return std::nullopt;
}

std::optional<bool>
is_library_loaded(pid_t pid, const std::string& library_path)
{
    auto maps = std::ifstream{"/proc/" + std::to_string(pid) + "/maps"};
    if(!maps.is_open())
    {
        std::cerr << "Test FAILED: could not inspect process mappings for pid " << pid << '\n';
        return std::nullopt;
    }
    auto line = std::string{};
    while(std::getline(maps, line))
    {
        if(line.find(library_path) != std::string::npos) return true;
    }
    return false;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " <attachment-test> <tool-library>\n";
        return 1;
    }

    auto root_pid = fork();
    if(root_pid < 0)
    {
        std::cerr << "Test FAILED: fork failed\n";
        return 1;
    }
    if(root_pid == 0)
    {
        setpgid(0, 0);
        execl(argv[1], argv[1], "--fork-child-after-init", "1", nullptr);
        _exit(1);
    }
    setpgid(root_pid, root_pid);

    auto cleanup = [&](int signal) {
        kill(root_pid, signal);
        if(signal == SIGKILL) kill(-root_pid, signal);
        auto status = int{0};
        waitpid(root_pid, &status, 0);
        return status;
    };

    if(!wait_for_attach_listener(root_pid))
    {
        std::cerr << "Test FAILED: root process did not create rocp-bg-attach\n";
        cleanup(SIGKILL);
        return 1;
    }

    auto children = wait_for_child_layout(root_pid);
    if(!children)
    {
        std::cerr << "Test FAILED: expected child process layout did not become ready\n";
        cleanup(SIGKILL);
        return 1;
    }

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], 1);
    if(rocattach_attach_tree(root_pid) != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "Test FAILED: tree attachment failed\n";
        cleanup(SIGKILL);
        return 1;
    }

    auto root_loaded         = is_library_loaded(root_pid, argv[2]);
    auto listener_loaded     = is_library_loaded(children->listener, argv[2]);
    auto listenerless_loaded = is_library_loaded(children->listenerless, argv[2]);
    auto detach_status       = rocattach_detach_tree(root_pid);
    if(!root_loaded || !root_loaded.value() || !listener_loaded || !listener_loaded.value() ||
       !listenerless_loaded || listenerless_loaded.value() ||
       detach_status != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "Test FAILED: listenerless child tree behavior was incorrect\n";
        cleanup(SIGKILL);
        return 1;
    }

    auto status = cleanup(SIGKILL);
    if(!WIFSIGNALED(status) || WTERMSIG(status) != SIGKILL)
    {
        std::cerr << "Test FAILED: target process did not terminate\n";
        return 1;
    }

    std::cout << "Test PASSED: tree attach skipped listenerless fork-only child and attached "
                 "listener-capable sibling\n";
    return 0;
}
