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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace
{
namespace fs = std::filesystem;

bool
has_attach_listener(pid_t pid)
{
    auto task_dir = fs::path{"/proc"} / std::to_string(pid) / "task";
    auto ec       = std::error_code{};
    for(const auto& entry : fs::directory_iterator(task_dir, ec))
    {
        if(!entry.is_directory()) continue;
        auto comm = std::ifstream{entry.path() / "comm"};
        auto name = std::string{};
        if(std::getline(comm, name) && name == "rocp-bg-attach") return true;
    }
    return false;
}

bool
wait_for_attach_listener(pid_t pid)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(has_attach_listener(pid)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    return false;
}

pid_t
wait_for_listenerless_child(pid_t parent_pid)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while(std::chrono::steady_clock::now() < deadline)
    {
        auto task_dir = fs::path{"/proc"} / std::to_string(parent_pid) / "task";
        auto ec       = std::error_code{};
        for(const auto& entry : fs::directory_iterator(task_dir, ec))
        {
            if(!entry.is_directory()) continue;
            auto children  = std::ifstream{entry.path() / "children"};
            auto child_pid = pid_t{-1};
            while(children >> child_pid)
            {
                if(!has_attach_listener(child_pid)) return child_pid;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    return -1;
}

bool
is_library_loaded(pid_t pid, const std::string& library_path)
{
    auto maps = std::ifstream{"/proc/" + std::to_string(pid) + "/maps"};
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
        kill(-root_pid, signal);
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

    auto child_pid = wait_for_listenerless_child(root_pid);
    if(child_pid < 0)
    {
        std::cerr << "Test FAILED: listenerless fork-only child did not start\n";
        cleanup(SIGKILL);
        return 1;
    }

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], 1);
    if(rocattach_attach_tree(root_pid) != ROCATTACH_STATUS_SUCCESS ||
       !is_library_loaded(root_pid, argv[2]) || is_library_loaded(child_pid, argv[2]) ||
       rocattach_detach_tree(root_pid) != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "Test FAILED: listenerless child tree behavior was incorrect\n";
        cleanup(SIGKILL);
        return 1;
    }

    auto status = cleanup(SIGINT);
    if(!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        std::cerr << "Test FAILED: target process did not exit cleanly\n";
        return 1;
    }

    std::cout << "Test PASSED: tree attach skipped listenerless fork-only child\n";
    return 0;
}
