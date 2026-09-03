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
#include <cstdlib>
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

    int child_pipe[2];
    if(pipe(child_pipe) != 0)
    {
        std::cerr << "Test FAILED: pipe creation failed\n";
        return 1;
    }

    auto root_pid = fork();
    if(root_pid < 0)
    {
        std::cerr << "Test FAILED: root fork failed\n";
        return 1;
    }
    if(root_pid == 0)
    {
        close(child_pipe[0]);
        setpgid(0, 0);
        unsetenv("ROCP_TOOL_ATTACH");

        auto child_pid = fork();
        if(child_pid < 0) _exit(1);
        if(child_pid == 0)
        {
            close(child_pipe[1]);
            // The descendant can remain uninterruptible briefly during GPU teardown.
            // Do not let an inherited CTest output pipe keep the completed harness alive.
            close(STDOUT_FILENO);
            close(STDERR_FILENO);
            setenv("ROCP_TOOL_ATTACH", "1", 1);
            execl(argv[1], argv[1], "1", "1", nullptr);
            _exit(1);
        }

        if(write(child_pipe[1], &child_pid, sizeof(child_pid)) !=
           static_cast<ssize_t>(sizeof(child_pid)))
            _exit(1);
        close(child_pipe[1]);
        while(true)
            pause();
    }

    close(child_pipe[1]);
    setpgid(root_pid, root_pid);

    auto cleanup = [&]() {
        kill(-root_pid, SIGKILL);
        auto status = int{};
        waitpid(root_pid, &status, 0);
    };

    auto child_pid = pid_t{-1};
    if(read(child_pipe[0], &child_pid, sizeof(child_pid)) !=
       static_cast<ssize_t>(sizeof(child_pid)))
    {
        std::cerr << "Test FAILED: could not receive descendant PID\n";
        close(child_pipe[0]);
        cleanup();
        return 1;
    }
    close(child_pipe[0]);

    if(!wait_for_attach_listener(child_pid))
    {
        std::cerr << "Test FAILED: descendant did not create an attachment listener\n";
        cleanup();
        return 1;
    }
    if(has_attach_listener(root_pid))
    {
        std::cerr << "Test FAILED: launcher root unexpectedly has an attachment listener\n";
        cleanup();
        return 1;
    }

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], 1);
    auto status = rocattach_attach_tree(root_pid);
    if(status != ROCATTACH_STATUS_ERROR)
    {
        std::cerr << "Test FAILED: non-attachable root returned status " << status << '\n';
        if(status == ROCATTACH_STATUS_SUCCESS) rocattach_detach_tree(root_pid);
        cleanup();
        return 1;
    }
    if(is_library_loaded(child_pid, argv[2]))
    {
        std::cerr << "Test FAILED: attachable descendant was attempted after root failure\n";
        cleanup();
        return 1;
    }

    cleanup();
    std::cout << "Test PASSED: non-attachable root prevented descendant attachment\n";
    return 0;
}
