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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Tests rocattach_attach_tree(), which attaches to a process and all of its
// descendant processes. The parent forks a child which executes the test app;
// rocattach_attach_tree() is called on the parent PID and must also attach to
// the child without the caller enumerating it explicitly.

#include <rocprofiler-sdk-rocattach/defines.h>
#include <rocprofiler-sdk-rocattach/rocattach.h>
#include <rocprofiler-sdk-rocattach/types.h>

#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

static bool
wait_for_attach_ready(pid_t pid)
{
    constexpr int max_wait_sec = 30;
    auto          task_dir     = "/proc/" + std::to_string(pid) + "/task";

    std::cout << "Waiting for rocp-bg-attach thread in PID " << pid << "...\n";
    for(int elapsed = 0; elapsed < max_wait_sec; ++elapsed)
    {
        std::error_code ec;
        for(const auto& entry : std::filesystem::directory_iterator(task_dir, ec))
        {
            if(!entry.is_directory()) continue;
            std::ifstream comm(entry.path() / "comm");
            std::string   name;
            if(std::getline(comm, name) && name == "rocp-bg-attach")
            {
                std::cout << "Attachment ready (" << elapsed << "s elapsed)\n";
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Timed out after " << max_wait_sec << "s waiting for rocp-bg-attach thread\n";
    return false;
}

#define ROCATTACH_CALL(func)                                                                       \
    {                                                                                              \
        std::cout << "starting call to " #func << std::endl;                                       \
        rocattach_status_t status = func;                                                          \
        if(status != ROCATTACH_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            std::cout << "error: call to " #func " returned non zero status " << status            \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            std::cout << "call to " #func " successful" << std::endl;                              \
        }                                                                                          \
    }

// Check whether a shared library appears in /proc/<pid>/maps, which confirms
// it was loaded into the target process by the attachment machinery.
bool
is_library_loaded(pid_t pid, const std::string& library_path)
{
    std::ifstream maps("/proc/" + std::to_string(pid) + "/maps");
    std::string   line;
    while(std::getline(maps, line))
    {
        if(line.find(library_path) != std::string::npos) return true;
    }
    return false;
}

int
main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "error: wrong number of arguments\n";
        return 1;
    }

    // Build a two-level process tree: parent → child → grandchild.
    // Both child and grandchild exec the test application.
    // rocattach_attach_tree() is called with only child_pid; it must discover
    // the grandchild automatically via /proc to exercise tree traversal.
    //
    // A pipe is used to communicate the grandchild's PID from child to parent
    // so the parent can verify the tool library was loaded in both processes.
    int pipefd[2];
    if(pipe(pipefd) != 0)
    {
        std::cout << "error: pipe failed\n";
        return 1;
    }

    pid_t child_pid = fork();
    if(child_pid < 0)
    {
        std::cout << "error: fork failed\n";
        return 1;
    }

    if(child_pid == 0)
    {
        // Child: spawn a grandchild before replacing itself with the test app.
        // The grandchild inherits child's PID as its PPID, which persists
        // across the child's exec, so collect_process_tree will find it.
        pid_t grandchild_pid = fork();
        if(grandchild_pid < 0)
        {
            std::cout << "error: grandchild fork failed\n";
            return 1;
        }

        if(grandchild_pid == 0)
        {
            // Grandchild: close both pipe ends — it doesn't use the pipe.
            close(pipefd[0]);
            close(pipefd[1]);

            // Grandchild: exec the test application.
            std::cout << "grandchild executing " << argv[1] << std::endl;
            execl(argv[1], argv[1], nullptr);
            std::cout << "error in grandchild execl(), errno=" << errno << std::endl;
            return 1;
        }

        // Child: send the grandchild PID to the parent via the pipe, then exec.
        close(pipefd[0]);
        if(write(pipefd[1], &grandchild_pid, sizeof(grandchild_pid)) !=
           static_cast<ssize_t>(sizeof(grandchild_pid)))
        {
            std::cout << "error: failed to write grandchild PID to pipe\n";
            return 1;
        }
        close(pipefd[1]);

        // Child: also exec the test application. The grandchild's PPID
        // remains this PID across the exec.
        std::cout << "child executing " << argv[1] << std::endl;
        execl(argv[1], argv[1], nullptr);
        std::cout << "error in child execl(), errno=" << errno << std::endl;
        return 1;
    }

    // Parent: close write end and read grandchild PID from pipe.
    close(pipefd[1]);
    pid_t grandchild_pid = -1;
    if(read(pipefd[0], &grandchild_pid, sizeof(grandchild_pid)) !=
       static_cast<ssize_t>(sizeof(grandchild_pid)))
    {
        std::cout << "error: failed to read grandchild PID from pipe\n";
        return 1;
    }
    close(pipefd[0]);

    // Parent process: wait for the child and grandchild to be ready for attachment.
    auto kill_children = [&]() {
        kill(grandchild_pid, SIGKILL);
        kill(child_pid, SIGKILL);
        waitpid(grandchild_pid, nullptr, 0);
        waitpid(child_pid, nullptr, 0);
    };

    if(!wait_for_attach_ready(child_pid))
    {
        kill_children();
        return 1;
    }
    if(!wait_for_attach_ready(grandchild_pid))
    {
        kill_children();
        return 1;
    }

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], true);

    // Attach to the child and any processes it may have spawned. Passing the
    // child PID to rocattach_attach_tree lets it discover the full subtree via
    // /proc without the caller enumerating descendants explicitly.
    {
        std::cout << "starting call to rocattach_attach_tree(child_pid)" << std::endl;
        rocattach_status_t status = rocattach_attach_tree(child_pid);
        if(status != ROCATTACH_STATUS_SUCCESS)
        {
            std::cout << "error: call to rocattach_attach_tree(child_pid) returned non zero status "
                      << status << std::endl;
            kill_children();
            return 1;
        }
        std::cout << "call to rocattach_attach_tree(child_pid) successful" << std::endl;
    }

    // Verify that the tool library was loaded into both the child and the
    // grandchild by inspecting /proc/<pid>/maps. setup() calls
    // rocprofiler_register_attach() synchronously via ptrace and waits for
    // it to return before returning itself, so the library is fully loaded
    // in each target process by the time rocattach_attach_tree() returns.
    const std::string tool_library = argv[2];

    if(!is_library_loaded(child_pid, tool_library))
    {
        std::cout << "error: tool library not found in child pid " << child_pid << " maps\n";
        kill_children();
        return 1;
    }
    std::cout << "verified: child pid " << child_pid << " loaded the tool library\n";

    if(!is_library_loaded(grandchild_pid, tool_library))
    {
        std::cout << "error: tool library not found in grandchild pid " << grandchild_pid
                  << " maps\n";
        kill_children();
        return 1;
    }
    std::cout << "verified: grandchild pid " << grandchild_pid << " loaded the tool library\n";

    // Let profiling run for a few seconds.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Detach from the process tree symmetrically with how we attached.
    {
        std::cout << "starting call to rocattach_detach_tree(child_pid)" << std::endl;
        rocattach_status_t status = rocattach_detach_tree(child_pid);
        if(status != ROCATTACH_STATUS_SUCCESS)
        {
            std::cout << "error: call to rocattach_detach_tree(child_pid) returned non zero status "
                      << status << std::endl;
            kill_children();
            return 1;
        }
        std::cout << "call to rocattach_detach_tree(child_pid) successful" << std::endl;
    }

    int grandchild_status = 0;
    kill(grandchild_pid, SIGINT);
    waitpid(grandchild_pid, &grandchild_status, 0);

    int child_status = 0;
    kill(child_pid, SIGINT);
    waitpid(child_pid, &child_status, 0);

    if(grandchild_status != 0)
    {
        std::cout << "error: grandchild process returned non-zero status: " << grandchild_status
                  << std::endl;
        return 1;
    }

    if(child_status != 0)
    {
        std::cout << "error: child process returned non-zero status: " << child_status << std::endl;
        return 1;
    }

    return 0;
}
