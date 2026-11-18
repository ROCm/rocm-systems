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

// Attach, detach and attach again to one process through rocattach, with a tool that finalizes
// itself from tool_detach (the rocprof-sys-attach flow). Every attach must succeed.

#include <rocprofiler-sdk-rocattach/rocattach.h>

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

namespace
{
bool
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
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "usage: " << argv[0] << " <target application> <tool library>\n";
        return 1;
    }

    // a parent may ptrace its child under ptrace_scope=1, so the target is a child process
    pid_t pid = fork();
    if(pid < 0)
    {
        std::cout << "error: fork failed\n";
        return 1;
    }
    if(pid == 0)
    {
        execl(argv[1], argv[1], nullptr);
        std::cout << "error in execl(), errno=" << errno << std::endl;
        _exit(1);
    }

    auto stop_target = [pid](int sig) {
        kill(pid, sig);
        int status = 0;
        waitpid(pid, &status, 0);
        return status;
    };

    if(!wait_for_attach_ready(pid))
    {
        std::cout << "error: target never became ready for attachment\n";
        stop_target(SIGKILL);
        return 1;
    }

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], 1);

    constexpr int sessions = 3;
    for(int session = 1; session <= sessions; ++session)
    {
        if(auto status = rocattach_attach(pid); status != ROCATTACH_STATUS_SUCCESS)
        {
            std::cout << "error: attach " << session << " returned status " << status << "\n";
            stop_target(SIGKILL);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if(auto status = rocattach_detach(pid); status != ROCATTACH_STATUS_SUCCESS)
        {
            std::cout << "error: detach " << session << " returned status " << status << "\n";
            stop_target(SIGKILL);
            return 1;
        }
        std::cout << "attach/detach session " << session << " succeeded" << std::endl;
    }

    if(auto status = stop_target(SIGINT); status != 0)
    {
        std::cout << "error: target exited with status " << status << "\n";
        return 1;
    }

    std::cout << "Test PASSED: reattached " << sessions << " times with a self-finalizing tool\n";
    return 0;
}
