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
#include <sstream>
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

int
main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "error: wrong number of arguments\n";
        return 1;
    }

    pid_t pid1 = fork();
    if(pid1 < 0)
    {
        std::cout << "error: Fork 1 failed.\n";
        return 1;
    }

    pid_t pid2 = 0;
    if(pid1 > 0)
    {
        // Parent process, will fork again to spawn 2 processes
        pid2 = fork();
    }

    if(pid2 < 0)
    {
        std::cout << "error: Fork 2 failed.\n";
        return 1;
    }

    if(pid1 == 0 || pid2 == 0)
    {
        // Child process
        std::cout << "child executing " << argv[1] << std::endl;
        int ret = execl(argv[1], argv[1], nullptr);
        if(ret == -1)
        {
            std::cout << "error in execl(), errno=" << errno << std::endl;
            return 1;
        }
    }
    else
    {
        auto kill_children = [&]() {
            kill(pid1, SIGKILL);
            kill(pid2, SIGKILL);
            waitpid(pid1, nullptr, 0);
            waitpid(pid2, nullptr, 0);
        };

        // Wait for child processes to be ready for attachment
        if(!wait_for_attach_ready(pid1))
        {
            kill_children();
            return 1;
        }
        if(!wait_for_attach_ready(pid2))
        {
            kill_children();
            return 1;
        }

        setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], true);

        {
            std::cout << "starting call to rocattach_attach(pid1)" << std::endl;
            rocattach_status_t status = rocattach_attach(pid1);
            if(status != ROCATTACH_STATUS_SUCCESS)
            {
                std::cout << "error: call to rocattach_attach(pid1) returned non zero status "
                          << status << std::endl;
                kill_children();
                return 1;
            }
            std::cout << "call to rocattach_attach(pid1) successful" << std::endl;
        }
        {
            std::cout << "starting call to rocattach_attach(pid2)" << std::endl;
            rocattach_status_t status = rocattach_attach(pid2);
            if(status != ROCATTACH_STATUS_SUCCESS)
            {
                std::cout << "error: call to rocattach_attach(pid2) returned non zero status "
                          << status << std::endl;
                kill_children();
                return 1;
            }
            std::cout << "call to rocattach_attach(pid2) successful" << std::endl;
        }

        // Send signal to child processes after attaching
        if(argc >= 4)
        {
            std::string signal_arg{argv[3]};
            if(signal_arg == "--send-signal")
            {
                std::cout << "Sending SIGWINCH to PID " << pid1 << "\n";
                if(kill(pid1, SIGWINCH) == -1)
                {
                    std::cout << "error: Failed to send SIGWINCH signal to pid1\n";
                }
                std::cout << "Sending SIGWINCH to PID " << pid2 << "\n";
                if(kill(pid2, SIGWINCH) == -1)
                {
                    std::cout << "error: Failed to send SIGWINCH signal to pid2\n";
                }
            }
        }

        // Wait for child processes to continue executing
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        {
            std::cout << "starting call to rocattach_detach(pid1)" << std::endl;
            rocattach_status_t status = rocattach_detach(pid1);
            if(status != ROCATTACH_STATUS_SUCCESS)
            {
                std::cout << "error: call to rocattach_detach(pid1) returned non zero status "
                          << status << std::endl;
                kill_children();
                return 1;
            }
            std::cout << "call to rocattach_detach(pid1) successful" << std::endl;
        }
        {
            std::cout << "starting call to rocattach_detach(pid2)" << std::endl;
            rocattach_status_t status = rocattach_detach(pid2);
            if(status != ROCATTACH_STATUS_SUCCESS)
            {
                std::cout << "error: call to rocattach_detach(pid2) returned non zero status "
                          << status << std::endl;
                kill_children();
                return 1;
            }
            std::cout << "call to rocattach_detach(pid2) successful" << std::endl;
        }

        if(kill(pid1, SIGINT) == -1)
        {
            std::cout << "error: Failed to send signal SIGINT to pid1\n";
        }
        if(kill(pid2, SIGINT) == -1)
        {
            std::cout << "error: Failed to send signal SIGINT to pid2\n";
        }

        int pid1status = 0;
        waitpid(pid1, &pid1status, 0);
        int pid2status = 0;
        waitpid(pid2, &pid2status, 0);

        if(pid1status != 0)
        {
            std::cout << "error in pid1, returned non-zero status: " << pid1status;
            return 1;
        }
        if(pid2status != 0)
        {
            std::cout << "error in pid2, returned non-zero status: " << pid2status;
            return 1;
        }
    }
    return 0;
}
