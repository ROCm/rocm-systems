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

#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#define ROCATTACH_CALL(func)                                                                       \
    {                                                                                              \
        rocattach_status_t status = func;                                                          \
        if(status != ROCATTACH_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            std::cout << "error: call to " #func " returned non zero status " << status            \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            std::cout << "call to " #func " successful " << std::endl;                             \
        }                                                                                          \
    }

int
main(int argc, char** argv)
{
    if(argc != 3)
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
        // Wait a small amount of time for child processes to start executing
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], true);

        ROCATTACH_CALL(rocattach_attach(pid1));
        ROCATTACH_CALL(rocattach_attach(pid2));

        // Wait a small amount of time for child processes to continue executing
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));

        ROCATTACH_CALL(rocattach_detach(pid1));
        ROCATTACH_CALL(rocattach_detach(pid2));

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
