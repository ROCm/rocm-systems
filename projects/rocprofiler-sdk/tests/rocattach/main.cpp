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

    // create pipes to capture output before forking
    int pid1link[2];
    int pid2link[2];

    if(pipe(pid1link) == -1)
    {
        std::cout << "error: Pipe 1 failed.\n";
        return 0;
    }

    if(pipe(pid2link) == -1)
    {
        std::cout << "error: Pipe 2 failed.\n";
        return 0;
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
        if(pid1 == 0)
        {
            dup2(pid1link[1], STDOUT_FILENO);
            dup2(pid1link[1], STDERR_FILENO);
        }
        else if(pid2 == 0)
        {
            dup2(pid2link[1], STDOUT_FILENO);
            dup2(pid2link[1], STDERR_FILENO);
        }
        close(pid1link[0]);
        close(pid1link[1]);
        close(pid2link[0]);
        close(pid2link[1]);

        const char* extra_env[] = {
            "ROCPROFILER_REGISTER_LOG_LEVEL=trace",
            "ROCPROFILER_LOG_LEVEL=trace",
            nullptr,  // array is null terminated per exec() convention
        };
        // Child process
        std::cout << "child executing " << argv[1] << std::endl;
        int ret = execle(argv[1], argv[1], nullptr, extra_env);
        if(ret == -1)
        {
            std::cout << "error in execl(), errno=" << errno << std::endl;
            return 1;
        }
    }
    else
    {
        // Wait a small amount of time for child processes to start executing
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], true);
        // setenv("ROCATTACH_LOG_LEVEL", "trace", true);

        ROCATTACH_CALL(rocattach_attach(pid1));
        ROCATTACH_CALL(rocattach_attach(pid2));

        // Wait a small amount of time for child processes to continue executing
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        ROCATTACH_CALL(rocattach_detach(pid1));
        ROCATTACH_CALL(rocattach_detach(pid2));

        int pid1status = 0;
        waitpid(pid1, &pid1status, 0);
        int pid2status = 0;
        waitpid(pid2, &pid2status, 0);

        close(pid1link[1]);
        close(pid2link[1]);

        std::stringstream pid1sstream;
        std::stringstream pid2sstream;
        int               readbytes   = 0;
        constexpr size_t  buffer_size = 8192;
        do
        {
            char buffer[buffer_size];
            readbytes = read(pid1link[0], buffer, buffer_size);
            pid1sstream << buffer;
        } while(readbytes > 0);

        do
        {
            char buffer[buffer_size];
            readbytes = read(pid2link[0], buffer, buffer_size);
            pid2sstream << buffer;
        } while(readbytes > 0);

        close(pid1link[0]);
        close(pid2link[0]);

        std::cout << pid1sstream.str() << std::endl;
        std::cout << pid2sstream.str() << std::endl;

        // CMake pass regex search doesn't handle multiline searches, so main has to verify instead.
        const char c_tool_output[] = "Test C tool (priority=0) is using rocprofiler-sdk v";
        if(pid1sstream.str().find(c_tool_output) != std::string::npos &&
           pid2sstream.str().find(c_tool_output) != std::string::npos)
        {
            std::cout << "C tool was loaded in both processes." << std::endl;
        }
    }
    return 0;
}