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

// Attach, detach and attach again to one process through rocattach, with a tool
// (tests/tools/reattach-tool.cpp) that finalizes itself from tool_detach: the rocprof-sys-attach
// flow. Every attach must succeed and the target must exit cleanly afterwards. The tool reports
// its own checks in the target's output ("reattach-tool[...]: FAIL: ..." and "tool_fini after N
// session(s)").

#include "attach_test_utils.hpp"

#include <rocprofiler-sdk-rocattach/rocattach.h>

#include <signal.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

namespace
{
// wait status of the target once it has been reaped
std::optional<int> target_status = {};

// returns the target's wait status once it has exited, without blocking
std::optional<int>
target_exited(pid_t pid)
{
    int status = 0;
    if(!target_status && waitpid(pid, &status, WNOHANG) == pid) target_status = status;
    return target_status;
}

std::string
describe_status(int status)
{
    if(WIFEXITED(status)) return "exited with code " + std::to_string(WEXITSTATUS(status));
    if(WIFSIGNALED(status))
        return std::string{"was killed by signal "} + std::to_string(WTERMSIG(status)) + " (" +
               strsignal(WTERMSIG(status)) + ")";
    return "ended with wait status " + std::to_string(status);
}

// sends `sig` to the target and returns its wait status, or nullopt if waiting failed
std::optional<int>
stop_target(pid_t pid, int sig)
{
    if(auto status = target_exited(pid)) return status;
    kill(pid, sig);

    int status = 0;
    while(waitpid(pid, &status, 0) < 0)
    {
        if(errno == EINTR) continue;
        std::cout << "error: waitpid failed: " << strerror(errno) << "\n";
        return std::nullopt;
    }
    target_status = status;
    return status;
}

int
fail(pid_t pid, const std::string& msg)
{
    std::cout << "error: " << msg << "\n";
    if(auto status = stop_target(pid, SIGKILL))
        std::cout << "target " << describe_status(*status) << "\n";
    std::cout << "Test FAILED" << std::endl;
    return 1;
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
        std::cout << "error: fork failed: " << strerror(errno) << "\nTest FAILED" << std::endl;
        return 1;
    }
    if(pid == 0)
    {
        // do not leave the target running (and holding GPUs) if this driver dies
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        execl(argv[1], argv[1], nullptr);
        std::cout << "error in execl(), errno=" << errno << std::endl;
        _exit(1);
    }

    if(!wait_for_attach_ready(pid)) return fail(pid, "target never became ready for attachment");

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", argv[2], 1);

    constexpr int sessions = 3;
    for(int session = 1; session <= sessions; ++session)
    {
        if(auto status = rocattach_attach(pid); status != ROCATTACH_STATUS_SUCCESS)
            return fail(
                pid,
                "attach " + std::to_string(session) + " returned status " + std::to_string(status));
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if(auto status = rocattach_detach(pid); status != ROCATTACH_STATUS_SUCCESS)
            return fail(
                pid,
                "detach " + std::to_string(session) + " returned status " + std::to_string(status));
        if(auto status = target_exited(pid))
            return fail(pid,
                        "target " + describe_status(*status) + " during session " +
                            std::to_string(session));
        std::cout << "attach/detach session " << session << " succeeded" << std::endl;
    }

    // SIGINT makes attachment-test return from main, so the SDK finalizes at exit
    auto status = stop_target(pid, SIGINT);
    if(!status) return fail(pid, "could not wait for the target");
    if(!WIFEXITED(*status) || WEXITSTATUS(*status) != 0)
    {
        std::cout << "error: target " << describe_status(*status) << "\nTest FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test PASSED: reattached " << sessions << " times with a self-finalizing tool\n";
    return 0;
}
