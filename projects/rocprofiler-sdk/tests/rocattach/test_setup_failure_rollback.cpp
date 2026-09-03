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
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>

namespace
{
void
kill_and_reap(pid_t pid)
{
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}
}  // namespace

int
main()
{
    int ready_pipe[2];
    if(pipe(ready_pipe) != 0)
    {
        std::cerr << "Test FAILED: pipe creation failed\n";
        return 1;
    }

    auto target_pid = fork();
    if(target_pid < 0)
    {
        std::cerr << "Test FAILED: target fork failed\n";
        return 1;
    }
    if(target_pid == 0)
    {
        close(ready_pipe[0]);
        if(prctl(PR_SET_NAME, "rocp-bg-attach", 0, 0, 0) != 0) _exit(1);
        auto ready = char{1};
        if(write(ready_pipe[1], &ready, sizeof(ready)) != static_cast<ssize_t>(sizeof(ready)))
            _exit(1);
        close(ready_pipe[1]);
        while(true)
            pause();
    }

    close(ready_pipe[1]);
    auto ready = char{};
    if(read(ready_pipe[0], &ready, sizeof(ready)) != static_cast<ssize_t>(sizeof(ready)))
    {
        std::cerr << "Test FAILED: listener handshake failed\n";
        close(ready_pipe[0]);
        kill_and_reap(target_pid);
        return 1;
    }
    close(ready_pipe[0]);

    // The fake listener lets setup() create and ptrace-attach a session, but the target
    // intentionally has no rocprofiler-register mapping, so symbol lookup fails afterward.
    auto first_status = rocattach_attach(target_pid);
    if(first_status == ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "Test FAILED: attachment unexpectedly succeeded\n";
        rocattach_detach(target_pid);
        kill_and_reap(target_pid);
        return 1;
    }

    // A failed setup must remove its session. If it leaked, detach would find the entry and
    // a retry would be rejected as an already-active attachment.
    if(rocattach_detach(target_pid) != ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT)
    {
        std::cerr << "Test FAILED: failed setup left an attachment session\n";
        kill_and_reap(target_pid);
        return 1;
    }

    auto retry_status = rocattach_attach(target_pid);
    if(retry_status == ROCATTACH_STATUS_SUCCESS ||
       retry_status == ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT)
    {
        std::cerr << "Test FAILED: retry did not perform a fresh setup\n";
        if(retry_status == ROCATTACH_STATUS_SUCCESS) rocattach_detach(target_pid);
        kill_and_reap(target_pid);
        return 1;
    }

    kill_and_reap(target_pid);
    std::cout << "Test PASSED: failed setup rolled back its attachment session\n";
    return 0;
}
