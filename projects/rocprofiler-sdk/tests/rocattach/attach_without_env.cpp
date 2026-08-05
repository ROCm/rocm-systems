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

// Test that attach fails gracefully when ROCP_TOOL_ATTACH=1 is not set, instead of
// crashing the target application.
//
// Spawns a child process WITHOUT setting ROCP_TOOL_ATTACH=1, then attempts to attach
// using either rocattach_attach() or rocattach_attach_tree() (based on command-line arg).
// The attach should fail with ROCATTACH_STATUS_ERROR and the child process should
// continue running normally (not crash).
//
// Usage: attach_without_env <test_app> [attach|attach-tree]

#include <rocprofiler-sdk-rocattach/defines.h>
#include <rocprofiler-sdk-rocattach/rocattach.h>
#include <rocprofiler-sdk-rocattach/types.h>

#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <thread>

int
main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "error: wrong number of arguments\n";
        std::cout << "usage: " << argv[0] << " <test_application_path> [attach|attach-tree]\n";
        return 1;
    }

    // Determine which attach function to test
    bool use_tree_attach = false;
    if(argc >= 3)
    {
        std::string mode{argv[2]};
        if(mode == "attach-tree")
        {
            use_tree_attach = true;
        }
        else if(mode != "attach")
        {
            std::cout << "error: invalid mode '" << mode
                      << "', expected 'attach' or 'attach-tree'\n";
            return 1;
        }
    }

    // Explicitly unset ROCP_TOOL_ATTACH to ensure the child process
    // does NOT have the attach thread initialized
    unsetenv("ROCP_TOOL_ATTACH");

    pid_t child_pid = fork();
    if(child_pid < 0)
    {
        std::cout << "error: fork failed\n";
        return 1;
    }

    if(child_pid == 0)
    {
        // Child process: exec the test application WITHOUT ROCP_TOOL_ATTACH=1
        std::cout << "child executing " << argv[1] << " (without ROCP_TOOL_ATTACH)\n";
        int ret = execl(argv[1], argv[1], nullptr);
        if(ret == -1)
        {
            std::cout << "error in execl(), errno=" << errno << std::endl;
            _exit(1);  // Use _exit() after fork, not return
        }
    }
    else
    {
        // Parent process: heuristic wait for the child to execl(). This is a synchronization
        // hack: if the child has not exec'd yet, the /proc scan can find no bg-attach thread
        // and the test may pass for the wrong reason, since that is the expected outcome here.
        // A pipe handshake would be more robust if this test becomes flaky or more critical.
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));

        const char* attach_mode_str =
            use_tree_attach ? "rocattach_attach_tree" : "rocattach_attach";
        std::cout << "attempting to attach to pid " << child_pid << " using " << attach_mode_str
                  << " (should fail with ROCATTACH_STATUS_ERROR)\n";

        // Attempt to attach - this should FAIL because bg-attach thread doesn't exist
        //
        // NOTE: for the attach-tree mode, this test spawns a single child with no descendants,
        // so rocattach_attach_tree() reduces to one setup() call and the returned last_status
        // is deterministically ROCATTACH_STATUS_ERROR. If this test is extended to a real
        // process tree, revisit this expectation because last_status only reflects the last
        // failure encountered.
        rocattach_status_t status =
            use_tree_attach ? rocattach_attach_tree(child_pid) : rocattach_attach(child_pid);

        // Check attach result
        if(status == ROCATTACH_STATUS_SUCCESS)
        {
            std::cout << "error: attach succeeded when it should have failed!\n";
            std::cout << "The fix is not working correctly.\n";
            if(use_tree_attach)
                rocattach_detach_tree(child_pid);
            else
                rocattach_detach(child_pid);
            kill(child_pid, SIGKILL);
            waitpid(child_pid, nullptr, 0);
            return 1;
        }

        if(status != ROCATTACH_STATUS_ERROR)
        {
            std::cout << "error: unexpected attach status " << status << "\n";
            std::cout << "Expected ROCATTACH_STATUS_ERROR (1)\n";
            kill(child_pid, SIGKILL);
            waitpid(child_pid, nullptr, 0);
            return 1;
        }

        std::cout << "SUCCESS: attach failed as expected with ROCATTACH_STATUS_ERROR\n";

        // Check if child process is still running (reliable check using waitpid WNOHANG)
        int  child_status = 0;
        auto wait_ret     = waitpid(child_pid, &child_status, WNOHANG);

        if(wait_ret == 0)
        {
            // Child is still running
            std::cout << "VERIFIED: child process " << child_pid << " is still running\n";
            std::cout << "Test PASSED: attach failed gracefully without crashing target\n";

            // Clean up: kill the child
            kill(child_pid, SIGTERM);
            waitpid(child_pid, nullptr, 0);
            return 0;
        }
        else if(wait_ret == child_pid)
        {
            // Child exited
            std::cout << "Test FAILED: child process " << child_pid
                      << " exited during failed attach\n";
            if(WIFSIGNALED(child_status))
            {
                std::cout << "ERROR: child exited due to signal " << WTERMSIG(child_status);
                if(WTERMSIG(child_status) == SIGSEGV)
                {
                    std::cout << " (SIGSEGV - segmentation fault)";
                }
                std::cout << "\n";
            }
            else if(WIFEXITED(child_status))
            {
                std::cout << "ERROR: child exited with status " << WEXITSTATUS(child_status)
                          << "\n";
            }
            else
            {
                std::cout << "ERROR: child exited unexpectedly\n";
            }
            return 1;
        }
        else
        {
            std::cout << "Test FAILED: waitpid failed, errno=" << errno << "\n";
            kill(child_pid, SIGKILL);
            waitpid(child_pid, nullptr, 0);
            return 1;
        }
    }

    return 0;
}
