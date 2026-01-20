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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk-rocattach/types.h>

#include <sys/ptrace.h>

#include <atomic>
#include <mutex>
#include <thread>

namespace rocprofiler
{
namespace rocattach
{
// ptrace sessions are tracked per thread ID, which means all ptrace operations for a single PID
// must originate from the same thread in our program. This class allows multiple threads (e.g. a
// main thread and a signal handler thread) to both use ptrace by routing their calls through a
// single worker thread.
class PTraceRunner
{
    static constexpr size_t DEFAULT_TIMEOUT_MS = 10000;

public:
    explicit PTraceRunner(pid_t pid);
    ~PTraceRunner();

    // Intended to mirror ptrace(), but with parameters for its return value and errno
    // A return value of ROCATTACH_STATUS_ERROR indicates a timeout while communicating with this
    // class's worker thread.
    rocattach_status_t ptrace_run(__ptrace_request op,
                                  void*            addr,
                                  void*            data,
                                  uint64_t*        ptrace_retval,
                                  int*             ptrace_errno,
                                  size_t           timeout_ms = DEFAULT_TIMEOUT_MS);

    pid_t get_pid() { return m_pid; };

private:
    // Data for a single ptrace operation.
    // ptrace_run fills in op, addr, and data when invoking ptrace
    // ptrace_runner worker thread fills in retval and ptrace_errno after running ptrace
    struct ptrace_data_t
    {
        __ptrace_request op;
        void*            addr;
        void*            data;
        uint64_t         retval;
        int              ptrace_errno;
    };

    const pid_t m_pid = 0;
    // Mutex controls access to m_ptrace_data, as well as ensuring ptrace_run is not run
    // concurrently.
    std::mutex m_ptrace_run_mutex;
    // Mailbox containing ptrace parameters and results
    std::atomic<ptrace_data_t> m_ptrace_data = {};

    // Controls the m_ptrace_data mailbox and signals when data is valid
    // Transition to true - Signals to ptrace_runner to invoke ptrace using the information in
    // m_ptrace_data Transition to false - Signals to ptrace_run that ptrace was invoked and results
    // are in m_ptrace_data
    std::atomic<bool> m_running = false;

    // Transition to true - Signals to ptrace_runner to terminate the thread
    std::atomic<bool> m_thread_done = false;

    // This definition must appear after m_ptrace_data, m_running, and m_thread_done to ensure
    // initialization ordering.
    std::thread m_ptrace_thread;

    static void ptrace_runner(pid_t                       _pid,
                              std::atomic<ptrace_data_t>& ptrace_data,
                              std::atomic<bool>&          running,
                              std::atomic<bool>&          thread_done);
};
}  // namespace rocattach
}  // namespace rocprofiler
