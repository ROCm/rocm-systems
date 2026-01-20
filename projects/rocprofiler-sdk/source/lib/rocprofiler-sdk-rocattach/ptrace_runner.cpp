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

#include "ptrace_runner.hpp"

#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace rocattach
{
namespace
{
template <typename T>
bool
wait_for_ne(std::atomic<T>& flag, T condition, size_t timeout_ms)
{
    auto start_time       = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::milliseconds(timeout_ms);
    auto end_time         = start_time + timeout_duration;

    while(std::chrono::steady_clock::now() < end_time)
    {
        if(flag.load() != condition)
        {
            return true;
        }
        std::this_thread::yield();
    }
    // Last chance check in case we were scheduled after timeout
    return flag.load() != condition;
}
}  // namespace

PTraceRunner::PTraceRunner(pid_t _pid)
: m_pid(_pid)
, m_ptrace_data{}
, m_running(false)
, m_thread_done(false)
, m_ptrace_thread(ptrace_runner,
                  _pid,
                  std::ref(m_ptrace_data),
                  std::ref(m_running),
                  std::ref(m_thread_done))
{}

PTraceRunner::~PTraceRunner()
{
    m_thread_done = true;
    m_ptrace_thread.join();
}

rocattach_status_t
PTraceRunner::ptrace_run(__ptrace_request op,
                         void*            addr,
                         void*            data,
                         uint64_t*        ptrace_retval,
                         int*             ptrace_errno,
                         size_t           timeout_ms)
{
    std::lock_guard<std::mutex> lg(m_ptrace_run_mutex);
    ptrace_data_t               ptrace_data{};
    ptrace_data.op   = op;
    ptrace_data.addr = addr;
    ptrace_data.data = data;
    m_ptrace_data.store(ptrace_data);
    m_running.store(true);
    if(!wait_for_ne(m_running, true, timeout_ms))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Timeout during ptrace(" << op << ", " << m_pid
                   << ", " << addr << ", " << data << ") duration: " << timeout_ms << "ms";
        return ROCATTACH_STATUS_ERROR;
    }

    if(ptrace_retval)
    {
        *ptrace_retval = m_ptrace_data.load().retval;
    }
    if(ptrace_errno)
    {
        *ptrace_errno = m_ptrace_data.load().ptrace_errno;
    }
    return ROCATTACH_STATUS_SUCCESS;
}

void
PTraceRunner::ptrace_runner(pid_t                       _pid,
                            std::atomic<ptrace_data_t>& ptrace_data,
                            std::atomic<bool>&          running,
                            std::atomic<bool>&          thread_done)
{
    while(thread_done.load() == false)
    {
        if(running.load() == true)
        {
            errno             = 0;
            auto _ptrace_data = ptrace_data.load();
            auto retval       = ptrace(_ptrace_data.op, _pid, _ptrace_data.addr, _ptrace_data.data);
            _ptrace_data.retval       = retval;
            _ptrace_data.ptrace_errno = errno;
            ptrace_data.store(_ptrace_data);
            running.store(false);
        }
        std::this_thread::yield();
    }
}

}  // namespace rocattach
}  // namespace rocprofiler
