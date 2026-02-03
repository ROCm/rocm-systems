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
#include "common/ptrace.hpp"
#include "common/wait_for_atomic.hpp"

#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace rocattach
{
namespace
{}  // namespace

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
PTraceRunner::ptrace_run(__ptrace_request   op,
                         ptrace_parameter_t _addr,
                         ptrace_parameter_t _data,
                         uint64_t*          ptrace_retval,
                         int*               ptrace_errno,
                         size_t             timeout_ms)
{
    std::lock_guard<std::mutex> lg(m_ptrace_run_mutex);

    // This does some work with variants to allow functions to send ptrace operations as if they
    // were addressing the original ptrace function, that is without strict typing. This is only
    // slightly safer than using a union, but it is no worse than ptrace's type punning. For our
    // sake, internally, we address these as uint64_t blobs to make for easier logging and typing on
    // our end.
    auto convert_ptrace_parameter = [](ptrace_parameter_t param) {
        if(std::holds_alternative<uint64_t>(param))
        {
            return std::get<uint64_t>(param);
        }
        else
        {
            return reinterpret_cast<uint64_t>(std::get<void*>(param));
        }
    };
    uint64_t addr = convert_ptrace_parameter(_addr);
    uint64_t data = convert_ptrace_parameter(_data);

    ROCP_TRACE << "[rocprofiler-sdk-rocattach] ptrace call params(" << ptrace_op_name(op) << "("
               << op << "), " << m_pid << ", " << addr << ", " << data << ")";

    ptrace_data_t ptrace_data{};
    ptrace_data.op   = op;
    ptrace_data.addr = addr;
    ptrace_data.data = data;
    // Store parameters for the ptrace operation in m_ptrace_data for retrieval by ptrace_runner()
    m_ptrace_data.store(ptrace_data);

    // Set m_running to true, requesting a ptrace operation be performed in ptrace_runner() with the
    // parameters in m_ptrace_data
    m_running.store(true);
    // Wait for m_running to be set to false, which indicates ptrace_runner() has finished the
    // requested ptrace operation.
    if(!wait_for_eq(m_running, false, timeout_ms))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Timeout during ptrace(" << op << ", " << m_pid
                   << ", " << addr << ", " << data << ") duration: " << timeout_ms << "ms";
        return ROCATTACH_STATUS_ERROR;
    }
    auto result = m_ptrace_data.load();
    if(ptrace_retval)
    {
        *ptrace_retval = result.retval;
    }
    if(ptrace_errno)
    {
        *ptrace_errno = result.ptrace_errno;
    }
    if(result.ptrace_errno != 0)
    {
        // log an error if it occurs, but the ptrace call was a success, so still return success
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] ptrace call failed. errno: "
                   << result.ptrace_errno << " - " << strerror(result.ptrace_errno) << ". params("
                   << ptrace_op_name(op) << "(" << op << "), " << m_pid << ", " << addr << ", "
                   << data << ")";
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
        // When running becomes true, a new ptrace operation has been requested with ptrace_data as
        // its parameters
        if(running.load() == true)
        {
            errno = 0;
            // Load ptrace_data, then call ptrace with its parameters.
            auto _ptrace_data = ptrace_data.load();
            auto retval       = ptrace(_ptrace_data.op, _pid, _ptrace_data.addr, _ptrace_data.data);
            // Write back the results of the ptrace operation, then store to ptrace_data.
            _ptrace_data.retval       = retval;
            _ptrace_data.ptrace_errno = errno;
            ptrace_data.store(_ptrace_data);
            // Clear running, indicating the requested operation is complete.
            running.store(false);
        }
        std::this_thread::yield();
    }
}

}  // namespace rocattach
}  // namespace rocprofiler
