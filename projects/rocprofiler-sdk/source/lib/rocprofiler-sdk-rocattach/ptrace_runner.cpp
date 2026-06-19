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
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"

#include <memory>

namespace rocprofiler
{
namespace rocattach
{
namespace
{
struct ptrace_data_t
{
    pid_t            pid;
    __ptrace_request op;
    uint64_t         addr;
    uint64_t         data;
    uint64_t         retval;
    int              ptrace_errno;
};

void
ptrace_runner(pid_t                       _pid,
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

class PTraceRunner
{
    struct ptrace_runner_data_t
    {
        // Shared data for the ptrace operation
        std::atomic<ptrace_data_t> data = {};

        // Shared flag, indicates the runner thread should operate on data
        // Set by run when data is ready
        // Cleared by ptrace_runner when operation is complete
        std::atomic<bool> running = {};

        // Shared flag, indicates the runner thread should end
        // Set by run after a PTRACE_DETACH operation is completed
        // Set by PTraceRunner's destructor (called on program exit)
        std::atomic<bool> done = {};

        // Runner thread
        std::thread thread = {};
    };
    using ptrace_runner_t = common::Synchronized<ptrace_runner_data_t>;
    using runner_map_t =
        common::Synchronized<std::unordered_map<int, std::shared_ptr<ptrace_runner_t>>>;
    runner_map_t m_runners;

public:
    PTraceRunner() = default;
    ~PTraceRunner()
    {
        m_runners.wlock([&](auto& all_runners) {
            for(auto& runner : all_runners)
            {
                runner.second->wlock([&](auto& data) {
                    data.done.store(true);
                    data.thread.join();
                });
            }
            all_runners.clear();
        });
    }
    rocattach_status_t run(pid_t              pid,
                           __ptrace_request   op,
                           ptrace_parameter_t _addr,
                           ptrace_parameter_t _data,
                           uint64_t*          ptrace_retval,
                           int*               ptrace_errno,
                           size_t             timeout_ms)
    {
        // This does some work with variants to allow functions to send ptrace operations as if they
        // were addressing the original ptrace function, that is without strict typing. This is only
        // slightly safer than using a union, but it is no worse than ptrace's type punning. For our
        // sake, internally, we address these as uint64_t blobs to make for easier logging and
        // typing on our end.
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
                   << op << "), " << pid << ", " << addr << ", " << data << ")";

        // Get the thread data for the target pid
        // Create it if this is the first operation
        // A shared_ptr is used so this thread can still safely wait on the mutex even if another
        // thread ends the worker thread
        std::shared_ptr<ptrace_runner_t> runner;
        m_runners.wlock([&](auto& runners) {
            if(runners.count(pid) > 0)
            {
                // if thread already exists, reuse it
                runner = runners.at(pid);
            }
            else if(op == PTRACE_SEIZE)
            {
                // if thread doesn't exist and this is a SEIZE, create it
                runner = std::make_shared<ptrace_runner_t>();
                runner->wlock([&](auto& runner_data) {
                    runner_data.thread = std::thread(ptrace_runner,
                                                     pid,
                                                     std::ref(runner_data.data),
                                                     std::ref(runner_data.running),
                                                     std::ref(runner_data.done));
                });

                runners.insert({pid, runner});
            }
            else
            {
                // if thread doesn't exist and this isn't a SEIZE, state is incorrect
                ROCP_ERROR
                    << "[rocprofiler-sdk-rocattach] PTraceRunner was called in an invalid state";
            }
        });

        if(runner == nullptr)
        {
            return ROCATTACH_STATUS_ERROR;
        }

        ptrace_data_t ptrace_data = {
            .pid          = pid,
            .op           = op,
            .addr         = addr,
            .data         = data,
            .retval       = 0,
            .ptrace_errno = 0,
        };

        bool runner_error = false;
        runner->wlock([&](auto& runner_data) {
            // If another thread called detach, the runner thread has been stopped and the
            // ptrace_runner_t has been removed from PTraceRunner's maps. While the ptrace_runner_t
            // is still valid in our shared_ptr copy, we should give up and release our reference
            if(runner_data.done.load())
            {
                ROCP_ERROR
                    << "[rocprofiler-sdk-rocattach] ptrace was detached while another operation "
                       "was in-flight";
                runner_error = true;
                return;
            }

            // Set up the next operation
            runner_data.data.store(ptrace_data);

            // Set running to true, requesting a ptrace operation be performed in ptrace_runner()
            // with the parameters in data
            runner_data.running.store(true);

            // Wait for running to be set to false, which indicates ptrace_runner() has finished the
            // requested ptrace operation.
            if(!wait_for_eq(runner_data.running, false, timeout_ms))
            {
                ROCP_ERROR << "[rocprofiler-sdk-rocattach] Timeout during ptrace(" << op << ", "
                           << pid << ", " << addr << ", " << data << ") duration: " << timeout_ms
                           << "ms";
                runner_error = true;
                return;
            }

            ptrace_data = runner_data.data.load();

            // If detaching, join the thread and remove its references
            if(op == PTRACE_DETACH)
            {
                runner_data.done.store(true);
                runner_data.thread.join();
                m_runners.wlock([&](auto& runners) { runners.erase(pid); });
            }
        });

        if(runner_error)
        {
            return ROCATTACH_STATUS_ERROR;
        }
        if(ptrace_retval)
        {
            *ptrace_retval = ptrace_data.retval;
        }
        if(ptrace_errno)
        {
            *ptrace_errno = ptrace_data.ptrace_errno;
        }
        if(ptrace_data.ptrace_errno != 0)
        {
            // log an error if it occurs, but the ptrace call was a success, so still return success
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] ptrace call failed. errno: "
                       << ptrace_data.ptrace_errno << " - " << strerror(ptrace_data.ptrace_errno)
                       << ". params(" << ptrace_op_name(op) << "(" << op << "), " << pid << ", "
                       << addr << ", " << data << ")";
        }

        return ROCATTACH_STATUS_SUCCESS;
    }
};

}  // namespace

rocattach_status_t
ptrace_run(pid_t              pid,
           __ptrace_request   op,
           ptrace_parameter_t addr,
           ptrace_parameter_t data,
           uint64_t*          ptrace_retval,
           int*               ptrace_errno,
           size_t             timeout)
{
    static auto*& ptrace_runner = common::static_object<PTraceRunner>::construct();

    return ptrace_runner->run(pid, op, addr, data, ptrace_retval, ptrace_errno, timeout);
}

}  // namespace rocattach
}  // namespace rocprofiler
