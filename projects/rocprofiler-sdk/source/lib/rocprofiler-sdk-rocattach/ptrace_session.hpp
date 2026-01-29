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
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace rocattach
{
class PTraceSession
{
public:
    explicit PTraceSession(int);
    ~PTraceSession();

    static bool is_supported();

    rocattach_status_t attach();
    rocattach_status_t detach();
    rocattach_status_t simple_mmap(void*& addr, size_t length);
    rocattach_status_t simple_munmap(void*& addr, size_t length);

    rocattach_status_t write(size_t addr, const std::vector<uint8_t>& data, size_t size);
    rocattach_status_t read(size_t addr, std::vector<uint8_t>& data, size_t size);
    rocattach_status_t swap(size_t                      addr,
                            const std::vector<uint8_t>& in_data,
                            std::vector<uint8_t>&       out_data,
                            size_t                      size);

    int get_pid() const { return m_pid; }

    rocattach_status_t call_function(const std::string& library,
                                     const std::string& symbol,
                                     uint64_t&          ret_value);
    rocattach_status_t call_function(const std::string& library,
                                     const std::string& symbol,
                                     uint64_t&          ret_value,
                                     void*              first);
    rocattach_status_t call_function(const std::string& library,
                                     const std::string& symbol,
                                     uint64_t&          ret_value,
                                     void*              first,
                                     void*              second);

private:
    rocattach_status_t start_signal_handler();
    rocattach_status_t stop_signal_handler();
    rocattach_status_t wait_for_breakpoint();
    rocattach_status_t wait_for_stop();

    rocattach_status_t stop();
    rocattach_status_t cont();
    rocattach_status_t write_internal(size_t                      addr,
                                      const std::vector<uint8_t>& data,
                                      size_t                      size) const;
    rocattach_status_t read_internal(size_t addr, std::vector<uint8_t>& data, size_t size) const;
    rocattach_status_t swap_internal(size_t                      addr,
                                     const std::vector<uint8_t>& in_data,
                                     std::vector<uint8_t>&       out_data,
                                     size_t                      size) const;

    enum ptrace_session_state_t
    {
        // When the session class is created, this is the initial state
        // Can transition to attached by calling attach()
        PTRACE_SESSION_STATE_INITIAL = 0,
        // State after attach()
        // Most functions can be run in this state
        // Can transition to stopped by calling stop()
        // Can transition to detached by calling detach()
        PTRACE_SESSION_STATE_RUNNING,
        // State after stop()
        // Required state for some internal functions, like write_internal
        // Can transition to running by calling cont()
        PTRACE_SESSION_STATE_STOPPED,
        // State after detach()
        // Class is left to be cleaned up
        PTRACE_SESSION_STATE_DETACHED
    };

    // This is the state of the signal handler thread. This thread intercepts signals that the
    // attachee is sent. Based on the current state, these signals are either consumed by the
    // attacher or forwarded to the attachee. Typically, the main thread updates this state. The
    // main thread can update this state only if the program is stopped (i.e. stop() has been
    // called). The main thread must also verify the state is not FINAL using a check-and-set
    // operation. The signal handler will initially transition the state from INITIAL to ATTACHED.
    // Otherwise, the signal handler update thread will only update this state to FINAL. This occurs
    // when the target program ends, an unexpected error occurs, or if a stop has been requested by
    // the main program (by transitioning this state to DETACHING). The main thread can check the
    // error state for more information.
    enum ptrace_session_signal_handler_state_t
    {
        PTRACE_SIGNAL_HANDLER_STATE_INITIAL = 0,
        // The signal handler is running normally. All signals are forwarded to the attachee.
        PTRACE_SIGNAL_HANDLER_STATE_ATTACHED,
        // The signal handler is running normally. A WUNTRACED stop will transition the state to
        // ATTACHED. All other signals are forwarded to the attachee.
        PTRACE_SIGNAL_HANDLER_STATE_WAITING_FOR_BREAKPOINT,
        // The signal handler will end operation. State is transitioned to FINAL when complete.
        PTRACE_SIGNAL_HANDLER_STATE_DETACHING,
        // The signal handler is not running, either by request or because some error has occurred.
        // If an error occurred, more information will be in m_signal_handler_error
        PTRACE_SIGNAL_HANDLER_STATE_FINAL
    };

    static void ptrace_signal_handler_func(
        int                                                 pid,
        std::atomic<ptrace_session_signal_handler_state_t>& state,
        std::atomic<rocattach_status_t>&                    error);

    ptrace_session_state_t                             m_state = PTRACE_SESSION_STATE_INITIAL;
    std::atomic<ptrace_session_signal_handler_state_t> m_ptrace_signal_handler_state =
        PTRACE_SIGNAL_HANDLER_STATE_INITIAL;
    std::atomic<rocattach_status_t> m_ptrace_signal_handler_error = ROCATTACH_STATUS_SUCCESS;

    std::thread m_ptrace_signal_handler_thread;

    const int m_pid = -1;
};

}  // namespace rocattach
}  // namespace rocprofiler
