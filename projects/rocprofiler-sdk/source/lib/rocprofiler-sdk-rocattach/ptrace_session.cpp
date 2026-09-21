// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ptrace_session.hpp"
#include "auxv.hpp"
#include "symbol_lookup.hpp"

#include "common/ptrace.hpp"
#include "common/wait_for_atomic.hpp"

#include "lib/common/logging.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fstream>
#include <functional>
#include <type_traits>

namespace rocprofiler
{
namespace rocattach
{
namespace
{
// Boilerplate around in-class ptrace calls. Macro returns on error!
#define PTRACE_CALL(op, addr, data)                                                                \
    {                                                                                              \
        auto status = ptrace_call(op, addr, data);                                                 \
        if(status != ROCATTACH_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            return status;                                                                         \
        }                                                                                          \
    }

// Helper macro for handling any rocattach_status returning call. Macro returns on error!
#define ROCATTACH_CALL(func)                                                                       \
    {                                                                                              \
        auto status = func;                                                                        \
        if(status != ROCATTACH_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] rocattach call failed. error: " << status   \
                       << ", invocation: " << #func;                                               \
            return status;                                                                         \
        }                                                                                          \
    }

// How long to wait for a process to start or stop after a ptrace operation
// This could wait for a long time due to detach triggering (and blocking for) output generation.
// TODO: When output generation is non-blocking or shorter, reduce this
constexpr size_t PTRACE_BREAKPOINT_TIMEOUT_MS = 1800000;
// How long to wait for the signal handler thread to start or stop
constexpr size_t PTRACE_HANDLER_START_STOP_TIMEOUT_MS = 10000;

}  // namespace

PTraceSession::PTraceSession(int _pid)
: m_pid{_pid}
{}

PTraceSession::~PTraceSession() { detach(); }

bool
PTraceSession::is_supported()
{
    // This file uses x64 assembly which is inherently platform dependent.
#ifdef __x86_64__
    const bool arch_supported = true;
#else
    const bool arch_supported = false;
#endif
    // ptrace memory operations use "word length" which is dependent on system architecture.
    const bool word_size_supported = (sizeof(void*) == 8);

    return (arch_supported && word_size_supported);
}

// Performs a ptrace operation using the given parameters with this PTraceSession's pid and
// PTraceRunner. All ptrace operations work the same EXCEPT PTRACE_PEEKDATA which has been changed
// to provide the 64-bit read data at the address given in data, instead of by return value. Returns
// a non-success status on operation timeout or nonzero errno from ptrace.
rocattach_status_t
PTraceSession::ptrace_call(__ptrace_request   op,
                           ptrace_parameter_t addr,
                           ptrace_parameter_t data) const
{
    uint64_t           retval      = 0;
    int                local_errno = 0;
    rocattach_status_t status      = ptrace_run(m_pid, op, addr, data, &retval, &local_errno);
    if(status != ROCATTACH_STATUS_SUCCESS)
    {
        return status;
    }
    if(local_errno != 0)
    {
        return convert_ptrace_error(local_errno);
    }

    // As a special case, rearrange the return value of PEEKDATA to be written to the address given
    // in data. This standardizes the PEEKDATA operation to how other read operations (e.g. GETREGS)
    // work.
    if(op == PTRACE_PEEKDATA)
    {
        if(!std::holds_alternative<void*>(data))
        {
            return ROCATTACH_STATUS_ERROR;
        }
        auto* data_as_ptr = reinterpret_cast<uint64_t*>(std::get<void*>(data));
        if(!data_as_ptr)
        {
            return ROCATTACH_STATUS_ERROR;
        }
        *data_as_ptr = retval;
    }
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
PTraceSession::attach()
{
    if(m_state != PTRACE_SESSION_STATE_INITIAL)
    {
        return ROCATTACH_STATUS_ERROR;
    }
    // SEIZE attaches without stopping the process
    PTRACE_CALL(PTRACE_SEIZE, 0UL, 0UL);
    ROCP_INFO << "[rocprofiler-sdk-rocattach] Successfully attached to pid " << m_pid;
    ROCATTACH_CALL(start_signal_handler());
    m_state = PTRACE_SESSION_STATE_RUNNING;
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
PTraceSession::detach()
{
    if(m_state == PTRACE_SESSION_STATE_INITIAL || m_state == PTRACE_SESSION_STATE_DETACHED)
    {
        return ROCATTACH_STATUS_ERROR;
    }

    if(m_state == PTRACE_SESSION_STATE_RUNNING)
    {
        // Must be stopped to use PTRACE_DETACH
        stop();
    }

    ROCATTACH_CALL(stop_signal_handler());
    PTRACE_CALL(PTRACE_DETACH, 0UL, 0UL);
    m_state = PTRACE_SESSION_STATE_DETACHED;
    ROCP_INFO << "[rocprofiler-sdk-rocattach] Detached from pid " << m_pid;
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
PTraceSession::start_signal_handler()
{
    if(m_ptrace_signal_handler_state != PTRACE_SIGNAL_HANDLER_STATE_INITIAL)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] PTraceSession signal handler was in an "
                      "unexpected state when started";
        return ROCATTACH_STATUS_ERROR;
    }
    m_ptrace_signal_handler_thread = std::thread(ptrace_signal_handler_func,
                                                 m_pid,
                                                 std::ref(m_ptrace_signal_handler_state),
                                                 std::ref(m_ptrace_signal_handler_error));
    if(!wait_for_ne(m_ptrace_signal_handler_state,
                    PTRACE_SIGNAL_HANDLER_STATE_INITIAL,
                    PTRACE_HANDLER_START_STOP_TIMEOUT_MS))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] PTraceSession signal handler failed to start";
        return ROCATTACH_STATUS_ERROR;
    }
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
PTraceSession::stop_signal_handler()
{
    auto status = ROCATTACH_STATUS_SUCCESS;

    // The lock-step provided in attach should prevent a state where state is RUNNING and
    // the signal handler state is INITIAL.  Accessing the atomic twice here is OK because
    // of that lock-step.
    if(m_ptrace_signal_handler_state.load() == PTRACE_SIGNAL_HANDLER_STATE_INITIAL)
    {
        // The signal handler thread was never started.
        m_ptrace_signal_handler_state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
    }
    else if(m_ptrace_signal_handler_state.load() != PTRACE_SIGNAL_HANDLER_STATE_FINAL)
    {
        m_ptrace_signal_handler_state.store(PTRACE_SIGNAL_HANDLER_STATE_DETACHING);
        if(!wait_for_ne(m_ptrace_signal_handler_state,
                        PTRACE_SIGNAL_HANDLER_STATE_DETACHING,
                        PTRACE_HANDLER_START_STOP_TIMEOUT_MS))
        {
            status = m_ptrace_signal_handler_error.load();
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] PTraceSession signal handler failed to stop "
                          "when requested. Last status: "
                       << status;
            return status;
        }
        m_ptrace_signal_handler_thread.join();
        status = m_ptrace_signal_handler_error.load();
    }
    return status;
}

// While we are attached, we must monitor the target process for:
// - Process exits (WIFEXITED)
// - Process is killed (WIFSIGNALED)
// - Process is stopped (WIFSTOPPED)
// When the process exits or is killed, we simply report the status change and end the signal
// handling function. When the process is stopped, we use our current state to determine what to do
// - If ATTACHED, call PTRACE_CONT with the signal to resume the process
// - If WAITING_FOR_BREAKPOINT, leave the process stopped and transition our state to ATTACHED to
//   signal to the main thread which is awaiting a breakpoint
// See the comment on ptrace_session_signal_handler_state_t for more information.
void
PTraceSession::ptrace_signal_handler_func(
    int                                                 _pid,
    std::atomic<ptrace_session_signal_handler_state_t>& _state,
    std::atomic<rocattach_status_t>&                    _error)
{
    _state.store(PTRACE_SIGNAL_HANDLER_STATE_ATTACHED);
    while(_state.load() != PTRACE_SIGNAL_HANDLER_STATE_DETACHING)
    {
        int status{0};
        int retval{0};

        // make a non-blocking call to waitpid to check on our tracee process
        retval = waitpid(_pid, &status, WNOHANG);
        // if retval is 0, no error occurred and no state change was observed
        if(retval == 0)
        {
            std::this_thread::yield();
            continue;
        }
        else if(retval == -1)
        {
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] waitpid failed in "
                          "ptrace_signal_handler_func for pid "
                       << _pid;
            _error.store(ROCATTACH_STATUS_ERROR);
            _state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
            return;
        }

        if(status != 0 && WIFEXITED(status))
        {
            // Process exited normally, report status and end this thread.
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] process " << _pid
                       << " exited, status=" << WEXITSTATUS(status);
            _error.store(ROCATTACH_STATUS_SUCCESS);
            _state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
            return;
        }
        else if(status != 0 && WIFSIGNALED(status))
        {
            // Process was killed, report signal and end this thread.
            ROCP_ERROR << "[rocprofiler-sdk-rocattach] process " << _pid << " killed by signal "
                       << WTERMSIG(status);
            _error.store(ROCATTACH_STATUS_SUCCESS);
            _state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
            return;
        }
        else if(status != 0 && WIFSTOPPED(status))
        {
            // Process was stopped, handle the signal
            auto sig = WSTOPSIG(status);
            ROCP_TRACE << "[rocprofiler-sdk-rocattach] process " << _pid << " stopped by signal "
                       << sig;
            // If the signal is SIGTRAP (5) AND we were expecting a breakpoint, change state to
            // signal the update, otherwise continue to forward it to the process. NOTE: If our
            // injection causes a SIGSEGV, we can technically recover by handling the signal and
            // restoring the code, which would allow the user code to continue normally. For now,
            // this will crash the user app.
            ptrace_session_signal_handler_state_t expected_state =
                PTRACE_SIGNAL_HANDLER_STATE_WAITING_FOR_BREAKPOINT;
            if(sig == SIGTRAP &&
               _state.compare_exchange_strong(expected_state, PTRACE_SIGNAL_HANDLER_STATE_ATTACHED))
            {
                ROCP_TRACE << "[rocprofiler-sdk-rocattach] process " << _pid
                           << " hit expected breakpoint.";
            }
            else
            {
                // Not our signal, forward the signal to the app using CONT
                uint64_t           _retval = 0;
                int                _errno  = 0;
                rocattach_status_t _status = ptrace_run(
                    _pid, PTRACE_CONT, nullptr, static_cast<uint64_t>(sig), &_retval, &_errno);
                if(_status != ROCATTACH_STATUS_SUCCESS)
                {
                    _error.store(_status);
                    _state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
                    return;
                }
                if(_errno != 0)
                {
                    ROCP_ERROR << "[rocprofiler-sdk-rocattach] ptrace call failed. errno: "
                               << _errno << " - " << strerror(_errno) << ". params(PTRACE_CONT(7), "
                               << _pid << ", 0, " << sig << ")";
                    _error.store(convert_ptrace_error(_errno));
                    _state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
                    return;
                }
            }
        }

        std::this_thread::yield();
    }

    // While loop ended because we are detaching, close out gracefully.
    _error.store(ROCATTACH_STATUS_SUCCESS);
    _state.store(PTRACE_SIGNAL_HANDLER_STATE_FINAL);
}

rocattach_status_t
PTraceSession::write(size_t addr, const std::vector<uint8_t>& data, size_t size)
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        // If process is already stopped, use write_internal instead.
        return ROCATTACH_STATUS_ERROR;
    }

    ROCATTACH_CALL(stop());
    ROCATTACH_CALL(write_internal(addr, data, size));
    ROCATTACH_CALL(cont());
    return ROCATTACH_STATUS_SUCCESS;
}

// pre-cond: process must be stopped
rocattach_status_t
PTraceSession::write_internal(size_t addr, const std::vector<uint8_t>& data, size_t size) const
{
    // Write each word one at a time
    constexpr size_t word_size = sizeof(void*);
    size_t           word_iter = 0;
    for(word_iter = 0; word_iter < (size / word_size); ++word_iter)
    {
        const size_t offset = (word_iter * word_size);
        uint64_t     word;
        std::memcpy(&word, data.data() + offset, word_size);
        PTRACE_CALL(PTRACE_POKEDATA, addr + offset, word);
    }

    // If not evenly divisible, read the last word to do a masked partial write.
    size_t remainder = size % word_size;
    if(remainder != 0u)
    {
        const size_t offset    = (word_iter * word_size);
        uint64_t     last_word = 0;
        PTRACE_CALL(PTRACE_PEEKDATA, addr + offset, &last_word);
        std::memcpy(&last_word, data.data() + offset, remainder);
        PTRACE_CALL(PTRACE_POKEDATA, addr + offset, last_word);
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] ptrace wrote " << size << " bytes at " << addr;
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
PTraceSession::read(size_t addr, std::vector<uint8_t>& data, size_t size)
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        // If process is already stopped, use read_internal instead.
        return ROCATTACH_STATUS_ERROR;
    }

    ROCATTACH_CALL(stop());
    ROCATTACH_CALL(read_internal(addr, data, size));
    ROCATTACH_CALL(cont());
    return ROCATTACH_STATUS_SUCCESS;
}

// pre-cond: process must be stopped
rocattach_status_t
PTraceSession::read_internal(size_t addr, std::vector<uint8_t>& data, size_t size) const
{
    // Read each word one at a time
    data.clear();
    data.resize(size);
    constexpr size_t word_size = sizeof(void*);
    size_t           word_iter = 0;
    for(word_iter = 0; word_iter < (size / word_size); ++word_iter)
    {
        const size_t offset = (word_iter * word_size);
        uint64_t     word   = 0;
        PTRACE_CALL(PTRACE_PEEKDATA, addr + offset, &word);
        std::memcpy(data.data() + offset, &word, word_size);
    }

    // If not evenly divisible, read the last word and mask off the remainder
    size_t remainder = size % word_size;
    if(remainder != 0u)
    {
        const size_t offset    = (word_iter * word_size);
        uint64_t     last_word = 0;
        PTRACE_CALL(PTRACE_PEEKDATA, addr + offset, &last_word);
        std::memcpy(data.data() + offset, &last_word, remainder);
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] ptrace read " << size << " bytes at " << addr;
    return ROCATTACH_STATUS_SUCCESS;
}

rocattach_status_t
PTraceSession::swap(size_t                      addr,
                    const std::vector<uint8_t>& in_data,
                    std::vector<uint8_t>&       out_data,
                    size_t                      size)
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        // If process is already stopped, use swap_internal instead.
        return ROCATTACH_STATUS_ERROR;
    }

    ROCATTACH_CALL(stop());
    ROCATTACH_CALL(swap_internal(addr, in_data, out_data, size));
    ROCATTACH_CALL(cont());
    return ROCATTACH_STATUS_SUCCESS;
}

// pre-cond: process must be stopped
rocattach_status_t
PTraceSession::swap_internal(size_t                      addr,
                             const std::vector<uint8_t>& in_data,
                             std::vector<uint8_t>&       out_data,
                             size_t                      size) const
{
    ROCATTACH_CALL(read_internal(addr, out_data, size));
    ROCATTACH_CALL(write_internal(addr, in_data, size));
    return ROCATTACH_STATUS_SUCCESS;
}

// Helper function which updates states and communicates with the signal handler thread to await a
// single breakpoint. Updates the state to STOPPED when complete. Returns an error if the signal
// handler or ptrace fail unexpectedly.
rocattach_status_t
PTraceSession::wait_for_breakpoint()
{
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] waiting for breakpoint after trap instruction added";

    // Enforce transition from ATTACHED to WAITING_FOR_BREAKPOINT
    ptrace_session_signal_handler_state_t expected_state = PTRACE_SIGNAL_HANDLER_STATE_ATTACHED;
    if(!m_ptrace_signal_handler_state.compare_exchange_strong(
           expected_state, PTRACE_SIGNAL_HANDLER_STATE_WAITING_FOR_BREAKPOINT))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] signal handler thread was in an unexpected "
                      "state when waiting for stop. "
                      "State code: "
                   << expected_state;
        return ROCATTACH_STATUS_ERROR;
    }

    // Continue until breakpoint is hit
    ROCATTACH_CALL(cont());
    if(!wait_for_ne(m_ptrace_signal_handler_state,
                    PTRACE_SIGNAL_HANDLER_STATE_WAITING_FOR_BREAKPOINT,
                    PTRACE_BREAKPOINT_TIMEOUT_MS))
    {
        auto status = m_ptrace_signal_handler_error.load();
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] signal handler thread did not receive expected "
                      "breakpoint within timeout. Last status: "
                   << status;
        return status;
    }

    // If signal handler is not ATTACHED, error has occurred
    if(m_ptrace_signal_handler_state.load() != PTRACE_SIGNAL_HANDLER_STATE_ATTACHED)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] signal handler thread was in an unexpected "
                      "state after waiting for stop. State code: "
                   << m_ptrace_signal_handler_state.load();
        return m_ptrace_signal_handler_error.load();
    }

    // Manually set state to stopped
    // usually stop() handles this, but this stop was triggered manually in assembly code
    m_state = PTRACE_SESSION_STATE_STOPPED;
    return ROCATTACH_STATUS_SUCCESS;
}

// Helper function which updates states and communicates with the signal handler thread to await a
// single stop. Updates the state to STOPPED when complete. Returns an error if the signal handler
// or ptrace fail unexpectedly.
rocattach_status_t
PTraceSession::wait_for_stop()
{
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] waiting for stop after PTRACE_INTERRUPT";
    // Enforce transition from ATTACHED to WAITING_FOR_BREAKPOINT
    ptrace_session_signal_handler_state_t expected_state = PTRACE_SIGNAL_HANDLER_STATE_ATTACHED;
    if(!m_ptrace_signal_handler_state.compare_exchange_strong(
           expected_state, PTRACE_SIGNAL_HANDLER_STATE_WAITING_FOR_BREAKPOINT))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] signal handler thread was in an unexpected "
                      "state when waiting for "
                      "breakpoint. State code: "
                   << expected_state;
        return ROCATTACH_STATUS_ERROR;
    }

    // Call interrupt and wait until process is stopped
    PTRACE_CALL(PTRACE_INTERRUPT, 0UL, 0UL);
    if(!wait_for_ne(m_ptrace_signal_handler_state,
                    PTRACE_SIGNAL_HANDLER_STATE_WAITING_FOR_BREAKPOINT,
                    PTRACE_BREAKPOINT_TIMEOUT_MS))
    {
        auto status = m_ptrace_signal_handler_error.load();
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] signal handler thread did not receive expected "
                      "breakpoint within timeout. Last status: "
                   << status;
        return status;
    }

    // If signal handler is not ATTACHED, error has occurred
    if(m_ptrace_signal_handler_state.load() != PTRACE_SIGNAL_HANDLER_STATE_ATTACHED)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] signal handler thread was in an unexpected "
                      "state after waiting for breakpoint "
                   << m_ptrace_signal_handler_state.load();
        return m_ptrace_signal_handler_error.load();
    }

    // Set state to stopped now that process is stopped.
    m_state = PTRACE_SESSION_STATE_STOPPED;
    return ROCATTACH_STATUS_SUCCESS;
}

// Makes a syscall to mmap in the target process.
// Some sensible default parameters are used that are suitable for most applications:
// prot = PROT_READ | PROT_WRITE
// flags = MAP_PRIVATE | MAP_ANONYMOUS
rocattach_status_t
PTraceSession::simple_mmap(void*& addr, size_t length)
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] simple_mmap called in invalid state: "
                   << m_state;
        return ROCATTACH_STATUS_ERROR;
    }

    // Stop the process
    ROCATTACH_CALL(stop());

    // Get entry address for safe injection of op codes
    void* entry_addr = nullptr;
    ROCATTACH_CALL(get_auxv_entry(m_pid, entry_addr));

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, 0UL, &oldregs);
    // Set register file for system call to mmap:
    // mmap(addr=nullptr, length, prot, flags, -1, 0);
    struct user_regs_struct newregs = oldregs;

    newregs.rax = 9;                            // calling convention: 9 is syscall ID for mmap
    newregs.rdi = 0;                            // addr=nullptr
    newregs.rsi = length;                       // length
    newregs.rdx = PROT_READ | PROT_WRITE;       // prot
    newregs.r10 = MAP_PRIVATE | MAP_ANONYMOUS;  // flags
    newregs.r8  = -1;                           // fd (unused)
    newregs.r9  = 0;                            // offset
    newregs.rip =
        reinterpret_cast<size_t>(entry_addr);  // safe injection addr given by get_auxv_entry
    newregs.rsp = oldregs.rsp - 128;    // move sp by at least 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);  // base sp should be on 16-byte boundary
    // Set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, 0UL, &newregs);

    // x64 assembly to perform a syscall and breakpoint when done
    // 0f 05  syscall
    // cc     int3
    std::vector<uint8_t> new_code({0x0f, 0x05, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    ROCATTACH_CALL(swap_internal(reinterpret_cast<size_t>(entry_addr), new_code, old_code, 3));

    // Execute
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attempting to execute mmap syscall";
    ROCATTACH_CALL(wait_for_breakpoint());

    // Get registers to see mmap's return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, 0UL, &returnregs);

    // Write in old opcodes
    ROCATTACH_CALL(write_internal(reinterpret_cast<size_t>(entry_addr), old_code, 3));

    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, 0UL, &oldregs);

    // Restart execution
    ROCATTACH_CALL(cont());

    addr = reinterpret_cast<void*>(returnregs.rax);  // NOLINT(performance-no-int-to-ptr)
    return ROCATTACH_STATUS_SUCCESS;
}

// Makes a syscall to munmap in the target process.
// addr and length should match (or be a subset of) addr and length given to a previous mmap call.
rocattach_status_t
PTraceSession::simple_munmap(void*& addr, size_t length)
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] simple_munmap called in invalid state: "
                   << m_state;
        return ROCATTACH_STATUS_ERROR;
    }

    // Stop the process
    ROCATTACH_CALL(stop());

    // Get entry address for safe injection of op codes
    void* entry_addr = nullptr;
    ROCATTACH_CALL(get_auxv_entry(m_pid, entry_addr));

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, 0UL, &oldregs);
    // Set register file for system call to munmap:
    // munmap(addr, length);
    struct user_regs_struct newregs = oldregs;

    newregs.rax = 11;                              // calling convention: 11 is syscall ID for mumap
    newregs.rdi = reinterpret_cast<size_t>(addr);  // addr
    newregs.rsi = length;                          // length
    newregs.rip =
        reinterpret_cast<size_t>(entry_addr);  // safe injection addr given by get_auxv_entry
    newregs.rsp = oldregs.rsp - 128;    // move sp by at least 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);  // base sp should be on 16-byte boundary
    // Set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, 0UL, &newregs);

    // x64 assembly to perform a syscall and breakpoint when done
    // 0f 05  syscall
    // cc     int3
    std::vector<uint8_t> new_code({0x0f, 0x05, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    ROCATTACH_CALL(swap_internal(reinterpret_cast<size_t>(entry_addr), new_code, old_code, 3));

    // Execute
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attempting to execute munmap syscall";
    ROCATTACH_CALL(wait_for_breakpoint());

    // Get registers to see mmap's return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, 0UL, &returnregs);

    // Write in old opcodes
    ROCATTACH_CALL(write_internal(reinterpret_cast<size_t>(entry_addr), old_code, 3));

    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, 0UL, &oldregs);

    // Restart execution
    ROCATTACH_CALL(cont());

    return ROCATTACH_STATUS_SUCCESS;
}

// Makes a call to library::symbol() in the target process
rocattach_status_t
PTraceSession::call_function(const std::string& library,
                             const std::string& symbol,
                             uint64_t&          ret_value)
{
    return call_function(library, symbol, ret_value, nullptr, nullptr);
}

// Makes a call to library::symbol(first_param) in the target process
rocattach_status_t
PTraceSession::call_function(const std::string& library,
                             const std::string& symbol,
                             uint64_t&          ret_value,
                             void*              first_param)
{
    return call_function(library, symbol, ret_value, first_param, nullptr);
}

// Makes a call to library::symbol(first_param, second_param) in the target process.
// This supports calling a dynamically loaded function with at most 2 parameters.
// Uses x64 calling convention: RAX for return value, RDI for first param, RSI for second param
rocattach_status_t
PTraceSession::call_function(const std::string& library,
                             const std::string& symbol,
                             uint64_t&          ret_value,
                             void*              first_param,
                             void*              second_param)
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] call_function called in invalid state: "
                   << m_state;
        return ROCATTACH_STATUS_ERROR;
    }

    // Stop the process
    ROCATTACH_CALL(stop());

    // Find address in target program to call
    void* target_addr = nullptr;
    if(!find_symbol(m_pid, target_addr, library, symbol))
    {
        ROCP_ERROR
            << "[rocprofiler-sdk-rocattach] call_function failed to find target symbol address for "
            << library << "::" << symbol;
        return ROCATTACH_STATUS_ERROR;
    }

    // Get entry address for safe injection of op codes
    void* entry_addr = nullptr;
    ROCATTACH_CALL(get_auxv_entry(m_pid, entry_addr));

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, 0UL, &oldregs);

    // Construct registers to call a function with 2 parameters
    // symbol(first_param, second_param)
    struct user_regs_struct newregs = oldregs;

    newregs.rax = reinterpret_cast<size_t>(target_addr);   // target function
    newregs.rdi = reinterpret_cast<size_t>(first_param);   // first parameter
    newregs.rsi = reinterpret_cast<size_t>(second_param);  // second parameter
    newregs.rip =
        reinterpret_cast<size_t>(entry_addr);  // safe injection addr given by get_auxv_entry
    newregs.rsp = oldregs.rsp - 128;    // move sp by at least 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);  // base sp should be on 16-byte boundary
    // Set function  registers
    PTRACE_CALL(PTRACE_SETREGS, 0UL, &newregs);

    // x64 assembly to call a function by register and breakpoint when done
    // ff d0  call rax
    // cc     int3
    std::vector<uint8_t> new_code({0xff, 0xd0, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    ROCATTACH_CALL(swap_internal(reinterpret_cast<size_t>(entry_addr), new_code, old_code, 3));

    // Execute
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Attempting to execute " << library << "::" << symbol
               << "(" << first_param << ", " << second_param << ")";
    ROCATTACH_CALL(wait_for_breakpoint());

    // Get registers to see  return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, 0UL, &returnregs);

    // Write in old opcodes
    ROCATTACH_CALL(write_internal(reinterpret_cast<size_t>(entry_addr), old_code, 3));

    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, 0UL, &oldregs);

    // Restart execution
    ROCATTACH_CALL(cont());

    ret_value = returnregs.rax;
    return ROCATTACH_STATUS_SUCCESS;
}

// Calls PTRACE_STOP and waits for the stop to complete.
// Target process will be stopped after this call.
rocattach_status_t
PTraceSession::stop()
{
    if(m_state != PTRACE_SESSION_STATE_RUNNING)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] stop called in invalid state: " << m_state;
        return ROCATTACH_STATUS_ERROR;
    }

    // Stop the process and update state
    ROCATTACH_CALL(wait_for_stop());

    ROCP_TRACE << "[rocprofiler-sdk-rocattach] ptrace stopped pid " << m_pid;
    return ROCATTACH_STATUS_SUCCESS;
}

// Calls PTRACE_CONT.
// Target process will be running after this call.
rocattach_status_t
PTraceSession::cont()
{
    if(m_state != PTRACE_SESSION_STATE_STOPPED)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] cont called in invalid state: " << m_state;
        return ROCATTACH_STATUS_ERROR;
    }

    PTRACE_CALL(PTRACE_CONT, 0UL, 0UL);
    m_state = PTRACE_SESSION_STATE_RUNNING;
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] ptrace resumed pid " << m_pid;
    return ROCATTACH_STATUS_SUCCESS;
}

}  // namespace rocattach
}  // namespace rocprofiler
