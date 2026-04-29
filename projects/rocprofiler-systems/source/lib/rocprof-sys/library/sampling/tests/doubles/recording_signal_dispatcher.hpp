// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Test double — must NOT include <signal.h> directly (NFR-PORT-3).
// sigset_t is not stored; the recorded-call struct uses opaque void* capture.
// The how integer from pthread_sigmask is mapped to sigmask_how to remove the
// need for SIG_BLOCK / SIG_UNBLOCK in test TUs (C-1c fix).

#include <cerrno>
#include <vector>

namespace rocprofsys::sampling::test
{

// Portable enum replacing SIG_BLOCK / SIG_UNBLOCK / SIG_SETMASK constants.
enum class sigmask_how : int
{
    block   = 0,
    unblock = 1,
    setmask = 2,
    unknown = -1
};

struct sigmask_call
{
    sigmask_how m_how     = sigmask_how::unknown;
    bool        m_has_set = false;
    bool        m_has_old = false;
};

struct recording_signal_dispatcher
{
    // Matches the production signal_dispatcher concept:
    //   apply_sigmask(int, T const*, T*).
    // Accepts sigset_t* via implicit void* conversion — no <signal.h> needed here.
    // Maps raw POSIX how-constant to sigmask_how enum on record.
    int apply_sigmask(int how, void const* set, void* oldset) noexcept
    {
        sigmask_how mapped = sigmask_how::unknown;
        // SIG_BLOCK, SIG_UNBLOCK, SIG_SETMASK are implementation-defined (not
        // POSIX-mandated). On Linux/glibc and musl their values are 0, 1, 2 respectively.
        if(how == 0)
            mapped = sigmask_how::block;
        else if(how == 1)
            mapped = sigmask_how::unblock;
        else if(how == 2)
            mapped = sigmask_how::setmask;
        m_calls.push_back({ mapped, set != nullptr, oldset != nullptr });
        if(m_fail_next)
        {
            m_fail_next = false;
            return EINVAL;
        }
        return 0;
    }

    void set_fail_next() noexcept { m_fail_next = true; }

    std::vector<sigmask_call> m_calls;
    bool                      m_fail_next = false;
};

}  // namespace rocprofsys::sampling::test
