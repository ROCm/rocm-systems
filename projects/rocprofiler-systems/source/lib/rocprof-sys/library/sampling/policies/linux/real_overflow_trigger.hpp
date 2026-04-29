// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Linux perf_event_open implementation of the overflow_trigger policy concept.
// Wraps rocprofsys::perf::perf_event (library/perf.hpp) to deliver a signal
// when a hardware counter overflows.
//
// Production-only: lives in sampling/policies/ and is
// never included by standalone test binaries. No ROCPROFSYS_INTERNAL_BUILD
// guards needed here.

#include "library/perf.hpp"
#include "sampling/policies/linux/sampling_signal_handler_fwd.hpp"

#include <csignal>
#include <cstdint>
#include <sys/types.h>

namespace rocprofsys::sampling
{

class real_overflow_trigger
{
public:
    real_overflow_trigger() = default;

    ~real_overflow_trigger() noexcept { stop(); }

    real_overflow_trigger(real_overflow_trigger const&)            = delete;
    real_overflow_trigger& operator=(real_overflow_trigger const&) = delete;

    // Configure: take the already-populated perf_event_attr, open the fd,
    // and arm for signal delivery to this thread.
    // attr must point to a perf_event_attr (cast from caller's void const*).
    // FatalErrorPolicy::fatal() is called on perf_event_open failure (NFR-FM-2, EC-3).
    template <class FatalErrorPolicy>
    void configure(int64_t /*tid*/, pid_t sys_tid, int signum, void const* attr,
                   FatalErrorPolicy& fatal)
    {
        if(!attr) return;

        m_signum = signum;
        auto pe  = *static_cast<perf_event_attr const*>(attr);

        auto err = m_event.open(pe, sys_tid, -1);
        if(err)
        {
            fatal.fatal(__FILE__, __LINE__,
                        "real_overflow_trigger: perf_event_open failed: {}", *err);
        }

        m_event.set_ready_signal(signum);
        m_open = true;
    }

    void start() noexcept
    {
        if(!m_open) return;

        // Install the sampling signal handler for the overflow signal so perf fd
        // delivery (F_SETSIG path) dispatches to rocprofsys_sampling_signal_handler.
        struct sigaction sa = {};
        sigemptyset(&sa.sa_mask);
        sa.sa_flags     = SA_RESTART | SA_SIGINFO;
        sa.sa_sigaction = rocprofsys_sampling_signal_handler;
        sigaction(m_signum, &sa, nullptr);

        m_event.start();
    }

    void stop() noexcept
    {
        if(m_open)
        {
            m_event.stop();
            m_event.close();
            m_open = false;
        }
    }

    [[nodiscard]] bool is_open() const noexcept { return m_open && m_event.is_open(); }

private:
    perf::perf_event m_event;
    int              m_signum = 0;
    bool             m_open   = false;
};

}  // namespace rocprofsys::sampling
