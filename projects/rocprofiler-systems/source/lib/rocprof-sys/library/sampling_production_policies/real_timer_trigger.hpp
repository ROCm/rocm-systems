// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Linux POSIX-timer implementation of the timer_trigger policy concept.
// Uses timer_create(SIGEV_THREAD_ID) + timer_settime to deliver SIGRTMIN+N
// to a specific thread.
//
// Production-only: lives in library/sampling_production_policies/ and is
// never included by standalone test binaries. No ROCPROFSYS_INTERNAL_BUILD
// guards needed here.

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <sys/types.h>
#include <time.h>

// Forward declaration of the production signal handler.
// Defined in services_accessor.cpp; declared here because start() installs it.
extern "C" void
rocprofsys_sampling_signal_handler(int, siginfo_t*, void*);

// POSIX defines sigmask(sig) as a macro; undefine to avoid conflicts.
#ifdef sigmask
#    undef sigmask
#endif

namespace rocprofsys::sampling
{

class real_timer_trigger
{
public:
    real_timer_trigger() noexcept = default;

    ~real_timer_trigger() noexcept
    {
        if(m_armed) stop();
    }

    real_timer_trigger(real_timer_trigger const&)            = delete;
    real_timer_trigger& operator=(real_timer_trigger const&) = delete;

    // Configure timer: deliver signum to sys_tid at freq_hz after delay_sec.
    // clk is CLOCK_REALTIME or CLOCK_THREAD_CPUTIME_ID.
    void configure(int64_t /*tid*/, pid_t sys_tid, int signum, clockid_t clk,
                   double freq_hz, double delay_sec) noexcept
    {
        m_signum    = signum;
        m_sys_tid   = sys_tid;
        m_clk       = clk;
        m_freq_hz   = freq_hz;
        m_delay_sec = delay_sec;
    }

    void start() noexcept
    {
        if(m_armed || m_signum == 0) return;

        // Install the sampling signal handler via sigaction (SA_RESTART | SA_SIGINFO).
        struct sigaction sa = {};
        sigemptyset(&sa.sa_mask);
        sa.sa_flags     = SA_RESTART | SA_SIGINFO;
        sa.sa_sigaction = rocprofsys_sampling_signal_handler;
        if(sigaction(m_signum, &sa, nullptr) != 0) return;

        sigevent sev              = {};
        sev.sigev_notify          = SIGEV_THREAD_ID;
        sev.sigev_signo           = m_signum;
        sev._sigev_un._tid        = m_sys_tid;
        sev.sigev_value.sival_ptr = nullptr;

        if(timer_create(m_clk, &sev, &m_timerid) != 0)
        {
            // Log errno so we can diagnose silent failures during setup.
            (void) errno;  // errno readable in debugger / strace
            // Attempt CLOCK_REALTIME fallback when CPUTIME is unavailable.
            if(m_clk == CLOCK_THREAD_CPUTIME_ID)
            {
                m_clk = CLOCK_REALTIME;
                if(timer_create(m_clk, &sev, &m_timerid) != 0) return;
            }
            else
            {
                return;
            }
        }

        // Period in nanoseconds from frequency.
        long period_ns = (m_freq_hz > 0.0) ? static_cast<long>(1e9 / m_freq_hz)
                                           : 1000000L;  // 1 ms default

        // Initial fire delay — honor ROCPROFSYS_SAMPLING_*_DELAY semantics.
        // delay 0 → first fire after one period (legacy default).
        // delay >0 → first fire after delay_sec; matches legacy timemory behaviour
        // where short workloads (e.g. pause_resume example) get no samples because
        // the workload finishes before the delay elapses.
        long delay_ns =
            (m_delay_sec > 0.0) ? static_cast<long>(m_delay_sec * 1e9) : period_ns;

        itimerspec ts          = {};
        ts.it_value.tv_sec     = static_cast<time_t>(delay_ns / 1000000000L);
        ts.it_value.tv_nsec    = static_cast<long>(delay_ns % 1000000000L);
        ts.it_interval.tv_sec  = 0;
        ts.it_interval.tv_nsec = period_ns;

        if(timer_settime(m_timerid, 0, &ts, nullptr) != 0)
        {
            timer_delete(m_timerid);
            return;
        }

        m_armed = true;
    }

    void stop() noexcept
    {
        if(!m_armed) return;
        itimerspec ts = {};
        timer_settime(m_timerid, 0, &ts, nullptr);
        timer_delete(m_timerid);
        m_armed = false;
    }

    [[nodiscard]] bool is_armed() const noexcept { return m_armed; }

private:
    timer_t   m_timerid{};
    pid_t     m_sys_tid   = 0;
    int       m_signum    = 0;
    clockid_t m_clk       = CLOCK_REALTIME;
    double    m_freq_hz   = 0.0;
    double    m_delay_sec = 0.0;
    bool      m_armed     = false;
};

}  // namespace rocprofsys::sampling
