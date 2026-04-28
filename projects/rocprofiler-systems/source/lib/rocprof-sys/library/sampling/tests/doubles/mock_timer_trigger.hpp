// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <sys/types.h>
#include <time.h>
#include <vector>

namespace rocprofsys::sampling::test
{

struct timer_trigger_call
{
    int64_t   m_tid       = 0;
    pid_t     m_sys_tid   = 0;
    int       m_signum    = 0;
    clockid_t m_clock     = 0;
    double    m_freq_hz   = 0;
    double    m_delay_sec = 0;
};

struct mock_timer_trigger
{
    void configure(int64_t tid, pid_t sys_tid, int signum, clockid_t clk, double freq_hz,
                   double delay_sec) noexcept
    {
        m_calls.push_back({ tid, sys_tid, signum, clk, freq_hz, delay_sec });
    }

    void               start() noexcept { m_armed = true; }
    void               stop() noexcept { m_armed = false; }
    [[nodiscard]] bool is_armed() const noexcept { return m_armed; }

    std::vector<timer_trigger_call> m_calls;
    bool                            m_armed = false;
};

}  // namespace rocprofsys::sampling::test
