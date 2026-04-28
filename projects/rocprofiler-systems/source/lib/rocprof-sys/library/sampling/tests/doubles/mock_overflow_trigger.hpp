// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Test double — must NOT include <linux/perf_event.h> (NFR-PORT-3).
// The perf_event_attr is not captured; the mock verifies that configure() was
// called with the right tid/signum, not that the kernel struct contents are correct.

#include <cstdint>
#include <sys/types.h>
#include <vector>

namespace rocprofsys::sampling::test
{

struct overflow_trigger_call
{
    int64_t m_tid     = 0;
    pid_t   m_sys_tid = 0;
    int     m_signum  = 0;
};

struct mock_overflow_trigger
{
    // accept attr as opaque void const* — callers pass perf_event_attr* transparently
    void configure(int64_t tid, pid_t sys_tid, int signum, void const* /*attr*/) noexcept
    {
        m_calls.push_back({ tid, sys_tid, signum });
        m_open = true;
    }

    void               start() noexcept { m_armed = true; }
    void               stop() noexcept { m_armed = false; }
    [[nodiscard]] bool is_open() const noexcept { return m_open; }

    std::vector<overflow_trigger_call> m_calls;
    bool                               m_open  = false;
    bool                               m_armed = false;
};

}  // namespace rocprofsys::sampling::test
