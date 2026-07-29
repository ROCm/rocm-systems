// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "steady.hpp"

#include "core/control/clock.hpp"
#include "core/state.hpp"

#include <chrono>
#include <mutex>

namespace rocprofsys::control::clocks
{
clock_time_point
steady::now() const noexcept
{
    return std::chrono::time_point_cast<clock_duration>(std::chrono::steady_clock::now());
}

bool
steady::sleep_until(clock_time_point deadline)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    std::unique_lock<std::mutex> clk_lcoks{ m_mutex };
    // wait_until's predicate-form returns the predicate value at wakeup:
    //   true  -> interrupted (predicate satisfied before timeout)
    //   false -> deadline reached
    return !m_cv.wait_until(clk_lcoks, deadline, [this] { return m_interrupted; });
}

void
steady::interrupt() noexcept
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    {
        const std::scoped_lock clk_lcoks{ m_mutex };
        m_interrupted = true;
    }
    m_cv.notify_all();
}

void
steady::reset() noexcept
{
    const std::scoped_lock lk{ m_mutex };
    m_interrupted = false;
}
}  // namespace rocprofsys::control::clocks
