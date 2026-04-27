// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "steady.hpp"

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
    std::unique_lock<std::mutex> lk{ m_mutex };
    // wait_until's predicate-form returns the predicate value at wakeup:
    //   true  -> interrupted (predicate satisfied before timeout)
    //   false -> deadline reached
    return !m_cv.wait_until(lk, deadline, [this] { return m_interrupted; });
}

void
steady::interrupt()
{
    {
        std::scoped_lock const lk{ m_mutex };
        m_interrupted = true;
    }
    m_cv.notify_all();
}
}  // namespace rocprofsys::control::clocks
