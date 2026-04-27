// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"

#include <condition_variable>
#include <mutex>

namespace rocprofsys::control::clocks
{
/// Production clock backed by std::chrono::steady_clock. sleep_until uses
/// std::condition_variable::wait_until so interrupt() can wake any waiter
/// (used during shutdown to cancel a sleeping time_window trigger).
///
/// Satisfies the clock concept (see core/control/clock.hpp).
class steady
{
public:
    steady()  = default;
    ~steady() = default;

    steady(const steady&)            = delete;
    steady& operator=(const steady&) = delete;
    steady(steady&&)                 = delete;
    steady& operator=(steady&&)      = delete;

    [[nodiscard]] clock_time_point now() const noexcept;
    [[nodiscard]] bool             sleep_until(clock_time_point deadline);
    void                           interrupt();

private:
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    bool                    m_interrupted{ false };
};
}  // namespace rocprofsys::control::clocks
