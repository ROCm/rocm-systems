// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace rocprofsys::control::testing
{
/// Test-only clock with virtual time. Tests advance the clock explicitly
/// via advance(); sleep_until blocks until the virtual now() catches up to
/// the deadline or interrupt() is called.
class mock_clock
{
public:
    explicit mock_clock(clock_time_point start = {}) noexcept
    : m_now_ns{ start.time_since_epoch().count() }
    {}

    ~mock_clock() = default;

    mock_clock(const mock_clock&)            = delete;
    mock_clock& operator=(const mock_clock&) = delete;
    mock_clock(mock_clock&&)                 = delete;
    mock_clock& operator=(mock_clock&&)      = delete;

    [[nodiscard]] clock_time_point now() const noexcept
    {
        return clock_time_point{ clock_duration{
            m_now_ns.load(std::memory_order_acquire) } };
    }

    [[nodiscard]] bool sleep_until(clock_time_point deadline)
    {
        std::unique_lock<std::mutex> lk{ m_mutex };
        m_cv.wait(lk, [this, deadline] { return now() >= deadline || m_interrupted; });
        return !m_interrupted;
    }

    void interrupt()
    {
        {
            std::scoped_lock const lk{ m_mutex };
            m_interrupted = true;
        }
        m_cv.notify_all();
    }

    /// Advance virtual time by @p delta and wake any sleepers whose
    /// deadline may now have been reached.
    void advance(clock_duration delta)
    {
        m_now_ns.fetch_add(delta.count(), std::memory_order_release);
        m_cv.notify_all();
    }

private:
    std::atomic<std::int64_t> m_now_ns;
    std::mutex                m_mutex;
    std::condition_variable   m_cv;
    bool                      m_interrupted{ false };
};
}  // namespace rocprofsys::control::testing
