// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"
#include "core/control/session.hpp"
#include "core/control/trigger.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string_view>
#include <thread>

namespace rocprofsys::control::triggers
{
template <typename Clock>
class time_window : public trigger
{
public:
    struct config
    {
        clock_duration delay{};
        clock_duration duration{};
    };

    time_window(session& sess, Clock& clk, config cfg) noexcept
    : m_session{ sess }
    , m_clock{ clk }
    , m_config{ cfg }
    {}

    ~time_window() override
    {
        stop();
        m_session.unregister_trigger(*this);
    }

    time_window(const time_window&)            = delete;
    time_window& operator=(const time_window&) = delete;
    time_window(time_window&&)                 = delete;
    time_window& operator=(time_window&&)      = delete;

    [[nodiscard]] std::string_view name() const noexcept override
    {
        return "time_window";
    }

    [[nodiscard]] vote initial_vote() const noexcept override
    {
        if(m_config.delay > clock_duration::zero()) return vote::paused;
        if(m_config.duration > clock_duration::zero()) return vote::active;
        return vote::abstain;
    }

    /// Spawn the worker thread that advances the window through delay and
    /// duration phases. Idempotent: a second call is a no-op. Not safe to
    /// call concurrently with stop() from a different thread than the one
    /// serializing start()/stop() calls (guarded via m_lifecycle_mutex).
    void start()
    {
        std::scoped_lock const lk{ m_lifecycle_mutex };
        if(!has_window()) return;
        if(m_thread.joinable()) return;
        m_thread = std::thread{ [this]() { worker(); } };
    }

    /// Interrupt the clock and join the worker thread. Idempotent.
    /// m_thread.join() can only throw if joinable() is false (guarded above)
    /// or if called from the worker thread itself, which never happens -
    /// stop() is only ever invoked from the owning thread (including via
    /// the destructor), never from worker().
    void stop() noexcept
    {
        std::scoped_lock const lk{ m_lifecycle_mutex };
        if(!m_thread.joinable()) return;
        m_clock.interrupt();
        m_thread.join();
    }

private:
    session&     m_session;
    Clock&       m_clock;
    const config m_config;
    std::thread  m_thread;
    std::mutex   m_lifecycle_mutex;

    [[nodiscard]] bool has_window() const noexcept
    {
        return m_config.delay > clock_duration::zero() ||
               m_config.duration > clock_duration::zero();
    }

    void worker()
    {
        const auto t0           = m_clock.now();
        const bool has_delay    = m_config.delay > clock_duration::zero();
        const bool has_duration = m_config.duration > clock_duration::zero();

        if(has_delay)
        {
            if(!m_clock.sleep_until(t0 + m_config.delay)) return;  // interrupted
            m_session.publish_vote(*this, vote::active);
        }

        if(has_duration)
        {
            const auto end = t0 + m_config.delay + m_config.duration;
            if(!m_clock.sleep_until(end)) return;         // interrupted
            m_session.publish_vote(*this, vote::paused);  // terminal
        }
    }
};
}  // namespace rocprofsys::control::triggers
