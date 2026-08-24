// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"
#include "core/control/session.hpp"
#include "core/state.hpp"

#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

namespace rocprofsys::control::triggers
{
template <typename Clock>
class time_window
{
public:
    struct config
    {
        clock_duration delay{};
        clock_duration duration{};
    };

    time_window(std::shared_ptr<session> sess, Clock& clk, config cfg)
    : m_session{ std::move(sess) }
    , m_clock{ clk }
    , m_config{ cfg }
    {
        m_session->register_trigger(trigger_name, initial_action(cfg));
    }

    ~time_window()
    {
        stop();
        m_session->unregister_trigger(trigger_name);
    }

    time_window(const time_window&)            = delete;
    time_window& operator=(const time_window&) = delete;
    time_window(time_window&&)                 = delete;
    time_window& operator=(time_window&&)      = delete;

    /// Spawn the worker thread that advances the window through delay and
    /// duration phases. Idempotent: a second call is a no-op. Not safe to
    /// call concurrently with stop() from a different thread than the one
    /// serializing start()/stop() calls (guarded via m_lifecycle_mutex).
    void start()
    {
        const std::scoped_lock window_lk{ m_lifecycle_mutex };
        if(!has_window())
        {
            return;
        }
        if(m_thread.joinable())
        {
            return;
        }
        m_thread = std::thread{ [this]() { worker(); } };
    }

    /// Interrupt the clock and join the worker thread. Idempotent.
    /// m_thread.join() can only throw if joinable() is false (guarded above)
    /// or if called from the worker thread itself, which never happens -
    /// stop() is only ever invoked from the owning thread (including via
    /// the destructor), never from worker().
    void stop() noexcept
    {
        const std::scoped_lock window_lk{ m_lifecycle_mutex };
        if(!m_thread.joinable())
        {
            return;
        }
        m_clock.interrupt();
        m_thread.join();
    }

private:
    static constexpr std::string_view trigger_name = "time_window";

    std::shared_ptr<session> m_session;
    Clock&                   m_clock;
    const config             m_config;
    std::thread              m_thread;
    std::mutex               m_lifecycle_mutex;

    [[nodiscard]] static action initial_action(const config& cfg) noexcept
    {
        if(cfg.delay > clock_duration::zero())
        {
            return action::pause;
        }
        if(cfg.duration > clock_duration::zero())
        {
            return action::trace;
        }
        return action::skip;
    }

    [[nodiscard]] bool has_window() const noexcept
    {
        return m_config.delay > clock_duration::zero() ||
               m_config.duration > clock_duration::zero();
    }

    void worker()
    {
        const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);

        const auto current_ts   = m_clock.now();
        const bool has_delay    = m_config.delay > clock_duration::zero();
        const bool has_duration = m_config.duration > clock_duration::zero();

        if(has_delay)
        {
            if(!m_clock.sleep_until(current_ts + m_config.delay))
            {
                return;  // interrupted
            }
            m_session->set_action(trigger_name, action::trace);
        }

        if(has_duration)
        {
            const auto end = current_ts + m_config.delay + m_config.duration;
            if(!m_clock.sleep_until(end))
            {
                return;  // interrupted
            }
            m_session->set_action(trigger_name, action::pause);  // terminal
        }
    }
};
}  // namespace rocprofsys::control::triggers
