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
/// A time_window's delay/duration schedule. Clock-agnostic - the same specs
/// can be handed to a time_window<Clock> for any Clock.
struct time_window_specs
{
    clock_duration delay{};
    clock_duration duration{};
};

template <typename Clock>
class time_window
{
public:
    time_window(std::shared_ptr<session> sess, Clock& clk, time_window_specs specs,
                scope event_scope = scope::global)
    : m_session{ std::move(sess) }
    , m_clock{ clk }
    , m_specs{ specs }
    , m_scope{ event_scope }
    {
        m_session->register_trigger(trigger_name, initial_action(m_specs), m_scope);
    }

    ~time_window()
    {
        stop();
        m_session->unregister_trigger(trigger_name, m_scope);
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
        const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
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

    void stop() noexcept
    {
        const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
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
    const time_window_specs  m_specs;
    const scope              m_scope;
    std::thread              m_thread;
    std::mutex               m_lifecycle_mutex;

    [[nodiscard]] static action initial_action(const time_window_specs& specs) noexcept
    {
        if(specs.delay > clock_duration::zero())
        {
            return action::pause;
        }
        if(specs.duration > clock_duration::zero())
        {
            return action::trace;
        }
        return action::skip;
    }

    [[nodiscard]] bool has_window() const noexcept
    {
        return m_specs.delay > clock_duration::zero() ||
               m_specs.duration > clock_duration::zero();
    }

    void worker()
    {
        const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);

        const auto current_ts   = m_clock.now();
        const bool has_delay    = m_specs.delay > clock_duration::zero();
        const bool has_duration = m_specs.duration > clock_duration::zero();

        if(has_delay)
        {
            if(!m_clock.sleep_until(current_ts + m_specs.delay))
            {
                return;  // interrupted
            }
            m_session->set_action(trigger_name, action::trace, m_scope);
        }

        if(has_duration)
        {
            const auto end = current_ts + m_specs.delay + m_specs.duration;
            if(!m_clock.sleep_until(end))
            {
                return;  // interrupted
            }
            m_session->set_action(trigger_name, action::pause, m_scope);  // terminal
        }
    }
};
}  // namespace rocprofsys::control::triggers
