// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "core/state.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::control
{
void
session::shutdown()
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    {
        const std::scoped_lock subs_lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }
    {
        const std::scoped_lock action_lock{ m_actions_mutex };
        m_actions.clear();
        m_active.store(true, std::memory_order_relaxed);
    }
}

void
session::subscribe(subscriber sub)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock subs_lk{ m_subscribers_mutex };
    m_subscribers.push_back(std::move(sub));
}

void
session::register_trigger(std::string_view name, action initial)
{
    apply_locked_transition([&] { m_actions[std::string{ name }] = initial; }, name);
}

void
session::unregister_trigger(std::string_view name)
{
    apply_locked_transition([&] { m_actions.erase(std::string{ name }); }, name);
}

void
session::set_action(std::string_view name, action act)
{
    apply_locked_transition([&] { m_actions[std::string{ name }] = act; }, name);
}

void
session::apply_locked_transition(const std::function<void()>& mutate,
                                 std::string_view             name)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_notify_mutex };

    bool was_active = false;
    bool now_active = false;
    {
        const std::scoped_lock action_lk{ m_actions_mutex };

        was_active = m_active.load(std::memory_order_relaxed);
        mutate();
        now_active = resolve_locked();
        m_active.store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active)
    {
        return;
    }

    LOG_DEBUG("session: trigger '{}' {} the session", name,
              now_active ? "resumed" : "paused");

    if(now_active)
    {
        notify_resume();
    }
    else
    {
        notify_pause();
    }
}

// Any pause action pauses the session. Skip is ignored.
// With no actions the session is active by default.
bool
session::resolve_locked() const noexcept
{
    return std::ranges::none_of(
        m_actions, [](const auto& entry) { return entry.second == action::pause; });
}

void
session::notify_pause()
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        LOG_DEBUG("session: pausing subscriber '{}'", sub.name);
        if(sub.on_pause)
        {
            sub.on_pause();
        }
    }
}

void
session::notify_resume()
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        LOG_DEBUG("session: resuming subscriber '{}'", sub.name);
        if(sub.on_resume)
        {
            sub.on_resume();
        }
    }
}
}  // namespace rocprofsys::control
