// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::control
{
void
session::shutdown()
{
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
    const std::scoped_lock subs_lk{ m_subscribers_mutex };
    m_subscribers.push_back(std::move(sub));
}

void
session::register_trigger(std::string_view name, action initial)
{
    const std::scoped_lock trig_lk{ m_actions_mutex };
    m_actions[std::string{ name }] = initial;
    m_active.store(resolve_locked(), std::memory_order_relaxed);
}

void
session::unregister_trigger(std::string_view name)
{
    const std::scoped_lock trig_lk{ m_actions_mutex };
    m_actions.erase(std::string{ name });
    m_active.store(resolve_locked(), std::memory_order_relaxed);
}

void
session::set_action(std::string_view name, action act)
{
    const std::scoped_lock notify_lk{ m_notify_mutex };

    bool was_active = false;
    bool now_active = false;
    {
        const std::scoped_lock action_lk{ m_actions_mutex };

        was_active                     = m_active.load(std::memory_order_relaxed);
        m_actions[std::string{ name }] = act;
        now_active                     = resolve_locked();
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

void
session::force_initial_pause()
{
    if(is_active())
    {
        return;
    }
    notify_pause();
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
