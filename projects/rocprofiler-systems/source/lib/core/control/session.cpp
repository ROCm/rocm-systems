// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include <algorithm>
#include <mutex>
#include <utility>

namespace rocprofsys::control
{
void
session::shutdown()
{
    {
        std::scoped_lock const lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }
    {
        std::scoped_lock const lk{ m_votes_mutex };
        m_votes.clear();
        m_active.store(true, std::memory_order_relaxed);
    }
}

void
session::subscribe(subscriber sub)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    m_subscribers.push_back(std::move(sub));
}

void
session::attach(trigger& trig)
{
    std::scoped_lock const lk{ m_votes_mutex };
    m_votes.push_back({ trig.name(), trig.initial_vote() });
    m_active.store(resolve_locked(), std::memory_order_relaxed);
}

void
session::force_initial_pause()
{
    if(is_active()) return;
    notify_pause();
}

void
session::publish(const trigger& trig, vote new_vote)
{
    bool was_active = false;
    bool now_active = false;
    {
        std::scoped_lock const lk{ m_votes_mutex };
        was_active = m_active.load(std::memory_order_relaxed);

        const auto name = trig.name();
        auto       it   = std::find_if(m_votes.begin(), m_votes.end(),
                                       [name](const vote_entry& e) { return e.name == name; });
        if(it == m_votes.end())
            m_votes.push_back({ name, new_vote });
        else
            it->current_vote = new_vote;

        now_active = resolve_locked();
        m_active.store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active) return;
    if(now_active)
        notify_resume();
    else
        notify_pause();
}

// Any paused vote pauses the session. Abstain is ignored.
// With no votes the session is active by default.
bool
session::resolve_locked() const noexcept
{
    for(const auto& entry : m_votes)
    {
        if(entry.current_vote == vote::paused) return false;
    }
    return true;
}

bool
session::is_active_excluding(std::string_view name) const noexcept
{
    // Cold path: typical caller (roctx marker gate) hits this only on the
    // recording side after a fast atomic check, and there are 1-2 votes.
    std::scoped_lock const lk{ m_votes_mutex };
    for(const auto& entry : m_votes)
    {
        if(entry.name == name) continue;
        if(entry.current_vote == vote::paused) return false;
    }
    return true;
}

void
session::notify_pause()
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(sub.on_pause) sub.on_pause();
    }
}

void
session::notify_resume()
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(sub.on_resume) sub.on_resume();
    }
}
}  // namespace rocprofsys::control
