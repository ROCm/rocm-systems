// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

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
    m_votes[&trig] = trig.initial_vote();
    m_active.store(resolve_locked(), std::memory_order_relaxed);
}

void
session::force_initial_pause()
{
    if(is_active()) return;
    notify_pause();
}

void
session::publish(const trigger& trig, vote v)
{
    bool was_active = false;
    bool now_active = false;
    {
        std::scoped_lock const lk{ m_votes_mutex };
        was_active     = m_active.load(std::memory_order_relaxed);
        m_votes[&trig] = v;
        now_active     = resolve_locked();
        m_active.store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active) return;
    if(now_active)
        notify_resume();
    else
        notify_pause();
}

// Unanimous-active: paused iff at least one trigger voted paused.
// Abstain votes are ignored. With no votes (no triggers attached), the
// session is active by default.
bool
session::resolve_locked() const
{
    for(const auto& [trig, v] : m_votes)
    {
        if(v == vote::paused) return false;
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
