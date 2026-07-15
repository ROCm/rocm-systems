// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "logger/debug.hpp"

#include <algorithm>
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
        const std::scoped_lock lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }
    {
        const std::scoped_lock lk{ m_votes_mutex };
        m_votes.clear();
        m_active.store(true, std::memory_order_relaxed);
    }
}

void
session::subscribe(subscriber sub)
{
    const std::scoped_lock lk{ m_subscribers_mutex };
    m_subscribers.push_back(std::move(sub));
}

void
session::attach(const trigger& trig)
{
    const std::scoped_lock lk{ m_votes_mutex };
    find_or_insert_locked(trig.name(), trig.initial_vote());
    m_active.store(resolve_locked(), std::memory_order_relaxed);
}

void
session::detach(const trigger& trig)
{
    const std::scoped_lock lk{ m_votes_mutex };
    const auto              name = trig.name();
    m_votes.erase(std::remove_if(m_votes.begin(), m_votes.end(),
                                  [name](const vote_entry& e) { return e.name == name; }),
                  m_votes.end());
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
    // Serializes compute-then-notify across concurrent publish() callers so
    // subscribers observe transitions in the same order they were computed.
    // Deliberately a separate mutex from m_votes_mutex (released below,
    // before notify_pause()/notify_resume() run) so a subscriber callback
    // that re-enters is_active()/is_active_excluding() cannot deadlock.
    const std::scoped_lock notify_lk{ m_notify_mutex };

    bool was_active = false;
    bool now_active = false;
    {
        const std::scoped_lock lk{ m_votes_mutex };
        was_active = m_active.load(std::memory_order_relaxed);
        find_or_insert_locked(trig.name(), new_vote);
        now_active = resolve_locked();
        m_active.store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active) return;
    if(now_active)
        notify_resume();
    else
        notify_pause();
}

vote_entry&
session::find_or_insert_locked(std::string_view name, vote v)
{
    auto it = std::find_if(m_votes.begin(), m_votes.end(),
                            [name](const vote_entry& e) { return e.name == name; });
    if(it == m_votes.end())
    {
        m_votes.push_back({ std::string{ name }, v });
        return m_votes.back();
    }
    it->current_vote = v;
    return *it;
}

// Any paused vote pauses the session. Abstain is ignored.
// With no votes the session is active by default.
bool
session::resolve_locked() const noexcept
{
    return std::none_of(m_votes.begin(), m_votes.end(),
                         [](const vote_entry& e) { return e.current_vote == vote::paused; });
}

bool
session::is_active_excluding(std::string_view name) const noexcept
{
    // Cold path: not yet called in this PR (see header doc); when a caller
    // arrives it will hit this after its own fast atomic check, with only
    // 1-2 votes attached, so a locked linear scan remains appropriate.
    const std::scoped_lock lk{ m_votes_mutex };
    return std::none_of(m_votes.begin(), m_votes.end(), [name](const vote_entry& e) {
        return e.name != name && e.current_vote == vote::paused;
    });
}

void
session::notify_pause()
{
    const std::scoped_lock lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        LOG_DEBUG("session: pausing subscriber '{}'", sub.name);
        if(sub.on_pause) sub.on_pause();
    }
}

void
session::notify_resume()
{
    const std::scoped_lock lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        LOG_DEBUG("session: resuming subscriber '{}'", sub.name);
        if(sub.on_resume) sub.on_resume();
    }
}
}  // namespace rocprofsys::control
