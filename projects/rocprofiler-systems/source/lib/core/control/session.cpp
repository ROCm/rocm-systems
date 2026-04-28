// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>

namespace rocprofsys::control
{
session::session() noexcept
{
    for(auto& a : m_active)
        a.store(true, std::memory_order_relaxed);
}

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
        for(auto& a : m_active)
            a.store(true, std::memory_order_relaxed);
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
    const auto event_scope = trig.event_scope();

    std::scoped_lock const lk{ m_votes_mutex };
    m_votes.push_back({ trig.name(), event_scope, trig.initial_vote() });
    m_active[static_cast<std::size_t>(event_scope)].store(resolve_locked(event_scope),
                                                          std::memory_order_relaxed);
}

void
session::force_initial_pause()
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        const bool any_paused_for_sub =
            std::any_of(sub.scopes.begin(), sub.scopes.end(),
                        [this](scope s) { return !is_active(s); });

        if(any_paused_for_sub && sub.on_pause) sub.on_pause();
    }
}

void
session::publish(const trigger& trig, vote new_vote)
{
    const auto event_scope = trig.event_scope();
    const auto scope_idx   = static_cast<std::size_t>(event_scope);
    const auto name        = trig.name();
    bool       was_active  = false;
    bool       now_active  = false;
    {
        std::scoped_lock const lk{ m_votes_mutex };
        was_active = m_active[scope_idx].load(std::memory_order_relaxed);

        auto it = std::find_if(m_votes.begin(), m_votes.end(),
                               [name, event_scope](const vote_entry& e) {
                                   return e.name == name && e.event_scope == event_scope;
                               });
        if(it == m_votes.end())
            m_votes.push_back({ name, event_scope, new_vote });
        else
            it->current_vote = new_vote;

        now_active = resolve_locked(event_scope);
        m_active[scope_idx].store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active) return;
    if(now_active)
        notify_resume(event_scope);
    else
        notify_pause(event_scope);
}

// Any paused vote (within the given scope) pauses the scope. Abstain is
// ignored. With no votes for the scope, the scope is active by default.
bool
session::resolve_locked(scope event_scope) const noexcept
{
    for(const auto& entry : m_votes)
    {
        if(entry.event_scope != event_scope) continue;
        if(entry.current_vote == vote::paused) return false;
    }
    return true;
}

bool
session::is_active_excluding(std::string_view name, scope event_scope) const noexcept
{
    std::scoped_lock const lk{ m_votes_mutex };
    for(const auto& entry : m_votes)
    {
        if(entry.event_scope != event_scope) continue;
        if(entry.name == name) continue;
        if(entry.current_vote == vote::paused) return false;
    }
    return true;
}

void
session::notify_pause(scope event_scope)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        const bool listens = std::find(sub.scopes.begin(), sub.scopes.end(),
                                       event_scope) != sub.scopes.end();
        if(listens && sub.on_pause) sub.on_pause();
    }
}

void
session::notify_resume(scope event_scope)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        const bool listens = std::find(sub.scopes.begin(), sub.scopes.end(),
                                       event_scope) != sub.scopes.end();
        if(listens && sub.on_resume) sub.on_resume();
    }
}
}  // namespace rocprofsys::control
