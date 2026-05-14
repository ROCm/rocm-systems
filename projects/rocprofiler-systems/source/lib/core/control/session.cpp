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
        for(const auto& sub : m_subscribers)
            sub.paused = false;
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
    // Seed the subscriber's resolved paused state from current session state
    // so a subscribe-after-attach ordering observes the right transitions.
    // No callback is fired here; force_initial_pause() is the explicit
    // broadcast for the production ordering.
    sub.paused = subscriber_should_be_paused(sub);

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
    // Explicit broadcast of the current paused state to all subscribers whose
    // net resolved state is paused, regardless of any per-subscriber paused
    // tracking. Production callers run this after attach(...) has already
    // computed the initial scope state but before subscribers have had a
    // chance to observe it (or after subscribe-after-attach has seeded the
    // tracking field). dispatch_for_scope's transition-only firing is correct
    // for runtime publish() but would silently skip subscribers whose seeded
    // state already matches the resolved state — we always want them to
    // observe the initial pause once.
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(!subscriber_should_be_paused(sub)) continue;
        sub.paused = true;
        if(sub.on_pause) sub.on_pause();
    }
}

void
session::publish(const trigger& trig, vote new_vote)
{
    const auto event_scope = trig.event_scope();
    const auto scope_idx   = static_cast<std::size_t>(event_scope);
    const auto name        = trig.name();
    {
        std::scoped_lock const lk{ m_votes_mutex };

        auto it = std::find_if(m_votes.begin(), m_votes.end(),
                               [name, event_scope](const vote_entry& e) {
                                   return e.name == name && e.event_scope == event_scope;
                               });
        if(it == m_votes.end())
            m_votes.push_back({ name, event_scope, new_vote });
        else
            it->current_vote = new_vote;

        m_active[scope_idx].store(resolve_locked(event_scope), std::memory_order_relaxed);
    }

    // A subscriber's resolved paused state can change even when this scope's
    // active state did not (multi-scope subscriber crossing the OR boundary),
    // so we always run dispatch and let it dedupe per-subscriber.
    dispatch_for_scope(event_scope);
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

bool
session::subscriber_should_be_paused(const subscriber& sub) const noexcept
{
    return std::any_of(sub.scopes.begin(), sub.scopes.end(),
                       [this](scope s) { return !is_active(s); });
}

void
session::dispatch_for_scope(scope event_scope)
{
    std::scoped_lock const lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        const bool listens = std::find(sub.scopes.begin(), sub.scopes.end(),
                                       event_scope) != sub.scopes.end();
        if(!listens) continue;

        const bool should_be_paused = subscriber_should_be_paused(sub);
        if(should_be_paused == sub.paused) continue;

        sub.paused = should_be_paused;
        if(should_be_paused)
        {
            if(sub.on_pause) sub.on_pause();
        }
        else
        {
            if(sub.on_resume) sub.on_resume();
        }
    }
}
}  // namespace rocprofsys::control
