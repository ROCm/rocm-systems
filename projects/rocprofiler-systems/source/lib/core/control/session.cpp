// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "core/state.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <string_view>
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
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    {
        const std::scoped_lock subs_lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }
    {
        const std::scoped_lock action_lock{ m_actions_mutex };
        for(auto& scoped : m_actions)
            scoped.clear();
        for(auto& a : m_active)
            a.store(true, std::memory_order_relaxed);
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
session::register_trigger(std::string_view name, action initial, scope event_scope)
{
    apply_locked_transition(
        [&] {
            m_actions[static_cast<std::size_t>(event_scope)][std::string{ name }] =
                initial;
        },
        name, event_scope);
}

void
session::unregister_trigger(std::string_view name, scope event_scope)
{
    apply_locked_transition(
        [&] {
            m_actions[static_cast<std::size_t>(event_scope)].erase(std::string{ name });
        },
        name, event_scope);
}

void
session::set_action(std::string_view name, action act, scope event_scope)
{
    apply_locked_transition(
        [&] {
            m_actions[static_cast<std::size_t>(event_scope)][std::string{ name }] = act;
        },
        name, event_scope);
}

void
session::apply_locked_transition(const std::function<void()>& mutate,
                                 std::string_view name, scope event_scope)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    // Serializes compute-then-notify across concurrent callers so subscribers
    // observe transitions in the same order they were computed. Deliberately
    // a separate mutex from m_actions_mutex (released below, before
    // notify_pause()/notify_resume() run) so a subscriber callback that
    // re-enters is_active()/is_active_without() cannot deadlock.
    const std::scoped_lock notify_lk{ m_notify_mutex };

    const auto scope_idx  = static_cast<std::size_t>(event_scope);
    bool       was_active = false;
    bool       now_active = false;
    {
        const std::scoped_lock action_lk{ m_actions_mutex };

        was_active = m_active[scope_idx].load(std::memory_order_relaxed);
        mutate();
        now_active = resolve_locked(event_scope);
        m_active[scope_idx].store(now_active, std::memory_order_relaxed);
    }

    if(was_active == now_active)
    {
        return;
    }

    LOG_DEBUG("session: trigger '{}' {} its scope", name,
              now_active ? "resumed" : "paused");

    if(now_active)
    {
        notify_resume(event_scope);
    }
    else
    {
        notify_pause(event_scope);
    }
}

// Any pause action within the given scope pauses that scope. Skip is
// ignored. With no actions for the scope, the scope is active by default.
bool
session::resolve_locked(scope event_scope) const noexcept
{
    const auto& scoped = m_actions[static_cast<std::size_t>(event_scope)];
    return std::none_of(scoped.begin(), scoped.end(),
                        [](const auto& kv) { return kv.second == action::pause; });
}

bool
session::is_active_without(std::string_view name, scope event_scope) const noexcept
{
    const std::scoped_lock lk{ m_actions_mutex };
    const auto&            scoped = m_actions[static_cast<std::size_t>(event_scope)];
    return std::none_of(scoped.begin(), scoped.end(), [name](const auto& kv) {
        return kv.first != name && kv.second == action::pause;
    });
}

namespace
{
bool
listens_to(const subscriber& sub, scope event_scope)
{
    return std::find(sub.scopes.begin(), sub.scopes.end(), event_scope) !=
           sub.scopes.end();
}
}  // namespace

void
session::notify_pause(scope event_scope)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(!listens_to(sub, event_scope)) continue;
        LOG_DEBUG("session: pausing subscriber '{}'", sub.name);
        if(sub.on_pause)
        {
            sub.on_pause();
        }
    }
}

void
session::notify_resume(scope event_scope)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_subscribers_mutex };
    for(const auto& sub : m_subscribers)
    {
        if(!listens_to(sub, event_scope)) continue;
        const bool all_active =
            std::all_of(sub.scopes.begin(), sub.scopes.end(),
                        [this](scope listened) { return is_active(listened); });
        if(!all_active) continue;
        LOG_DEBUG("session: resuming subscriber '{}'", sub.name);
        if(sub.on_resume)
        {
            sub.on_resume();
        }
    }
}
}  // namespace rocprofsys::control
