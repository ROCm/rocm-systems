// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "session.hpp"

#include "core/state.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::control
{
namespace
{
using callback_list = std::vector<std::function<void()>>;

void
invoke_all(const callback_list& callbacks)
{
    for(const auto& callback : callbacks)
    {
        callback();
    }
}

bool
listens_to(const subscriber& sub, scope event_scope)
{
    return sub.scopes.contains(event_scope);
}
}  // namespace

session::session() noexcept
{
    for(auto& tracing_flag : m_scope_tracing)
    {
        tracing_flag.store(true, std::memory_order_release);
    }
}

void
session::shutdown()
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    const std::scoped_lock notify_lk{ m_notify_mutex };
    {
        const std::scoped_lock subs_lk{ m_subscribers_mutex };
        m_subscribers.clear();
    }
    {
        const std::scoped_lock action_lock{ m_actions_mutex };
        for(auto& scoped : m_actions)
        {
            scoped.clear();
        }
        for(auto& tracing_flag : m_scope_tracing)
        {
            tracing_flag.store(true, std::memory_order_release);
        }
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
    //
    // WARNING: on_pause/on_resume run with this mutex still held (see
    // notify_pause/notify_resume below). A callback must never call
    // set_action() itself, directly or transitively - that re-enters
    // m_notify_mutex and deadlocks.
    const std::scoped_lock notify_lk{ m_notify_mutex };

    const auto scope_idx  = static_cast<std::size_t>(event_scope);
    bool       was_active = false;
    bool       now_active = false;
    {
        const std::scoped_lock action_lk{ m_actions_mutex };

        was_active = m_scope_tracing[scope_idx].load(std::memory_order_acquire);
        mutate();
        now_active = resolve_locked(event_scope);
        m_scope_tracing[scope_idx].store(now_active, std::memory_order_release);
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
    return std::ranges::none_of(
        scoped, [](const auto& entry) { return entry.second == action::pause; });
}

bool
session::is_active_without(std::string_view name, scope event_scope) const noexcept
{
    const std::scoped_lock actions_lk{ m_actions_mutex };
    const auto&            scoped = m_actions[static_cast<std::size_t>(event_scope)];
    return std::ranges::none_of(scoped, [name](const auto& entry) {
        return entry.first != name && entry.second == action::pause;
    });
}

callback_list
session::collect_pause_callbacks(scope event_scope)
{
    callback_list          to_fire;
    const std::scoped_lock subs_lk{ m_subscribers_mutex };
    to_fire.reserve(m_subscribers.size());
    for(const auto& sub : m_subscribers)
    {
        if(!listens_to(sub, event_scope))
        {
            continue;
        }
        LOG_DEBUG("session: pausing subscriber '{}'", sub.name);
        if(sub.on_pause)
        {
            to_fire.push_back(sub.on_pause);
        }
    }
    return to_fire;
}

void
session::notify_pause(scope event_scope)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    invoke_all(collect_pause_callbacks(event_scope));
}

callback_list
session::collect_resume_callbacks(scope event_scope)
{
    callback_list          to_fire;
    const std::scoped_lock subs_lk{ m_subscribers_mutex };
    to_fire.reserve(m_subscribers.size());
    for(const auto& sub : m_subscribers)
    {
        if(!listens_to(sub, event_scope))
        {
            continue;
        }
        const bool all_active =
            sub.scopes.all_of([this](scope listened) { return is_active(listened); });
        if(!all_active)
        {
            continue;
        }
        LOG_DEBUG("session: resuming subscriber '{}'", sub.name);
        if(sub.on_resume)
        {
            to_fire.push_back(sub.on_resume);
        }
    }
    return to_fire;
}

void
session::notify_resume(scope event_scope)
{
    const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
    invoke_all(collect_resume_callbacks(event_scope));
}
}  // namespace rocprofsys::control
