// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::control
{
/// What a single trigger currently wants. `skip` and `trace` both mean "not
/// asking for a pause"; only `pause` affects resolution.
enum class action
{
    skip,
    trace,
    pause
};

/// A trigger's blast radius: which subscribers its actions can reach.
enum class scope : std::size_t
{
    global = 0,
    sampling,
    count_,  // sentinel: number of scopes
};

inline constexpr std::size_t SCOPE_COUNT = static_cast<std::size_t>(scope::count_);

class scope_set
{
public:
    scope_set() noexcept { m_bits.set(static_cast<std::size_t>(scope::global)); }

    scope_set(std::initializer_list<scope> scopes) noexcept
    {
        for(const auto event_scope : scopes)
        {
            m_bits.set(static_cast<std::size_t>(event_scope));
        }
    }

    [[nodiscard]] bool contains(scope event_scope) const noexcept
    {
        return m_bits.test(static_cast<std::size_t>(event_scope));
    }

    template <typename Predicate>
    [[nodiscard]] bool all_of(Predicate&& pred) const
    {
        for(std::size_t i = 0; i < SCOPE_COUNT; ++i)
        {
            if(m_bits.test(i) && !pred(static_cast<scope>(i)))
            {
                return false;
            }
        }
        return true;
    }

    template <typename Predicate>
    [[nodiscard]] bool any_of(Predicate&& pred) const
    {
        for(std::size_t i = 0; i < SCOPE_COUNT; ++i)
        {
            if(m_bits.test(i) && pred(static_cast<scope>(i)))
            {
                return true;
            }
        }
        return false;
    }

private:
    std::bitset<SCOPE_COUNT> m_bits;
};

struct subscriber
{
    std::function<void()> on_pause;
    std::function<void()> on_resume;
    std::string           name;
    scope_set             scopes;
};

class session
{
public:
    session() noexcept;
    ~session() = default;

    session(const session&)            = delete;
    session& operator=(const session&) = delete;
    session(session&&)                 = delete;
    session& operator=(session&&)      = delete;

    void shutdown();

    /// Register a subscriber. Callers MUST subscribe every subscriber
    /// before constructing any trigger: register_trigger()/unregister_trigger()
    /// broadcast on a pause/resume transition at the moment they're called, so
    /// a subscriber that joins after a trigger's initial vote already flipped
    /// the session will miss that broadcast.
    void subscribe(subscriber sub);

    /// Seed a trigger's action. @p name identifies the trigger for the
    /// lifetime of its registration; @p event_scope fixes which subscribers
    /// its later actions can reach. Broadcasts to subscribers of that scope
    /// if this registration changes the scope's active/paused state.
    void register_trigger(std::string_view name, action initial,
                          scope event_scope = scope::global);

    void unregister_trigger(std::string_view name, scope event_scope = scope::global);

    void set_action(std::string_view name, action act, scope event_scope = scope::global);

    [[nodiscard]] bool is_active(scope event_scope = scope::global) const noexcept
    {
        assert(static_cast<std::size_t>(event_scope) < SCOPE_COUNT);
        return m_scope_tracing[static_cast<std::size_t>(event_scope)].load(
            std::memory_order_acquire);
    }

    /// True iff every trigger of @p event_scope except @p name currently has
    /// a trace/skip action. Used where a trigger's own write decision must
    /// also respect other triggers' actions without double-counting its own.
    [[nodiscard]] bool is_active_without(
        std::string_view name, scope event_scope = scope::global) const noexcept;

private:
    using scoped_actions = std::unordered_map<std::string, action>;

    std::array<scoped_actions, SCOPE_COUNT>    m_actions;
    std::vector<subscriber>                    m_subscribers;
    std::array<std::atomic<bool>, SCOPE_COUNT> m_scope_tracing{};

    mutable std::mutex m_actions_mutex;
    std::mutex         m_subscribers_mutex;
    std::mutex         m_notify_mutex;

    [[nodiscard]] bool resolve_locked(scope event_scope) const noexcept;
    void               notify_pause(scope event_scope);
    void               notify_resume(scope event_scope);

    [[nodiscard]] std::vector<std::function<void()>> collect_pause_callbacks(
        scope event_scope);
    [[nodiscard]] std::vector<std::function<void()>> collect_resume_callbacks(
        scope event_scope);

    /// Applies @p mutate to the scoped action map under lock, recomputes
    /// that scope's active state, and broadcasts if it changed. Shared by
    /// register_trigger(), unregister_trigger(), and set_action().
    void apply_locked_transition(const std::function<void()>& mutate,
                                 std::string_view name, scope event_scope);
};
}  // namespace rocprofsys::control
