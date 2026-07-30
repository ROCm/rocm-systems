// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
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
    sampling_only,
    count_,  // sentinel: number of scopes
};

struct subscriber
{
    std::function<void()> on_pause;
    std::function<void()> on_resume;
    std::string           name;
    std::vector<scope>    scopes = { scope::global };
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

    void unregister_trigger(std::string_view name);

    void set_action(std::string_view name, action act);

    [[nodiscard]] bool is_active(scope event_scope = scope::global) const noexcept
    {
        assert(static_cast<std::size_t>(event_scope) < scope_count);
        return m_active[static_cast<std::size_t>(event_scope)].load(
            std::memory_order_relaxed);
    }

    /// True iff every trigger of @p event_scope except @p name currently has
    /// a trace/skip action. Used where a trigger's own write decision must
    /// also respect other triggers' actions without double-counting its own.
    [[nodiscard]] bool is_active_excluding_trigger(
        std::string_view name, scope event_scope = scope::global) const noexcept;

private:
    static constexpr std::size_t scope_count = static_cast<std::size_t>(scope::count_);

    struct entry
    {
        action act{ action::trace };
        scope  event_scope{ scope::global };
    };

    std::unordered_map<std::string, entry>     m_actions;
    std::vector<subscriber>                    m_subscribers;
    std::array<std::atomic<bool>, scope_count> m_active{};

    mutable std::mutex m_actions_mutex;
    std::mutex         m_subscribers_mutex;
    std::mutex         m_notify_mutex;

    [[nodiscard]] bool resolve_locked(scope event_scope) const noexcept;
    void               notify_pause(scope event_scope);
    void               notify_resume(scope event_scope);

    /// Applies @p mutate to m_actions under lock, recomputes the active
    /// state for the scope it reports affected (nullopt = no-op), and
    /// broadcasts if that state changed. Shared by register_trigger(),
    /// unregister_trigger(), and set_action().
    void apply_locked_transition(const std::function<std::optional<scope>()>& mutate,
                                 std::string_view                             name);
};
}  // namespace rocprofsys::control
