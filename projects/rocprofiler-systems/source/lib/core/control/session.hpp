// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "subscriber.hpp"
#include "trigger.hpp"
#include "vote_entry.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string_view>
#include <vector>

namespace rocprofsys::control
{
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

    void subscribe(subscriber sub);

    /// Register a trigger and seed its initial vote into the trigger's scope.
    /// Subscribers are NOT notified on initial registration.
    void attach(trigger& trig);

    /// Called by triggers when their vote changes. Recomputes the resolved
    /// state for the trigger's scope (any-paused-wins) and fires pause/resume
    /// on subscribers whose `scopes` contains that scope, only on transitions.
    void publish(const trigger& trig, vote new_vote);

    /// If any scope is currently paused, fire pause once on each subscriber
    /// whose `scopes` overlaps a paused scope. Subscribers default to running,
    /// so only the paused-initial case needs broadcasting. Each subscriber
    /// is fired at most once even if multiple of its scopes are paused.
    void force_initial_pause();

    [[nodiscard]] bool is_active(scope event_scope = scope::global) const noexcept
    {
        // acquire pairs with the release store under m_votes_mutex; ensures
        // a reader that observes a transition also observes any subscriber-
        // visible state writes ordered before the publish that caused it.
        return m_active[static_cast<std::size_t>(event_scope)].load(
            std::memory_order_acquire);
    }

    /// True iff every trigger of @p event_scope except @p name has voted
    /// active or abstain. Used by consumers (e.g. roctx_client's marker gate)
    /// that combine a trigger-local rule with "no other trigger pausing us".
    [[nodiscard]] bool is_active_excluding(
        std::string_view name, scope event_scope = scope::global) const noexcept;

    /// True iff a trigger with @p name has registered a vote for @p event_scope.
    /// Used by late-initializing subscribers that need to distinguish "the
    /// trigger has not been attached yet" from "the trigger is attached and
    /// currently active".
    [[nodiscard]] bool has_trigger(std::string_view name,
                                   scope event_scope = scope::global) const noexcept;

private:
    static constexpr std::size_t SCOPE_COUNT = static_cast<std::size_t>(scope::COUNT_);

    std::vector<vote_entry>                    m_votes;
    std::vector<subscriber>                    m_subscribers;
    std::array<std::atomic<bool>, SCOPE_COUNT> m_active{};

    mutable std::mutex m_votes_mutex;
    std::mutex         m_subscribers_mutex;

    [[nodiscard]] bool resolve_locked(scope event_scope) const noexcept;
    [[nodiscard]] bool subscriber_should_be_paused(const subscriber& sub) const noexcept;
    void               dispatch_for_scope(scope event_scope);
};
}  // namespace rocprofsys::control
