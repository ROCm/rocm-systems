// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "subscriber.hpp"
#include "trigger.hpp"
#include "vote_entry.hpp"

#include <atomic>
#include <mutex>
#include <string_view>
#include <vector>

namespace rocprofsys::control
{
class session
{
public:
    session()  = default;
    ~session() = default;

    session(const session&)            = delete;
    session& operator=(const session&) = delete;
    session(session&&)                 = delete;
    session& operator=(session&&)      = delete;

    void shutdown();

    void subscribe(subscriber sub);

    /// Register a trigger and seed its initial vote. Calling attach() twice
    /// for the same trigger name updates the existing entry rather than
    /// creating a duplicate (same find-or-insert rule as publish()).
    /// Subscribers are NOT notified on initial registration; the resolved
    /// initial state is reflected only in is_active().
    void attach(const trigger& trig);

    /// Remove a trigger's vote so it no longer contributes to resolution.
    /// Triggers that can be destroyed independently of the session (i.e.
    /// any trigger not guaranteed to share the session's exact lifetime)
    /// must call this from their destructor.
    void detach(const trigger& trig);

    /// Called by triggers when their vote changes. Recomputes the resolved
    /// state (any-paused-wins) and fires pause/resume callbacks only on
    /// transitions. Serialized across concurrent callers (via a dedicated
    /// notify mutex, not m_votes_mutex) so notifications are delivered in
    /// the same order their state transitions were computed - m_votes_mutex
    /// itself is still released before subscriber callbacks run, so a
    /// callback that re-enters is_active()/is_active_excluding() cannot
    /// deadlock.
    void publish(const trigger& trig, vote new_vote);

    /// If the session is currently paused, fire pause on all subscribers
    /// to reflect the initial state. Subscribers default to "running", so
    /// only the paused-initial case needs to be broadcast.
    void force_initial_pause();

    [[nodiscard]] bool is_active() const noexcept
    {
        return m_active.load(std::memory_order_relaxed);
    }

    /// True iff every trigger except @p name has voted active or abstain.
    /// Not yet consumed by any trigger in this PR (roctx's marker gate is
    /// still purely local) - intended for a later PR's "gate trace windows
    /// at source" composition, where a trigger's own write decision must
    /// also respect other triggers' votes without double-counting its own.
    [[nodiscard]] bool is_active_excluding(std::string_view name) const noexcept;

private:
    std::vector<vote_entry> m_votes;
    std::vector<subscriber> m_subscribers;
    std::atomic<bool>       m_active{ true };

    mutable std::mutex m_votes_mutex;
    std::mutex         m_subscribers_mutex;
    std::mutex         m_notify_mutex;

    [[nodiscard]] bool       resolve_locked() const noexcept;
    vote_entry&              find_or_insert_locked(std::string_view name, vote v);
    void                     notify_pause();
    void                     notify_resume();
};
}  // namespace rocprofsys::control
