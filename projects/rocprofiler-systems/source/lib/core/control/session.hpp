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

    /// Register a trigger and seed its initial vote.
    /// Subscribers are NOT notified on initial registration; the resolved
    /// initial state is reflected only in is_active().
    void attach(trigger& trig);

    /// Called by triggers when their vote changes. Recomputes the resolved
    /// state (any-paused-wins) and fires pause/resume callbacks only on
    /// transitions.
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
    /// Used by consumers (e.g. roctx_client's marker gate) that combine a
    /// trigger-local rule with "no external trigger pausing us".
    [[nodiscard]] bool is_active_excluding(std::string_view name) const noexcept;

private:
    std::vector<vote_entry> m_votes;
    std::vector<subscriber> m_subscribers;
    std::atomic<bool>       m_active{ true };

    mutable std::mutex m_votes_mutex;
    std::mutex         m_subscribers_mutex;

    [[nodiscard]] bool resolve_locked() const noexcept;
    void               notify_pause();
    void               notify_resume();
};
}  // namespace rocprofsys::control
