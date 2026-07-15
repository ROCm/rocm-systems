// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "subscriber.hpp"
#include "trigger.hpp"

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
    void register_trigger(const trigger& trig);

    /// Remove a trigger's vote so it no longer contributes to resolution.
    void unregister_trigger(const trigger& trig);

    void publish_vote(const trigger& trig, vote new_vote);

    /// If the session is currently paused, fire pause on all subscribers
    /// to reflect the initial state. Subscribers default to "running", so
    /// only the paused-initial case needs to be broadcast.
    void force_initial_pause();

    [[nodiscard]] bool is_active() const noexcept
    {
        return m_active.load(std::memory_order_relaxed);
    }

    /// True iff every trigger except @p name has voted active or abstain.
    /// Used where a trigger's own write decision must
    /// also respect other triggers' votes without double-counting its own.
    [[nodiscard]] bool is_active_excluding_trigger(std::string_view name) const noexcept;

private:
    std::vector<vote_entry> m_votes;
    std::vector<subscriber> m_subscribers;
    std::atomic<bool>       m_active{ true };

    mutable std::mutex m_votes_mutex;
    std::mutex         m_subscribers_mutex;
    std::mutex         m_notify_mutex;

    [[nodiscard]] bool resolve_locked() const noexcept;
    vote_entry&        find_or_insert_locked(std::string_view name, vote v);
    void               notify_pause();
    void               notify_resume();
};
}  // namespace rocprofsys::control
