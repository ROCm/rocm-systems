// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "subscriber.hpp"
#include "trigger.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
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

    /// Register a trigger and seed its initial action. The returned setter
    /// is the only way to update this trigger's action afterward.
    [[nodiscard]] trigger::action_setter register_trigger(const trigger& trig);

    /// Remove a trigger's action so it no longer contributes to resolution.
    void unregister_trigger(const trigger& trig);

    /// If the session is currently paused, fire pause on all subscribers
    /// to reflect the initial state. Subscribers default to "running", so
    /// only the paused-initial case needs to be broadcast.
    void force_initial_pause();

    [[nodiscard]] bool is_active() const noexcept
    {
        return m_active.load(std::memory_order_relaxed);
    }

private:
    std::unordered_map<std::string, action> m_actions;
    std::vector<subscriber>                 m_subscribers;
    std::atomic<bool>                       m_active{ true };

    mutable std::mutex m_actions_mutex;
    std::mutex         m_subscribers_mutex;
    std::mutex         m_notify_mutex;

    [[nodiscard]] bool resolve_locked() const noexcept;
    void               notify_pause();
    void               notify_resume();
};
}  // namespace rocprofsys::control
