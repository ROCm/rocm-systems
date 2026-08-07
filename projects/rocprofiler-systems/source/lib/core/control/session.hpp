// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <functional>
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

struct subscriber
{
    std::function<void()> on_pause;
    std::function<void()> on_resume;
    std::string           name;
};

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

    /// Seed a trigger's action. @p name identifies the trigger for the
    /// lifetime of its registration.
    void register_trigger(std::string_view name, action initial);

    void unregister_trigger(std::string_view name);

    void set_action(std::string_view name, action act);

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
