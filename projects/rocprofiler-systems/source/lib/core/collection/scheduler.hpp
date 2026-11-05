// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/clocks.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>

namespace rocprofsys::collection
{
/**
 * Calls one callback on a drift-free grid on its own thread; a missed tick is skipped.
 * The callback gets the timeline time at which the tick started. A callback that throws
 * is reported once and not called again.
 *
 * @tparam Clock a std-style clock the grid is computed on
 */
template <typename Clock = std::chrono::steady_clock>
class scheduler
{
public:
    using callback = std::function<void(std::uint64_t timeline_ns)>;

    /** `gate` returning false skips a tick. */
    explicit scheduler(std::string name, std::function<void()> thread_init = {},
                       std::function<bool()> gate = {})
    : m_name(std::move(name))
    , m_thread_init(std::move(thread_init))
    , m_gate(std::move(gate))
    {}

    ~scheduler() { stop(); }

    scheduler(const scheduler&)            = delete;
    scheduler& operator=(const scheduler&) = delete;
    scheduler(scheduler&&)                 = delete;
    scheduler& operator=(scheduler&&)      = delete;

    void start(std::chrono::nanoseconds interval, callback func)
    {
        {
            const std::lock_guard lock(m_mutex);
            if(m_thread.joinable())
            {
                return;
            }
            m_callback = std::move(func);
            m_paused   = false;
            m_stopping = false;
            m_restart  = false;
            m_next     = Clock::now();
            m_interval = interval;
            m_failed.store(false);
        }
        m_thread = std::jthread([this](const std::stop_token& token) { run(token); });
    }

    /** Does not wait for a running callback. @return the time after which none starts. */
    std::uint64_t pause()
    {
        std::uint64_t paused_at = 0;
        {
            const std::lock_guard lock(m_mutex);
            m_paused  = true;
            paused_at = clocks::timeline::now_ns();
        }
        m_wake.notify_all();
        return paused_at;
    }

    void resume()
    {
        {
            const std::lock_guard lock(m_mutex);
            if(!m_paused)
            {
                return;
            }
            m_paused  = false;
            m_restart = true;
        }
        m_wake.notify_all();
    }

    /**
     * Waits for a running callback and keeps further ones from starting until
     * `unblock()`, independently of `pause()`. A callback that blocks its own scheduler
     * (it makes the fork) is not waited for.
     */
    void block()
    {
        {
            const std::lock_guard lock(m_mutex);
            m_blocked = true;
        }
        m_wake.notify_all();
        if(std::this_thread::get_id() == m_worker.load())
        {
            return;
        }
        std::unique_lock lock(m_mutex);
        m_idle.wait(lock, [this] { return !m_running; });
    }

    void unblock()
    {
        {
            const std::lock_guard lock(m_mutex);
            if(!m_blocked)
            {
                return;
            }
            m_blocked = false;
            m_restart = true;
        }
        m_wake.notify_all();
    }

    void stop()
    {
        {
            const std::lock_guard lock(m_mutex);
            m_stopping = true;
        }
        if(m_thread.joinable())
        {
            m_thread.request_stop();
            m_thread.join();
        }
    }

private:
    static constexpr std::chrono::nanoseconds k_min_interval{ 10'000 };

    void run(const std::stop_token& token)
    {
        m_worker.store(std::this_thread::get_id());
        if(m_thread_init)
        {
            m_thread_init();
        }

        std::unique_lock lock(m_mutex);
        while(!token.stop_requested())
        {
            if(std::exchange(m_restart, false))
            {
                m_next = Clock::now();
            }
            if(is_idle())
            {
                wait_until_runnable(lock, token);
                continue;
            }
            if(wait_for_tick(lock, token))
            {
                continue;  // paused, resumed or a new interval: look at the state again
            }
            if(token.stop_requested())
            {
                break;
            }

            lock.unlock();
            fire();
            lock.lock();
        }
    }

    void wait_until_runnable(std::unique_lock<std::mutex>& lock,
                             const std::stop_token&        token)
    {
        m_wake.wait(lock, token, [this] { return m_restart || !is_idle(); });
    }

    // @return true when woken by a state change before the tick
    bool wait_for_tick(std::unique_lock<std::mutex>& lock, const std::stop_token& token)
    {
        return m_wake.wait_until(lock, token, m_next,
                                 [this] { return m_restart || m_paused || m_blocked; });
    }

    // The caller holds `m_mutex`.
    [[nodiscard]] bool is_idle() const
    {
        return m_paused || m_blocked || m_failed.load();
    }

    void fire()
    {
        if(m_gate && !m_gate())
        {
            reschedule();
            return;
        }

        std::uint64_t started_at = 0;
        if(!begin_callback(started_at))
        {
            return;
        }
        const bool done = invoke(started_at);
        end_callback();
        if(done)
        {
            reschedule();
        }
    }

    [[nodiscard]] bool begin_callback(std::uint64_t& started_at)
    {
        const std::lock_guard lock(m_mutex);
        if(m_paused || m_blocked || m_stopping)
        {
            return false;
        }
        started_at = clocks::timeline::now_ns();
        m_running  = true;
        return true;
    }

    [[nodiscard]] bool invoke(std::uint64_t started_at)
    {
        try
        {
            m_callback(started_at);
        } catch(const std::exception& error)
        {
            fail(error.what());
            return false;
        }
        return true;
    }

    void end_callback()
    {
        {
            const std::lock_guard lock(m_mutex);
            m_running = false;
        }
        m_idle.notify_all();
    }

    // The first grid point (previous + k * interval) after now.
    void reschedule()
    {
        using duration = Clock::duration;

        const auto interval = std::max(m_interval, k_min_interval);
        const auto step     = std::chrono::duration_cast<duration>(interval);
        const auto now      = Clock::now();

        const std::lock_guard lock(m_mutex);
        m_next += step * ((now - m_next) / step + 1);
    }

    void fail(const char* what)
    {
        m_failed.store(true);
        LOG_ERROR("collection: '{}' stopped polling: {}", m_name, what);
    }

    std::string           m_name;
    std::function<void()> m_thread_init;
    std::function<bool()> m_gate;
    callback              m_callback;

    std::mutex                   m_mutex;  // guards the flags, m_interval and m_next
    std::condition_variable_any  m_wake;
    std::condition_variable_any  m_idle;  // m_running became false
    std::atomic<std::thread::id> m_worker;
    bool                         m_paused   = false;
    bool                         m_blocked  = false;
    bool                         m_running  = false;
    bool                         m_stopping = false;
    bool                         m_restart  = false;
    std::chrono::nanoseconds     m_interval{};
    std::atomic<bool>            m_failed{ false };
    Clock::time_point            m_next{};
    std::jthread                 m_thread;
};

template <typename Clock = std::chrono::steady_clock>
using scheduler_factory =
    std::function<std::unique_ptr<scheduler<Clock>>(std::string name)>;
}  // namespace rocprofsys::collection
