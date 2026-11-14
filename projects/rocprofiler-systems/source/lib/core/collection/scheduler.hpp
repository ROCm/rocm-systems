// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/tick_grid.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

namespace rocprofsys::collection
{
/** Owns one thread that calls one callback on a tick_grid until stopped. */
class scheduler
{
public:
    /** Gets the timeline time (ns) of the tick. @return false to end the loop (the
     * thread exits). */
    using callback = std::function<bool(std::uint64_t timeline_ns)>;
    /** Runs once on the new thread before its first tick; a throw ends the loop. */
    using thread_start = std::function<void()>;

    /** @param name used in the one log line written when the callback throws */
    explicit scheduler(std::string name);
    ~scheduler();

    scheduler(const scheduler&)            = delete;
    scheduler& operator=(const scheduler&) = delete;
    scheduler(scheduler&&)                 = delete;
    scheduler& operator=(scheduler&&)      = delete;

    /** Starts the thread; on_start (may be empty) runs on it first, then the first tick
     * is immediate (unless paused). Once per scheduler.
     * @throws std::logic_error on a second call
     * @throws std::invalid_argument when interval <= 0 */
    void start(std::chrono::nanoseconds interval, callback func,
               thread_start on_start = {});
    /** Non-blocking: a running callback is not waited for. */
    void pause();
    void resume();
    /** request_stop + join; no-op if not started. Called on the thread itself (a
     * callback that finalizes): request_stop + detach, no join. */
    void stop();

private:
    void run(const std::stop_token& token, tick_grid& grid);
    /** One tick outside the lock, then moves the grid on or finishes it. */
    void fire(std::unique_lock<std::mutex>& lock, tick_grid& grid);
    /** Calls func(). A throw is logged. @return func()'s result; false when it threw */
    template <typename Fn>
    [[nodiscard]] bool guarded(Fn&& func) const noexcept;
    /** Logs that the loop ended; never throws. */
    void log_stopped(std::string_view reason) const noexcept;

    std::string  m_name;
    callback     m_callback;
    thread_start m_on_start;
    /** Guards m_grid, m_changed and m_pauses. */
    std::mutex                  m_mutex;
    std::condition_variable_any m_wake;
    std::optional<tick_grid>    m_grid;
    /** pause() or resume() was called since the last wait. */
    bool m_changed = false;
    /** Counts pause() calls; a change across a callback means the grid moved under it. */
    std::uint64_t m_pauses = 0;
    /** Last member: the thread uses the members above. */
    std::jthread m_thread;
};
}  // namespace rocprofsys::collection
