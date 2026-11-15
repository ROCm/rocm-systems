// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/fork_gate.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>

namespace rocprofsys::collection::detail
{
/** Settings shared by every source of one collector. allowed and thread_init are called
 * from every poll thread, concurrently: they must be thread-safe. */
struct run_options
{
    /** Each tick, inside the fork pass. false: skip the tick (the profiler is not
     * active). Empty: always allowed. A throw ends this source's polling. */
    std::function<bool()> allowed;
    /** Once per poll thread, at its start (before the first tick, also while paused),
     * inside a fork pass that waits for a fork in progress. May be empty. A throw ends
     * this source's polling. */
    std::function<void(std::string_view source)> thread_init;
    /** Timeline ns; a tick at or after it ends the poll thread. */
    std::uint64_t deadline_ns = std::numeric_limits<std::uint64_t>::max();
};

/** Owned by the collector; its runners keep a reference. */
struct run_context
{
    fork_gate&  gate;
    run_options options;
    /** Written by the collector only. */
    std::atomic<bool> paused{ false };
};

/** Polls one source on its own thread and interval. Private to the collector. */
class source_runner
{
public:
    source_runner(std::unique_ptr<source_interface> source,
                  std::chrono::nanoseconds interval, const run_context& context,
                  sink_interface& out);

    [[nodiscard]] source_interface& source() noexcept;

    /** Starts the poll thread; it runs thread_init first.
     * @throws std::invalid_argument when the interval <= 0 */
    void start();
    /** Non-blocking. */
    void pause();
    void resume();
    /** Stops the poll thread, then calls the source's release(). On the poll thread
     * itself (a poll that finalizes) release() is skipped and logged: poll() is still
     * on the stack. */
    void stop();

    /** One tick; public so it can be tested without a thread.
     * Order: past the deadline -> end; no fork pass, paused or not allowed -> skip;
     * poll with the timeline time read after these checks, so a tick that started
     * before a pause does not write inside the gap (a throw ends this source's loop).
     * @return false to end the loop */
    [[nodiscard]] bool tick(std::uint64_t timeline_ns);

private:
    /** Poll-thread start: records the thread, then runs thread_init in a waiting fork
     * pass (a child forked meanwhile would inherit any lock it holds). */
    void init_thread();
    /** allowed(); a throw is rethrown naming allowed() instead of poll(). */
    [[nodiscard]] bool is_allowed() const;

    std::unique_ptr<source_interface> m_source;
    std::chrono::nanoseconds          m_interval;
    const run_context&                m_context;
    sink_interface&                   m_out;
    /** Set at poll-thread start. */
    std::atomic<std::thread::id> m_poll_thread;
    /** Last member: its thread uses the members above. */
    scheduler m_scheduler;
};
}  // namespace rocprofsys::collection::detail
