// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/detail/source_runner.hpp"
#include "core/collection/fork_gate.hpp"
#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace rocprofsys::collection
{
/**
 * Owns the sources of one sampler and polls each on its own thread and interval.
 *
 * Caller contract:
 * - add_source(), start(), stop() and the destructor are called by one thread at a time
 *   (the owner serializes them).
 * - pause() and resume() may be called from any thread at any time, also concurrently
 *   with the calls above; they only take a short internal lock and never wait for a
 *   poll or for describe().
 * - A source started while paused starts paused.
 * - The pause gap is written once per pause, at resume() or at stop() while paused, as
 *   [pause time, resume or stop time].
 */
class sample_collector
{
public:
    sample_collector(sink_interface& out, fork_gate& gate, detail::run_options options);
    /** stop(); catches everything. */
    ~sample_collector();

    sample_collector(const sample_collector&)            = delete;
    sample_collector& operator=(const sample_collector&) = delete;
    sample_collector(sample_collector&&)                 = delete;
    sample_collector& operator=(sample_collector&&)      = delete;

    /** Queues a source with its polling interval; start() describes and starts it.
     * @throws std::invalid_argument when source is null or interval <= 0
     * @throws std::logic_error after stop() */
    void add_source(std::unique_ptr<source_interface> source,
                    std::chrono::nanoseconds          interval);
    /** Describes and starts every source added since the last start(). describe() runs
     * inside a fork pass that waits for a fork in progress. A source that throws is
     * disabled and logged; the others are not affected. */
    void start();
    void pause();
    void resume();
    /** Terminal: each source's thread is stopped and release() called at most once,
     * even if release() throws. */
    void stop();

private:
    enum class state : std::uint8_t
    {
        created,
        running,
        failed,
        stopped
    };

    struct entry
    {
        std::unique_ptr<detail::source_runner> runner;
        state                                  status = state::created;
    };

    /** Caller holds m_mutex. Marks the collector and every entry stopped.
     * @return the entries that were running or failed (their threads and providers
     * still need stopping) */
    [[nodiscard]] std::vector<entry*> take_active();

    /** Describes and starts one source outside the lock. @return its new status */
    [[nodiscard]] state launch(entry& item);

    /** Calls func(); an exception is logged once with the source name and `step`.
     * @return false when func() threw */
    template <typename Fn>
    [[nodiscard]] bool guarded(entry& item, std::string_view step, Fn&& func);

    sink_interface&     m_sink;
    detail::run_context m_context;
    /** Guards m_entries (the vector and every status), m_paused_at and m_stopped. */
    std::mutex m_mutex;
    /** Entries are never removed, so pointers to them stay valid. */
    std::vector<std::unique_ptr<entry>> m_entries;
    std::uint64_t                       m_paused_at = 0;
    bool                                m_stopped   = false;
};
}  // namespace rocprofsys::collection
