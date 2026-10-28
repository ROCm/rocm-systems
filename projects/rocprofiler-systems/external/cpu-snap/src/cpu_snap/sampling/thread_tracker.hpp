#pragma once

/**
 * @file thread_tracker.hpp
 * @brief Tracks which threads of the process are attached to a trigger.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t

#include "cpu_snap/sampling/logger.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/sampling/triggers/i_trigger.hpp"

namespace cpu_snap::sampling::internal
{

/** @brief One live thread as reported by a thread source. */
struct thread_entry
{
    pid_t tid = -1;

    /** Start time in clock ticks since boot; empty when it could not be read. */
    std::optional<std::uint64_t> start_ticks;
};

/** @brief Returns the current threads, or std::nullopt when they cannot be listed. */
using thread_source = std::function<std::optional<std::vector<thread_entry>>()>;

/** @brief Thread source reading /proc/self/task. */
[[nodiscard]] thread_source
proc_thread_source();

/** @brief Outcome of one rescan(): attaches that threw. */
struct rescan_result
{
    std::uint64_t failed_attaches = 0;
};

/**
 * @brief Owns the per-thread attach bookkeeping of the sampler: start ticks of
 *        attached threads (to spot recycled tids), attach failures already
 *        logged and whether the thread listing failure was already logged.
 *
 * Not thread-safe and has no mutex of its own. The owner calls attach_initial(),
 * rescan() and reset() under the same lock that guards the trigger topology;
 * poll() touches only tracker state and may run outside that lock, from the one
 * thread that calls rescan().
 */
class thread_tracker
{
public:
    /**
     * @param source            Where the current threads come from.
     * @param logger            Receives the warnings of poll() and rescan().
     * @param clock             Clock stamping those records.
     * @param include_main_thread When false attach_initial() skips the thread whose
     *                          tid equals the process id.
     */
    thread_tracker(thread_source source, log_sink logger, clock_kind clock,
                   bool include_main_thread);

    /**
     * Attach every listed thread except @p exclude_tid.
     *
     * @throws environment_error when the threads cannot be listed; whatever the
     *         trigger's attach_tid throws.
     */
    void attach_initial(triggers::i_trigger& trigger, pid_t exclude_tid);

    /**
     * List the current threads. A failed listing is logged once until a
     * listing succeeds again.
     *
     * @return The threads, or std::nullopt when listing failed.
     */
    [[nodiscard]] std::optional<std::vector<thread_entry>> poll();

    /**
     * Reconcile @p current with the trigger: detach (and tell @p pipeline about)
     * threads that exited or whose tid was recycled, attach new ones. An attach
     * that throws is counted and logged once per tid.
     *
     * @param pipeline  May be null.
     * @param exclude_tid Never attached (the sampler worker).
     */
    rescan_result rescan(triggers::i_trigger& trigger, streaming_pipeline* pipeline,
                         pid_t exclude_tid, const std::vector<thread_entry>& current);

    /** @brief Forget all bookkeeping. */
    void reset() noexcept;

private:
    void try_attach(triggers::i_trigger& trigger, const thread_entry& entry, pid_t exclude_tid);
    void log_attach_failure(pid_t tid, std::string_view what);

    thread_source                            m_source;
    log_sink                                 m_logger;
    clock_kind                               m_clock;
    bool                                     m_include_main_thread;
    std::unordered_map<pid_t, std::uint64_t> m_tid_start_ticks;
    std::unordered_set<pid_t>                m_logged_attach_failures;
    bool                                     m_listing_failed{ false };
};

}  // namespace cpu_snap::sampling::internal
