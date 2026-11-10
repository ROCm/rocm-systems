// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file sampler.hpp
 * @brief Online CPU stack recorder for ALL threads of the current process.
 *
 *   * One perf_event_open fd + mmap ring per tid in /proc/self/task.
 *   * One internal worker thread polls all rings, decodes
 *     PERF_RECORD_SAMPLE into POD frame chains and streams them, encoded
 *     on the wire format, to a host-supplied sink (see config::streaming).
 *   * Periodic rescan of /proc/self/task picks up new threads.
 *
 * This class is the HOT PATH. It does NO symbol resolution, NO
 * user-facing filtering, and NO user callbacks during sampling.
 * All work is fixed-cost decode + encode.
 *
 * For symbol resolution + filtering of the recorded data, use
 * cpu_snap::processing::symbols::sample_postprocessor after stop(). To keep a capture
 * in memory for same-process analysis, pair a core::memory_sink (bounded,
 * stop-when-full) with core::sample_reconstructor to decode the captured
 * bytes back into per-tid core::raw_sample vectors plus a core::module_map.
 *
 * Concurrency
 *   * The worker thread is the only caller into the sink. Readers of
 *     sink-owned state MUST call stop() first; the join in stop()
 *     synchronises-with subsequent reads.
 *   * Stats counters use std::atomic with relaxed ordering.
 *
 * Out-of-line bodies live in sampling/src/sampler.cpp. Declarations and
 * trivial accessors stay in this header. The neutral driver owns a
 * triggers::i_trigger (the capture mechanism, via unique_ptr), the one
 * unwind::unwinder every trigger drains against, and the
 * streaming_pipeline (the per-sample sink). C++20.
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

#include "cpu_snap/core/frame_limits.hpp"
#include "cpu_snap/core/raw_sample.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/backend_selection.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/logger.hpp"
#include "cpu_snap/sampling/sampler_capabilities.hpp"
#include "cpu_snap/sampling/streaming_pipeline.hpp"
#include "cpu_snap/sampling/thread_tracker.hpp"
#include "cpu_snap/sampling/triggers/i_trigger.hpp"
#include "cpu_snap/sampling/unwind/unwinder.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"

namespace cpu_snap::sampling
{

/** @brief Default sampling frequency in Hz when caller does not override. */
inline constexpr std::uint32_t k_default_sample_freq_hz = 200;

/**
 * @brief Per-thread CPU stack recorder.
 *
 * Opens one perf_event_open fd + mmap ring per existing tid in
 * /proc/self/task, drains on an internal worker thread, and
 * optionally rescans for newly spawned threads.
 *
 * @note The instance is the HOT PATH. It performs no symbol resolution,
 *       no user-facing filtering, and invokes no callbacks during
 *       sampling.
 * @note A streaming sink is mandatory (see config::streaming::sink). Use
 *       core::sample_reconstructor to decode the captured bytes after
 *       stop(), then cpu_snap::processing::symbols::sample_postprocessor for
 *       symbolization + filtering.
 */
class sampler
{
public:
    /**
     * @brief perf_event_open attribute options.
     *
     * Knobs that map directly to struct perf_event_attr fields and
     * the kernel-side sampling behavior.
     */
    struct perf_attr_options
    {
        /** Target sample rate in Hertz. Kernel auto-tunes the period. */
        std::uint32_t sample_freq_hz = k_default_sample_freq_hz;

        /**
         * Maximum number of frames stored per sample, at most
         * core::k_max_callchain_depth. In frame-pointer mode on the perf backend the
         * kernel additionally caps the chain at kernel.perf_event_max_stack.
         */
        std::uint32_t callchain_depth = core::k_default_callchain_depth;

        /**
         * Number of data pages in each per-tid mmap ring. Must be a power of
         * two. 0 (default) sizes the ring automatically: 8 pages for
         * frame_pointer, and in dwarf mode enough pages to hold at least 8
         * samples of dwarf_stack_bytes each. An explicit value must hold at
         * least one full sample record, otherwise the sampler throws.
         */
        std::uint32_t kernel_pages_per_tid = 0;

        /** When true, attr.exclude_kernel = 1 (no kernel-mode IPs captured). */
        bool exclude_kernel = true;

        /** When true, attr.exclude_hv = 1 (no hypervisor IPs captured). */
        bool exclude_hv = true;

        /**
         * Bytes of user stack to snapshot per sample in DWARF mode (perf
         * backend). Must be 8-byte aligned and at most 65528 (kernel limit).
         * Larger = more frames walkable, more mmap-ring pressure. The 32 KiB
         * default covers deep HIP/LLVM stacks; a shallower snapshot truncates
         * them and splits outer frames such as main in the timeline.
         */
        std::uint32_t dwarf_stack_bytes = 32768;

        /** Stack-walk strategy. See unwind_mode. */
        unwind_mode unwind = unwind_mode::auto_best;
    };

    /**
     * @brief Signal/timer-backend attribute options.
     *
     * Used when @ref config::backend resolves to backend_kind::signal. The backend
     * arms one per-thread wall-time (CLOCK_MONOTONIC) timer (SIGEV_THREAD_ID) delivering
     * a realtime signal; the handler snapshots regs + a bounded stack copy into a
     * preallocated per-thread ring, and the worker runs an off-handler unwind
     * (DWARF or frame-pointer, per @ref perf_attr_options::unwind) to emit the
     * full frame chain. Sampling is wall-time per thread (see signal/backend.hpp
     * for why the clock is not per-thread CPU-time).
     */
    struct signal_attr_options
    {
        /** Target per-thread sample rate in Hertz. Timer period = 1 / freq. */
        std::uint32_t sample_freq_hz = k_default_sample_freq_hz;

        /**
         * Realtime delivery-signal offset above SIGRTMIN. The backend uses
         * SIGRTMIN + signal_rt_offset (NOT SIGPROF, to avoid collision). Must
         * resolve to a signal within [SIGRTMIN, SIGRTMAX].
         */
        std::uint32_t signal_rt_offset = 4;

        /**
         * Bytes of user stack copied per snapshot inside the handler. Must be
         * 8-byte aligned. Larger = deeper walkable chains, larger ring.
         */
        std::uint32_t stack_snapshot_bytes = 8192;

        /** Snapshot slots per per-thread ring; rounded up to a power of two. */
        std::uint32_t ring_capacity = 256;

        /** Startup delay before the first tick per thread, in nanoseconds. */
        std::uint64_t start_delay_ns = 0;
    };

    /**
     * @brief Thread-attach behavior.
     *
     * Controls which tids the recorder opens events on and how it
     * discovers new threads spawned after start().
     */
    struct thread_attach_options
    {
        /** Period between /proc/self/task rescans for newly spawned threads. */
        std::chrono::milliseconds rescan_period{ 100 };

        /** When true, the recorder rescans /proc/self/task and attaches to new tids. */
        bool watch_new_threads = true;

        /** When false, the calling process's main thread is NOT attached. */
        bool include_main_thread = true;

        /**
         * When true (default), the recorder will NOT attach to its own
         * internal worker thread. Prevents the sampler from sampling its
         * own drain loop, which would pollute the trace with self-cost
         * stacks like `worker_loop` / `drain_ring`.
         */
        bool exclude_sampler_worker = true;

        /**
         * Tids for which this returns true are never attached, and are detached
         * if already tracked. Called from the sampler's worker thread (and from
         * start()); may be empty.
         */
        std::function<bool(pid_t)> is_excluded;

        /**
         * Called before every thread listing + attach pass (start() and each
         * rescan); the returned object is held until the pass ends. Lets the
         * embedder block thread registration while the sampler decides which
         * tids to attach. May be empty.
         */
        std::function<std::shared_ptr<void>()> rescan_guard;
    };

    /**
     * @brief Streaming-pipeline options.
     */
    struct streaming_options
    {
        /**
         * Mandatory streaming sink. The recorder routes every sample
         * through cpu_snap::sampling::internal::streaming_pipeline -> sink
         * (host owns persistence and any memory bound). Constructing a
         * sampler without a sink throws cpu_snap::config_error. For a
         * bounded in-memory capture (analyze after stop() in this
         * process), use core::memory_sink plus core::sample_reconstructor.
         * Compatible with every unwind_mode, including dwarf: the perf
         * backend unwinds on the worker thread before handing frames to
         * the pipeline, same as the signal backend does.
         */
        cpu_snap::core::sink_fn sink;

        /** Streaming pipeline chunk size in bytes (forwarded to streaming_pipeline). */
        std::uint32_t chunk_bytes = cpu_snap::core::k_default_chunk_bytes;

        /** Sample-count cadence for dl_module_watcher polling. */
        std::size_t maps_poll_interval = internal::k_default_pipeline_interval;

        /** Sample-count cadence for CHECKPOINT emission. */
        std::size_t checkpoint_interval = internal::k_default_pipeline_interval;

        /** Hard cap on frame_trie node count (default 2^22). Zero = uncapped. */
        std::size_t trie_max_nodes = internal::k_default_trie_max_nodes;

        /** Process name embedded in file_header. Empty = read /proc/self/comm. */
        std::string process_name;
    };

    /**
     * @brief Recorder configuration.
     *
     * Aggregates the backend selector plus four groups of knobs:
     *   - @ref backend, which capture mechanism to build (default automatic)
     *   - @ref perf, perf_event_open attributes and kernel-side sampling
     *   - @ref threads, which tids the recorder attaches to
     *   - @ref streaming, streaming-pipeline options (sink is mandatory)
     *   - @ref signal, signal/timer-backend options (used when backend == signal)
     *
     * Passed by value to sampler at construction. Designated init recommended.
     *
     * @note perf / threads / streaming keep their position and shape; existing
     *       call sites compile unchanged. backend defaults to automatic.
     */
    struct config
    {
        perf_attr_options     perf{};
        thread_attach_options threads{};
        streaming_options     streaming{};
        backend_kind          backend = backend_kind::automatic;
        signal_attr_options   signal{};

        /**
         * Optional log sink (empty = logging off, no cost). Receives
         * lifecycle and diagnostic messages; see logger.hpp for the
         * threading contract. Never called per sample or from a signal handler.
         */
        log_sink logger{};

        /**
         * Clock of every timestamp the library produces: sample timestamps
         * (perf attr.clockid / signal-handler clock_gettime), module records,
         * lost/checkpoint/eof record wall_ns, and log_record::timestamp_ns. The
         * choice is recorded in file_header.flags bits 3-4. Not affected:
         * file_header.start_unix_ns (always CLOCK_REALTIME), timer scheduling
         * clocks, and the thread rescan period. With the perf backend a
         * non-monotonic clock the kernel rejects makes thread attach throw
         * cpu_snap::environment_error instead of falling back.
         */
        clock_kind clock = clock_kind::monotonic;
    };

    /**
     * @brief Construct the recorder.
     *
     * Validates @p cfg and probes kernel/CPU capabilities. Does NOT
     * open any perf events yet.
     *
     * @param cfg Configuration (see sampler::config).
     * @param backend Optional capture trigger to use instead of the one
     *        selected by cfg.backend (null = build the configured one).
     * @throws cpu_snap::environment_error When cfg.backend is perf (explicit)
     *         and perf_event_open is not permitted (code() is the errno).
     * @throws cpu_snap::config_error When streaming.sink is unset (a sink is
     *         mandatory; see core::memory_sink for a bounded in-memory sink).
     * @throws cpu_snap::config_error When callchain_depth is zero or above
     *         core::k_max_callchain_depth, kernel_pages_per_tid is not a power
     *         of two (or, for the perf backend, too small for one sample
     *         record), or dwarf_stack_bytes is zero, misaligned or above 65528.
     * @throws cpu_snap::config_error When the requested unwind_mode is not
     *         supported on this kernel/CPU, or the selected (trigger,
     *         unwinder) pair is incompatible.
     */
    explicit sampler(config cfg, std::unique_ptr<triggers::i_trigger> backend = nullptr);

    /**
     * @brief Capabilities probed at construction time.
     * @return Const reference to the probed capability set.
     */
    [[nodiscard]] const sampler_capabilities& caps() const noexcept { return m_caps; }

    /**
     * @brief Backend actually in use after resolving backend_kind::automatic.
     * @return The chosen backend; kind is never automatic. reason is non-empty
     *         only when automatic fell back to signal.
     */
    [[nodiscard]] const backend_choice& selected_backend() const noexcept
    {
        return m_choice;
    }

    /**
     * @brief Unwind mode actually in use after auto_best resolution.
     * @return Resolved stack-walking strategy.
     */
    [[nodiscard]] unwind_mode resolved_unwind_mode() const noexcept
    {
        return m_cfg.perf.unwind;
    }

    /**
     * @brief Destructor.
     *
     * Stops the recorder if still running and removes this instance
     * from the fork-handler registry.
     */
    ~sampler();

    sampler(const sampler&)            = delete;
    sampler& operator=(const sampler&) = delete;
    sampler(sampler&&)                 = delete;
    sampler& operator=(sampler&&)      = delete;

    /**
     * @brief Tear down child-side state after fork().
     *
     * Called in the forked child through internal::fork_registry once
     * the sampler is started.
     *
     * @note Closes inherited fds, unmaps inherited rings, detaches the
     *       (gone) worker thread handle. Safe to call only from the
     *       child of a fork().
     */
    void on_child_fork() noexcept;

    /**
     * @brief Start the recorder.
     *
     * Opens one perf_event fd + mmap ring per existing thread, spawns
     * the internal worker thread, and arms the rescan loop. Idempotent
     * (re-entry is a no-op).
     *
     * @throws cpu_snap::environment_error On perf_event_open, mmap, epoll or
     *         timer failure, or when /proc/self/task cannot be enumerated.
     * @throws cpu_snap::resource_error When a backend capacity is exhausted
     *         (e.g. the signal backend's tid table is full). On either, the
     *         recorder is rolled back to its pre-start state first: every
     *         attached thread is released and nothing more is written to
     *         the sink.
     */
    void start();

    /**
     * @brief Temporarily disable sample firing on every per-tid event.
     *
     * The worker thread keeps running so there is no create/join cost,
     * but no new PERF_RECORD_SAMPLE records appear until resume().
     * Idempotent; no-op when not running.
     */
    void pause() noexcept;

    /**
     * @brief Re-enable sample firing after pause().
     *
     * Idempotent; no-op when not running.
     */
    void resume() noexcept;

    /**
     * @brief Whether the recorder is paused.
     * @return True if the recorder is currently paused.
     */
    [[nodiscard]] bool is_paused() const noexcept
    {
        return m_paused.load(std::memory_order_acquire);
    }

    /**
     * @brief Whether the recorder is between start() and stop().
     * @return True while the recorder is running.
     */
    [[nodiscard]] bool is_running() const noexcept
    {
        return m_running.load(std::memory_order_acquire);
    }

    /**
     * @brief Stop the recorder.
     *
     * Disables every fd, joins the worker thread, performs one final
     * drain, flushes the streaming pipeline, and releases all rings.
     * Idempotent.
     */
    void stop();

    // ----- stats (callable any time) ---------------------------------------

    /**
     * @brief Total samples received from the kernel (including dropped).
     * @return Cumulative sample count.
     */
    [[nodiscard]] std::size_t total_samples() const noexcept
    {
        return m_total.load(std::memory_order_relaxed);
    }

    /**
     * @brief Samples lost by kernel ring overflow (PERF_RECORD_LOST).
     * @return Cumulative lost-sample count.
     */
    [[nodiscard]] std::size_t lost_samples() const noexcept
    {
        return m_lost.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of currently-attached tids, or peak after stop().
     * @return Live tid count while running, peak after stop().
     */
    [[nodiscard]] std::size_t tracked_threads() const noexcept
    {
        const std::lock_guard<std::mutex> lk(m_state_mutex);
        const auto                        live = m_backend->tracked_tids();
        return live != 0 ? live : m_peak_tracked.load(std::memory_order_relaxed);
    }

    /**
     * @brief Internal failures swallowed by the streaming pipeline.
     *
     * Counts streaming-mode errors (sink throw, bad_alloc, frame_trie cap)
     * that the pipeline absorbs to keep the recorder alive. Distinct from
     * lost_samples(), which only counts kernel-side PERF_RECORD_LOST.
     *
     * The final value of the last run stays readable after stop().
     *
     * @return Cumulative internal-error count, or 0 when not in streaming mode.
     */
    [[nodiscard]] std::uint64_t streaming_internal_errors() const noexcept
    {
        const std::lock_guard<std::mutex> lk(m_state_mutex);
        return m_streaming ? m_streaming->internal_errors()
                           : m_final_stream_stats.internal_errors;
    }

    /**
     * @brief Samples dropped by the streaming pipeline.
     *
     * Counts samples the pipeline discarded itself (frame_trie cap, internal
     * error). Cumulative for the run and still readable after stop().
     *
     * @return Cumulative dropped-sample count, or 0 when not in streaming mode
     *         or never started.
     */
    [[nodiscard]] std::uint64_t streaming_dropped_samples() const noexcept
    {
        const std::lock_guard<std::mutex> lk(m_state_mutex);
        return m_streaming ? m_streaming->dropped_samples()
                           : m_final_stream_stats.dropped_samples;
    }

    /**
     * @brief Samples whose DWARF unwind read stack memory outside the
     *        captured snapshot window.
     *
     * Those words come from the sampled process's live memory, which may
     * differ from the memory at sample time, so a high count relative to
     * total_samples() means parts of the stacks may be stale. Raise
     * perf.dwarf_stack_bytes or signal.stack_snapshot_bytes to shrink it.
     * Cumulative for the sampler's lifetime and still readable after stop().
     * Counted over every sampled thread, so one thread with a deep stack (even
     * an idle one) raises it.
     *
     * @return Cumulative count, or 0 for unwinders that do not copy stacks
     *         (kernel callchain, frame-pointer copy).
     */
    [[nodiscard]] std::uint64_t unwinder_live_stack_samples() const noexcept
    {
        const auto* dwarf = std::get_if<unwind::dwarf_unwinder>(&m_unwinder);
        return dwarf != nullptr ? dwarf->live_stack_read_samples() : 0;
    }

    /**
     * @brief Rescan rounds or per-thread attaches that failed.
     *
     * A failed attach (fd/mmap/timer exhaustion, ...) skips that thread and is
     * retried on the next rescan; a failed /proc/self/task listing skips the
     * whole round. The worker keeps running in both cases.
     *
     * @return Cumulative failure count.
     */
    [[nodiscard]] std::uint64_t rescan_failures() const noexcept
    {
        return m_rescan_failures.load(std::memory_order_relaxed);
    }

    /**
     * @brief Whether the worker thread terminated on an unexpected exception.
     *
     * Once true, no more samples are drained until stop(); is_running()
     * still reports true until then.
     *
     * @return True if the worker died.
     */
    [[nodiscard]] bool worker_failed() const noexcept
    {
        return m_worker_failed.load(std::memory_order_acquire);
    }

private:
    /**
     * Drain the backend once into m_streaming.
     *
     * The stat counters are bumped inside streaming_pipeline (a concrete
     * sink), and the backend's per-sample loop calls it directly, so
     * per-sample dispatch stays inlinable - no std::function thunk on the
     * hot path.
     */
    void drain_once() noexcept;

    internal::streaming_pipeline::config build_streaming_config();

    void                                rescan_threads();
    [[nodiscard]] pid_t                 excluded_tid() const noexcept;
    [[nodiscard]] std::shared_ptr<void> hold_rescan_guard() const;
    void                                rollback_start() noexcept;

    // ----- worker thread (hot path; no allocations per sample) -------------
    void worker_loop() noexcept;

    config                                      m_cfg;
    sampler_capabilities                        m_caps;
    backend_choice                              m_choice;
    unwind::unwinder                            m_unwinder{};
    std::unique_ptr<triggers::i_trigger>        m_backend;
    std::optional<internal::streaming_pipeline> m_streaming;
    internal::streaming_pipeline::stats         m_final_stream_stats{};
    std::thread                                 m_worker;
    std::atomic<bool>                           m_running{ false };
    std::atomic<bool>                           m_paused{ false };
    std::atomic<bool>                           m_shutdown{ false };
    std::atomic<std::size_t>                    m_total{ 0 };
    std::atomic<std::size_t>                    m_lost{ 0 };
    std::atomic<std::size_t>                    m_peak_tracked{ 0 };
    std::atomic<pid_t>                          m_worker_tid{ -1 };
    std::atomic<std::uint64_t>                  m_rescan_failures{ 0 };
    std::atomic<bool>                           m_worker_failed{ false };
    // Guards m_backend topology, m_streaming lifetime and m_threads
    // between the worker's rescan and caller-thread pause/resume/stats/stop.
    mutable std::mutex                      m_state_mutex;
    std::optional<internal::thread_tracker> m_threads;
};

}  // namespace cpu_snap::sampling
