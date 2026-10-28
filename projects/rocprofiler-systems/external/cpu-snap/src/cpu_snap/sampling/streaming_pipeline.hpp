#pragma once

/**
 * @file streaming_pipeline.hpp
 * @brief Facade that composes the v1 sampling collector pipeline.
 *
 * A temporal_diff_filter feeds a frame_trie, divergent suffix is
 * written to a chunk_framer that emits framed chunks via the host
 * sink, and a dl_module_watcher is polled on a sample-count cadence.
 *
 * The contract is:
 *   * on_sample / on_lost / on_tid_exit are noexcept end-to-end; every
 *     internal failure (bad_alloc, trie cap exhaustion, sink throw) is
 *     absorbed and counted, never propagated.
 *   * NEW_NODE records are always emitted to the framer BEFORE the
 *     SAMPLE that references the new leaf_id, so a forward-replay
 *     reader sees every node before any reference to it.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <sys/types.h>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/dl_module_watcher.hpp"
#include "cpu_snap/sampling/flat_tid_map.hpp"
#include "cpu_snap/sampling/frame_trie.hpp"
#include "cpu_snap/sampling/temporal_diff_filter.hpp"

namespace cpu_snap::sampling::internal
{

/** @brief Default cadence (in samples) for both maps polling and checkpoint emission. */
// NOLINTNEXTLINE(readability-identifier-naming): codebase k_*-prefix convention
inline constexpr std::size_t k_default_pipeline_interval = 65536;

/** @brief Default frame_trie node cap (about 60 B of memory per node). */
// NOLINTNEXTLINE(readability-identifier-naming): codebase k_*-prefix convention
inline constexpr std::size_t k_default_trie_max_nodes = std::size_t{ 1 } << 22;

/**
 * @brief Single-thread streaming pipeline that turns per-sample frame
 *        arrays into the v1 wire byte-stream and feeds it to a host sink.
 *
 * Owns one frame_trie, one temporal_diff_filter, one chunk_framer, one
 * dl_module_watcher; the host only sees the sink callback + the on_* methods.
 *
 * Lifetime:
 *   1. Construct with a non-null sink in config. start_unix_ns / pid /
 *      process_name / flags are captured into the file_header.
 *   2. Construction emits the file_header + an initial MAPS_INIT.
 *   3. Drive on_sample / on_lost / on_tid_exit from the collector
 *      worker thread, one call at a time (never reentrant).
 *   4. Call finalize() to emit the closing CHECKPOINT + EOF and flush
 *      any partial chunk. Destruction calls finalize() once if it
 *      hasn't been called.
 *
 * @note Instances are NOT thread-safe. The hot path is single-threaded.
 */
class streaming_pipeline
{
public:
    /** @brief Construction parameters for the pipeline. */
    struct config
    {
        /** Output sink the framer flushes 1 MiB chunks into. Required. */
        cpu_snap::core::sink_fn sink;
        /** Soft flush threshold in bytes forwarded to chunk_framer. */
        std::uint32_t chunk_bytes = cpu_snap::core::k_default_chunk_bytes;
        /** Bitfield copied into file_header.flags (unwind bits 1-2, stream-clock bits 3-4). */
        std::uint16_t flags = 0;
        /** Process id stamped into file_header (0 = framer queries getpid()). */
        std::uint64_t pid = 0;
        /** Unix-epoch nanoseconds for the start-of-capture instant (0 = framer fills it).
         */
        std::uint64_t start_unix_ns = 0;
        /** Process label written into file_header (empty = /proc/self/comm). */
        std::string process_name;
        /** Sample-count cadence for dl_module_watcher polling. */
        std::size_t maps_poll_interval = k_default_pipeline_interval;
        /** Sample-count cadence for CHECKPOINT emission. */
        std::size_t checkpoint_interval = k_default_pipeline_interval;
        /** Hard cap on frame_trie node count (0 = uncapped). */
        std::size_t trie_max_nodes = k_default_trie_max_nodes;
        /** Replaces the dl_iterate_phdr module walk; tests only. */
        dl_module_watcher::iterator_fn module_iterator_for_tests;
        /** Optional cumulative received-sample counter (relaxed); must outlive the pipeline. */
        std::atomic<std::size_t>* total_counter = nullptr;
        /** Optional cumulative lost-sample counter (relaxed); must outlive the pipeline. */
        std::atomic<std::size_t>* lost_counter = nullptr;
        /** Clock of the lost/checkpoint/eof/module record timestamps; must match the flags clock bits. */
        clock_kind clock = clock_kind::monotonic;
    };

    /**
     * @brief Construct a pipeline that emits via @p cfg.sink.
     * @param cfg Pipeline configuration.
     * @throws cpu_snap::config_error if @p cfg.sink is null, or if any
     *         underlying component constructor throws (e.g. chunk_bytes
     *         below chunk_framer's minimum).
     */
    explicit streaming_pipeline(config cfg);

    ~streaming_pipeline();

    streaming_pipeline(const streaming_pipeline&)            = delete;
    streaming_pipeline& operator=(const streaming_pipeline&) = delete;
    streaming_pipeline(streaming_pipeline&&)                 = delete;
    streaming_pipeline& operator=(streaming_pipeline&&)      = delete;

    /**
     * @brief Process one captured sample.
     *
     * Emits any NEW_NODE records before the SAMPLE that references the
     * new leaf id. Polls dl_module_watcher and emits a CHECKPOINT on the
     * configured sample-count cadences.
     *
     * @param tid          Linux thread id of the sampled thread.
     * @param timestamp_ns Per-sample timestamp (kernel-supplied).
     * @param frames       Innermost-first frame array (matches core::raw_sample).
     */
    void on_sample(pid_t tid, std::uint64_t timestamp_ns,
                   std::span<const std::uint64_t> frames) noexcept;

    /**
     * @brief Record a PERF_RECORD_LOST count.
     *
     * Emits a single LOST record with tid 0 (aggregate / not per-tid
     * attributable).
     * @param lost_count Number of samples reported lost by the kernel.
     */
    void on_lost(std::uint64_t lost_count) noexcept;

    /**
     * @brief Drop per-tid state for an exiting tid.
     *
     * Calls temporal_diff_filter::forget_tid and removes the per-tid
     * counter row. No record is emitted; the next CHECKPOINT reflects
     * the new set.
     * @param tid Linux thread id of the exiting thread.
     */
    void on_tid_exit(pid_t tid) noexcept;

    /**
     * @brief Final drain.
     *
     * Emits a closing CHECKPOINT, polls dl_module_watcher one last time,
     * emits an EOF record, then flushes the framer. Safe to call
     * multiple times; subsequent calls are no-ops. Called from the
     * destructor if not called explicitly.
     */
    void finalize() noexcept;

    /**
     * @brief Mark the pipeline finished without emitting anything.
     *
     * For a pipeline whose sink must no longer be written to (e.g. the
     * forked child of a process that owns the sink). Later finalize() and
     * destruction are no-ops and any partially buffered chunk is dropped.
     */
    void abandon() noexcept;

    /** @brief Cumulative number of SAMPLE records emitted. */
    [[nodiscard]] std::uint64_t samples_emitted() const noexcept;

    /** @brief Cumulative number of NEW_NODE records emitted. */
    [[nodiscard]] std::uint64_t new_nodes_emitted() const noexcept;

    /** @brief Aggregate kernel-reported lost-sample count seen so far. */
    [[nodiscard]] std::uint64_t lost_total() const noexcept;

    /** @brief Number of CHECKPOINT records emitted. */
    [[nodiscard]] std::uint64_t checkpoints_emitted() const noexcept;

    /** @brief Internal failures absorbed by the noexcept entry points. */
    [[nodiscard]] std::uint64_t internal_errors() const noexcept;

    /** @brief Samples dropped by the pipeline itself (trie cap, internal error). */
    [[nodiscard]] std::uint64_t dropped_samples() const noexcept;

    /** @brief Failure counters that outlive the pipeline. */
    struct stats
    {
        std::uint64_t internal_errors = 0;
        std::uint64_t dropped_samples = 0;
    };

    /** @brief Current internal_errors() and dropped_samples() in one value. */
    [[nodiscard]] stats snapshot() const noexcept;

private:
    static constexpr std::uint64_t k_max_resync_backoff = 1U << 16;

    struct per_tid_counters
    {
        std::uint64_t samples = 0;
        std::uint64_t lost    = 0;
    };

    void record_sample(pid_t tid, std::uint64_t timestamp_ns,
                       std::span<const std::uint64_t> frames) noexcept;
    void emit_pending_nodes();
    void record_dropped_sample(pid_t tid) noexcept;
    void note_sink_errors() noexcept;
    void resync_if_needed() noexcept;
    void emit_checkpoint() noexcept;
    [[nodiscard]] std::uint64_t wall_ns_now() const noexcept;

    config                                             m_cfg;
    frame_trie                                         m_trie;
    temporal_diff_filter                               m_diff;
    cpu_snap::core::chunk_framer                       m_framer;
    cpu_snap::sampling::dl_module_watcher              m_maps;
    cpu_snap::sampling::flat_tid_map<per_tid_counters> m_tid_counters;

    // written by the drain thread, readable from any thread
    std::atomic<std::uint64_t> m_samples_emitted{0};
    std::atomic<std::uint64_t> m_new_nodes_emitted{0};
    std::atomic<std::uint64_t> m_lost_total{0};
    std::atomic<std::uint64_t> m_checkpoints_emitted{0};
    std::atomic<std::uint64_t> m_internal_errors{0};
    std::atomic<std::uint64_t> m_dropped{0};
    std::uint64_t       m_sample_calls        = 0;
    std::uint64_t       m_seen_sink_errors    = 0;
    std::uint64_t       m_resync_backoff      = 1;
    std::uint64_t       m_resync_retry_at     = 0;
    frame_trie::node_id m_next_node_id        = 1;
    bool                m_resync_pending      = false;
    bool                m_finalized           = false;
};

}  // namespace cpu_snap::sampling::internal
