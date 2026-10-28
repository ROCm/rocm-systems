#pragma once

/**
 * @file triggers/signal/backend.hpp
 * @brief Timer + signal CPU-sampling trigger (implements i_trigger).
 *
 * A privilege-free alternative to triggers::perf::backend: instead of perf_event_open it
 * arms one per-thread POSIX timer (timer_create with SIGEV_THREAD_ID) that
 * delivers a realtime signal to the target tid. A single process-wide signal
 * handler captures, async-signal-safely, the interrupted register state plus a
 * bounded copy of the user stack into a preallocated per-thread lock-free ring,
 * then wakes the worker via an eventfd. The worker drains the rings off-handler.
 *
 * Clock: by default CLOCK_MONOTONIC (wall-time), so a sampled thread is
 * interrupted at the sample rate even while it sleeps. A blocking host syscall
 * (poll, epoll_wait, nanosleep, ...) in such a thread returns EINTR once per
 * tick, because SA_RESTART does not cover them; hosts must retry on EINTR (as
 * robust code does anyway) or select config::cpu_time, which uses the target
 * thread's CPU-time clock so only running threads are signalled. The worker
 * can create a timer on another thread's CPU clock (it is addressed by tid).
 * Sample timestamps are independent of the timer clock: the handler stamps
 * them with config::clock (default CLOCK_MONOTONIC).
 *
 * drain_into runs the off-handler unwind over the snapshot's saved GPRs
 * + stack copy + stack base and emits the full innermost-first frame chain
 * (capped at config::callchain_depth); a snapshot whose unwind yields nothing falls back to
 * a single leaf-IP frame so no sample is dropped.
 *
 * Backend selection (config.backend == signal) builds this as the sampler's
 * trigger; the perf path is unchanged and remains the default.
 *
 * Declarations only; async-signal-safety-critical bodies (the handler, the
 * timer arming, the ucontext access) live in
 * sampling/src/triggers/signal/backend.cpp, where _GNU_SOURCE is set. This
 * header is free of glibc signal/ucontext symbols so it stays includable
 * without _GNU_SOURCE.
 *
 * C++20. x86_64.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)
#include <time.h>       // NOLINT(misc-include-cleaner): timer_t

#include "cpu_snap/core/frame_limits.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/triggers/i_trigger.hpp"
#include "cpu_snap/sampling/triggers/signal/snapshot.hpp"
#include "cpu_snap/sampling/unwind/unwinder.hpp"

namespace cpu_snap::sampling::triggers::signal
{

/**
 * @brief Trigger backed by per-thread timers + a signal handler.
 *
 * Implements i_trigger. Owns one timer per attached tid, one snapshot ring
 * per attached tid, and the eventfd the handler posts to wake wait(). The
 * process-wide signal handler is installed once on first construction.
 *
 * @note Not movable/copyable: it owns the per-tid ring registry the handler
 *       reads by raw pointer, and kernel timer handles. The sampler owns it
 *       through a std::unique_ptr<i_trigger> and never moves the object.
 */
class backend : public i_trigger
{
public:
    /**
     * @brief Whether the signal trigger can supply @p kind of snapshot.
     *
     * The signal trigger has no kernel-walked-callchain concept at all (no
     * equivalent of perf's PERF_SAMPLE_CALLCHAIN); it can only ever hand an
     * unwinder a regs+stack snapshot.
     * @param kind Capture kind an unwinder would require.
     * @return True only for capture_kind::regs_and_stack.
     */
    [[nodiscard]] bool can_provide(unwind::capture_kind kind) const noexcept override
    {
        return supports(kind);
    }

    /**
     * @brief Context-free form of @ref can_provide, usable before a signal
     *        backend instance exists.
     * @param kind Capture kind to check.
     * @return True only for capture_kind::regs_and_stack.
     */
    [[nodiscard]] static constexpr bool supports(unwind::capture_kind kind) noexcept
    {
        return kind == unwind::capture_kind::regs_and_stack;
    }

    /**
     * @brief Validated configuration for one backend.
     *
     * Mirrors the relevant sampler::signal_attr_options fields (already
     * validated by the sampler) plus nothing else; the backend is otherwise
     * self-contained.
     */
    struct config
    {
        /** Per-thread sample frequency in Hz. Timer period = 1 / freq_hz. */
        std::uint32_t freq_hz;
        /**
         * Realtime delivery signal number. Must be in [SIGRTMIN, SIGRTMAX].
         * Resolved from signal_attr_options::signal_rt_offset by the sampler.
         */
        int signal_no;
        /** Bytes of user stack copied per snapshot in the handler (8-byte aligned). */
        std::uint32_t stack_snapshot_bytes;
        /** Maximum frames emitted per sample. */
        std::uint32_t callchain_depth;
        /** Snapshot slots per per-thread ring (rounded up to a power of two). */
        std::size_t ring_capacity;
        /** Startup delay before the first tick, in nanoseconds. 0 = no delay. */
        std::uint64_t start_delay_ns;
        /**
         * Tick on each thread's CPU-time clock instead of wall time. Sleeping
         * threads are then neither sampled nor interrupted with EINTR; the
         * period is CPU time consumed by the thread.
         */
        bool cpu_time = false;
        /** Clock the signal handler stamps snapshots with (sampler::config::clock). */
        clock_kind clock = clock_kind::monotonic;
    };

    /**
     * @brief Construct the signal backend.
     *
     * Installs the process-wide signal handler, opens the wake eventfd, and
     * registers this backend as the active sink the handler publishes into. The
     * previous disposition of the signal is restored by the destructor.
     *
     * @param cfg Already-validated configuration.
     * @throws cpu_snap::environment_error If the eventfd cannot be created, the signal
     *         handler cannot be installed, or the host already installed its own
     *         handler (or SIG_IGN) for config::signal_no (EBUSY); pick another
     *         signal_rt_offset in that case.
     * @throws cpu_snap::usage_error If another backend is already active
     *         (the process-wide handler routes to one backend at a time).
     */
    explicit backend(config cfg);

    ~backend() override;

    backend(const backend&)            = delete;
    backend& operator=(const backend&) = delete;
    backend(backend&&)                 = delete;
    backend& operator=(backend&&)      = delete;

    /**
     * @brief Create and arm a per-thread timer + snapshot ring for @p tid.
     *
     * Allocates the per-thread ring (off the signal path), publishes it into the
     * handler-readable registry, then creates a timer with SIGEV_THREAD_ID
     * targeting @p tid (CLOCK_MONOTONIC, or the thread's CPU-time clock when
     * config::cpu_time is set) and arms it at the configured frequency. No-op if
     * @p tid is already attached.
     *
     * @param tid Linux thread id to attach.
     * @throws cpu_snap::environment_error on timer_create / timer_settime failure
     *         other than the thread having already exited (silent no-op).
     * @throws cpu_snap::resource_error (code() == ENOSPC) when the
     *         handler-readable tid table has no room for @p tid. Nothing is
     *         armed or tracked in either case.
     */
    void attach_tid(pid_t tid) override;

    /**
     * @brief Disarm and delete the timer for @p tid and retire its ring.
     * @param tid Linux thread id to detach.
     */
    void detach_tid(pid_t tid) noexcept override;

    /**
     * @brief Block up to @p timeout_ms for the handler to post a wake.
     *
     * Blocks on the wake eventfd via poll(). The handler posts it only when a
     * thread's ring crosses a quarter of its capacity, so a ring that fills
     * more slowly is collected by the caller's timeout: poll at a period well
     * below ring_capacity / freq_hz. Returns promptly so the worker drains and
     * can shut down cleanly.
     *
     * @param timeout_ms Max time to block, ms. 0 polls, -1 blocks forever.
     * @return 1 if woken (data likely pending), 0 on timeout / EINTR, -1 on error.
     */
    [[nodiscard]] int wait(int timeout_ms) noexcept override;

    /**
     * @brief Drain every per-thread ring once into @p pipeline.
     *
     * std::visits @p unw exactly once, then for each snapshot runs the
     * off-handler unwind over the retained GPRs + stack copy and emits the
     * full innermost-first frame chain (capped at config::callchain_depth). When
     * the unwind yields zero frames (for example a truncated stack copy),
     * falls back to a single leaf-IP frame so no sample is dropped. Per-ring
     * lost (overflow) deltas are forwarded via pipeline.on_lost.
     *
     * @param pipeline  Concrete per-sample sink owned by the driver.
     * @param unw       Sampler-owned unwinder.
     */
    void drain_into(internal::streaming_pipeline& pipeline,
                    unwind::unwinder&           unw) noexcept override;

    /** @brief Disarm every per-thread timer (best-effort); rings stay allocated. */
    void disable_all() noexcept override;

    /** @brief Re-arm every per-thread timer at the configured frequency. */
    void enable_all() noexcept override;

    /**
     * @brief Disarm + delete every timer, retire every ring, drop the registry.
     *
     * Waits for handlers still inside a retired ring before freeing it. The
     * backend stays usable afterwards: a later attach_tid arms again and the
     * handler resumes waking wait().
     *
     * In a forked child the backend only holds the parent's state: close_all()
     * just drops it, and the process-wide registry is reset so a new backend can
     * be constructed there.
     */
    void close_all() noexcept override;

    /**
     * @brief Number of currently-attached tids.
     *
     * Reads an atomic count the worker republishes on every attach/detach, so it
     * is safe to call concurrently with the running worker (unlike a direct
     * m_timers.size(), which would race the worker's map mutations).
     * @return Attached-tid count.
     */
    [[nodiscard]] std::size_t tracked_tids() const noexcept override
    {
        return m_tracked_count.load(std::memory_order_relaxed);
    }

    /**
     * @brief Whether @p tid is currently attached.
     * @param tid Linux thread id to query.
     * @return True if a timer is armed for @p tid.
     * @note Reads m_timers directly; call only from the sampler worker thread
     *       (the map's sole mutator), never concurrently with it.
     */
    [[nodiscard]] bool is_tracked(pid_t tid) const noexcept override
    {
        return m_timers.contains(tid);
    }

    /**
     * @brief Snapshot the currently-attached tids into @p out (clears it first).
     * @param out Destination vector, overwritten with the live tid set.
     */
    void collect_tids(std::vector<pid_t>& out) const override;

private:
    /** Arm @p timer at the configured frequency, honoring start_delay_ns once. */
    void arm_timer(timer_t timer, bool first_arm) const noexcept;

    /** Per-attached-tid state owned by the backend (handler reads the ring*). */
    struct thread_state
    {
        timer_t                                       timer{};
        std::unique_ptr<snapshot_ring> ring;
        std::uint64_t                                 forwarded_lost = 0;
    };

    config   m_cfg;
    int      m_wake_fd    = -1;
    unsigned m_fork_epoch = 0;
    bool   m_armed   = true;
    std::unordered_map<pid_t, thread_state> m_timers;

    // Live tracked-tid count, republished by the worker on every attach/detach
    // so tracked_tids() can be read concurrently without racing m_timers.
    std::atomic<std::size_t> m_tracked_count{ 0 };

    // Reused per-drain scratch for unwind() output; avoids a per-sample
    // allocation on the drain hot path.
    std::vector<std::uint64_t> m_frame_buf;

    /** A detached thread's ring, kept until no handler can still be writing to it. */
    struct retired_ring
    {
        std::unique_ptr<snapshot_ring> ring;
        pid_t                          tid;
        std::size_t                    slot;
    };

    /** Whether a fork happened since construction; the backend then only holds the parent's state. */
    [[nodiscard]] bool stale() const noexcept;

    /** Unpublish @p tid's ring from the handler table and park it for reclaiming. */
    void retire_ring(pid_t tid, std::unique_ptr<snapshot_ring> ring) noexcept;

    /** Free every retired ring whose table slot no handler is inside any more. */
    void reclaim_retired() noexcept;

    std::vector<retired_ring> m_retired;
};

}  // namespace cpu_snap::sampling::triggers::signal
