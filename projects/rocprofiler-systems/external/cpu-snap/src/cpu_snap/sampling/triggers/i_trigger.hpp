#pragma once

/**
 * @file i_trigger.hpp
 * @brief Abstract interface for a capture-mechanism strategy (the trigger axis).
 *
 * A trigger owns the kernel-facing state of one sampling mechanism
 * (perf_event_open via triggers::perf::backend, or per-thread timers + a
 * signal handler via triggers::signal::backend) and exposes a
 * backend-neutral surface the sampler driver drives from its worker loop.
 * The driver holds one trigger through a std::unique_ptr<i_trigger>, chosen
 * at construction from config.backend.
 *
 * The per-sample path is intentionally NOT virtual: drain_into hands each
 * decoded sample to the concrete streaming_pipeline and unwinds against a
 * single concrete unwinder chosen once per call (see unwind::unwinder), so
 * per-sample dispatch stays direct and inlinable inside the trigger. Virtual
 * dispatch happens only at the coarse, per-drain-batch granularity
 * (attach/detach, wait, drain, teardown, tid bookkeeping), which is cold.
 *
 * Header-only. C++20.
 */

#include <cstddef>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

#include "cpu_snap/sampling/unwind/unwinder.hpp"

namespace cpu_snap::sampling::internal
{
class streaming_pipeline;
}  // namespace cpu_snap::sampling::internal

namespace cpu_snap::sampling::triggers
{

/**
 * @brief Strategy interface every trigger implements.
 *
 * The sampler driver owns one implementation via std::unique_ptr<i_trigger>
 * and drives it through these operations from its worker loop.
 *
 * @note attach_tid may throw (resource acquisition, e.g. perf_event_open); every
 *       other operation is noexcept, matching the driver's hot/cold split where
 *       only enrollment is allowed to fail.
 */
class i_trigger
{
public:
    i_trigger()          = default;
    virtual ~i_trigger() = default;

    i_trigger(const i_trigger&)            = delete;
    i_trigger& operator=(const i_trigger&) = delete;
    i_trigger(i_trigger&&)                 = delete;
    i_trigger& operator=(i_trigger&&)      = delete;

    /**
     * @brief Whether this trigger can hand an unwinder a @p kind snapshot.
     *
     * Part of the trigger/unwinder handshake: the sampler checks
     * `can_provide(unwinder's required_capture())` once at construction, so
     * an impossible (trigger, unwinder) pairing (e.g. the signal trigger with
     * an unwinder that needs a kernel-walked callchain) is rejected
     * immediately rather than discovered later at start().
     *
     * @param kind Capture kind an unwinder would require from this trigger.
     * @return True if this trigger can supply that kind of snapshot.
     */
    [[nodiscard]] virtual bool can_provide(unwind::capture_kind kind) const noexcept = 0;

    /**
     * @brief Open per-tid capture state and arm sampling for @p tid.
     * @param tid Linux thread id to attach.
     * @throws cpu_snap::environment_error / cpu_snap::resource_error on
     *         resource-acquisition failure (backend-specific).
     */
    virtual void attach_tid(pid_t tid) = 0;

    /**
     * @brief Tear down per-tid capture state for @p tid (if attached).
     * @param tid Linux thread id to detach.
     */
    virtual void detach_tid(pid_t tid) noexcept = 0;

    /**
     * @brief Block up to @p timeout_ms for capture data to become available.
     * @param timeout_ms Max time to block, ms. 0 polls, -1 blocks forever.
     * @return >0 if data is likely ready, 0 on timeout / EINTR, -1 on error.
     */
    [[nodiscard]] virtual int wait(int timeout_ms) noexcept = 0;

    /**
     * @brief Drain one batch, dispatching decoded samples into @p pipeline.
     *
     * Implementations std::visit @p unw exactly once per call (never inside
     * the per-sample loop), so the per-sample decode runs against a single
     * concrete unwinder alternative - concrete, inlinable, no virtual call,
     * no std::function.
     *
     * @param pipeline  Concrete per-sample sink owned by the driver.
     * @param unw       Sampler-owned unwinder; every trigger shares the same
     *                  instance.
     */
    virtual void drain_into(internal::streaming_pipeline& pipeline,
                            unwind::unwinder&           unw) noexcept = 0;

    /** @brief Disable sample firing on every attached tid (best-effort). */
    virtual void disable_all() noexcept = 0;

    /** @brief Re-enable sample firing on every attached tid (best-effort). */
    virtual void enable_all() noexcept = 0;

    /** @brief Tear down all per-tid capture state without draining. */
    virtual void close_all() noexcept = 0;

    /**
     * @brief Number of currently-attached tids.
     * @return Attached-tid count.
     */
    [[nodiscard]] virtual std::size_t tracked_tids() const noexcept = 0;

    /**
     * @brief Whether @p tid is currently attached.
     * @param tid Linux thread id to query.
     * @return True if @p tid is attached.
     */
    [[nodiscard]] virtual bool is_tracked(pid_t tid) const noexcept = 0;

    /**
     * @brief Snapshot the currently-attached tids into @p out (clears it first).
     * @param out Destination vector, overwritten with the live tid set.
     */
    virtual void collect_tids(std::vector<pid_t>& out) const = 0;
};

}  // namespace cpu_snap::sampling::triggers
