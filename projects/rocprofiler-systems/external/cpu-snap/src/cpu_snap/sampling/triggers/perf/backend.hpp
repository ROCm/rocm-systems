#pragma once

/**
 * @file triggers/perf/backend.hpp
 * @brief perf_event_open trigger: thin i_trigger adapter over kernel_drain.
 *
 * drain_into threads the concrete pipeline and unwinder through
 * kernel_drain::drain_all, so per-sample dispatch stays direct and inlinable.
 */

#include <cstddef>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

#include "cpu_snap/sampling/triggers/i_trigger.hpp"
#include "cpu_snap/sampling/triggers/perf/kernel_drain.hpp"

namespace cpu_snap::sampling::triggers::perf
{

/** @brief Trigger backed by perf_event_open + mmap rings; owns the kernel_drain. */
class backend : public i_trigger
{
public:
    /**
     * @brief Construct the perf backend.
     * @param cfg Already-validated kernel_drain configuration.
     * @throws cpu_snap::environment_error If the underlying epoll fd cannot be created.
     */
    explicit backend(kernel_drain::config cfg);

    /** Always true: either capture kind is selected via kernel_drain::config::unwind. */
    [[nodiscard]] bool can_provide(unwind::capture_kind kind) const noexcept override
    {
        return supports(kind);
    }

    /** Context-free form of can_provide, usable before an instance exists. */
    [[nodiscard]] static constexpr bool supports(unwind::capture_kind /*kind*/) noexcept
    {
        return true;
    }

    /** An exited thread (ESRCH) is a silent no-op. */
    void attach_tid(pid_t tid) override;

    void detach_tid(pid_t tid) noexcept override;

    [[nodiscard]] int wait(int timeout_ms) noexcept override;

    /** Visits the unwinder variant once, then runs the per-sample loop. */
    void drain_into(internal::streaming_pipeline& pipeline,
                    unwind::unwinder&           unw) noexcept override;

    void disable_all() noexcept override;

    void enable_all() noexcept override;

    void close_all() noexcept override;

    [[nodiscard]] std::size_t tracked_tids() const noexcept override;

    [[nodiscard]] bool is_tracked(pid_t tid) const noexcept override;

    void collect_tids(std::vector<pid_t>& out) const override;

private:
    kernel_drain m_drain;
};

}  // namespace cpu_snap::sampling::triggers::perf
