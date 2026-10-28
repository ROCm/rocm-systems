#pragma once

/**
 * @file sampler_capabilities.hpp
 * @brief Probe what perf_event_open features the running kernel + CPU
 *        accept. Each probe opens a throw-away perf event with the
 *        minimum config needed to exercise the feature, then closes it.
 *        Safe to call repeatedly (no global state, no extra permissions).
 */

#include <cerrno>
#include <filesystem>
#include <string>
#include <system_error>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace cpu_snap::sampling
{

/**
 * @brief What perf_event_open features are usable on this kernel + CPU.
 */
struct sampler_capabilities
{
    /** PERF_SAMPLE_CALLCHAIN with FP-walked stacks. Always available. */
    bool frame_pointer = true;
    /** attr.precise_ip = 1 (Intel PEBS, AMD IBS-Op). */
    bool precise_ip    = false;
    /** /sys/devices/ibs_op/type present (AMD IBS-Op PMU available). */
    bool ibs_op        = false;

    /** @return e.g. `{ frame_pointer=yes precise_ip=yes ibs_op=no }`. */
    [[nodiscard]] std::string to_string() const
    {
        std::string out = "{ frame_pointer=";
        out += frame_pointer ? "yes" : "no";
        out += " precise_ip=";
        out += precise_ip ? "yes" : "no";
        out += " ibs_op=";
        out += ibs_op ? "yes" : "no";
        out += " }";
        return out;
    }
};

namespace detail
{

inline int
try_open(const perf_event_attr& attr) noexcept
{
    const int fd =
        static_cast<int>(::syscall(SYS_perf_event_open, &attr, /*pid=*/0, /*cpu=*/-1,
                                   /*group_fd=*/-1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0) return errno;
    ::close(fd);
    return 0;
}

inline bool
probe_precise_ip() noexcept
{
    perf_event_attr attr{};
    attr.size           = sizeof(attr);
    attr.type           = PERF_TYPE_HARDWARE;
    attr.config         = PERF_COUNT_HW_CPU_CYCLES;
    attr.freq           = 1;
    attr.sample_freq    = 100;
    attr.sample_type    = PERF_SAMPLE_IP;
    attr.disabled       = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv     = 1;
    attr.precise_ip     = 1;  // -> PEBS / IBS-Op
    return try_open(attr) == 0;
}

inline bool
probe_ibs_op() noexcept
{
    // AMD-specific PMU exposed by the kernel when IBS is available.
    std::error_code ec;
    return std::filesystem::exists("/sys/devices/ibs_op/type", ec) && !ec;
}

}  // namespace detail

/** @brief Outcome of probe_perf_open(). */
struct perf_probe_result
{
    bool available = false;
    int  err       = 0;  ///< errno of the failed call; 0 when available.
};

/**
 * @brief Whether a software CPU-clock perf event can be opened on the calling
 *        thread right now (the same call shape the perf backend makes).
 *
 * Opens and closes one throw-away event, so it reflects the sysctl, CAP_PERFMON,
 * seccomp and LSM policy that actually apply.
 *
 * @param exclude_kernel attr.exclude_kernel of the probed event; unprivileged
 *        access needs it set when perf_event_paranoid is 2.
 */
inline perf_probe_result
probe_perf_open(bool exclude_kernel) noexcept
{
    perf_event_attr attr{};
    attr.size           = sizeof(attr);
    attr.type           = PERF_TYPE_SOFTWARE;
    attr.config         = PERF_COUNT_SW_CPU_CLOCK;
    attr.sample_type    = PERF_SAMPLE_IP;
    attr.disabled       = 1;
    attr.exclude_kernel = exclude_kernel ? 1U : 0U;
    attr.exclude_hv     = 1;
    const int err       = detail::try_open(attr);
    return perf_probe_result{ err == 0, err };
}

/**
 * @brief Probe what perf_event_open features are usable right now.
 *
 * Performs one throw-away @c perf_event_open per probe; the file
 * descriptors are closed before returning. Requires no elevated
 * privileges beyond what would be needed to open a real sampler.
 *
 * @return A sampler_capabilities with one flag per probed feature.
 */
inline sampler_capabilities
probe_sampler_capabilities() noexcept
{
    sampler_capabilities caps;
    caps.frame_pointer = true;  // PERF_SAMPLE_CALLCHAIN never refused
    caps.precise_ip    = detail::probe_precise_ip();
    caps.ibs_op        = detail::probe_ibs_op();
    return caps;
}

}  // namespace cpu_snap::sampling
