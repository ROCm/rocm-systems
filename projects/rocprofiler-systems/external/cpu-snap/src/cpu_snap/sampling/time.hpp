#pragma once

/**
 * @file time.hpp
 * @brief Wall-clock time helpers shared across the cpu_snap pipeline.
 */

#include <chrono>
#include <cstdint>

#include <time.h>

#include "cpu_snap/sampling/clock_kind.hpp"

namespace cpu_snap::sampling::utils
{

/**
 * Wall-clock nanoseconds since the Unix epoch (system_clock).
 *
 * @return Current value of std::chrono::system_clock cast to ns since epoch.
 */
[[nodiscard]] inline std::uint64_t
wall_unix_ns_now() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

/** @return The POSIX clock id that implements @p kind. */
[[nodiscard]] constexpr clockid_t
clock_id_of(clock_kind kind) noexcept
{
    switch(kind)
    {
        case clock_kind::boottime: return CLOCK_BOOTTIME;
        case clock_kind::wallclock: return CLOCK_REALTIME;
        case clock_kind::monotonic: break;
    }
    return CLOCK_MONOTONIC;
}

/** @return Current reading of the clock selected by @p kind, in nanoseconds. Async-signal-safe. */
[[nodiscard]] inline std::uint64_t
clock_now_ns(clock_kind kind) noexcept
{
    timespec ts{};
    ::clock_gettime(clock_id_of(kind), &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

}  // namespace cpu_snap::sampling::utils
