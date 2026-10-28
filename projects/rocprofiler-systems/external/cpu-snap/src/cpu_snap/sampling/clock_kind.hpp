#pragma once

/**
 * @file clock_kind.hpp
 * @brief Selector of the single clock every library timestamp is read from.
 */

#include <cstdint>

namespace cpu_snap::sampling
{

/**
 * @brief Clock of every timestamp the library produces (samples, module
 *        records, lost/checkpoint/eof records, log records).
 */
enum class clock_kind : std::uint8_t
{
    /** CLOCK_MONOTONIC: not adjusted by wall-time steps, stops during system suspend. */
    monotonic,
    /** CLOCK_BOOTTIME: like monotonic but keeps counting during system suspend. */
    boottime,
    /** CLOCK_REALTIME: Unix-epoch nanoseconds; can jump when the system time is set. */
    wallclock,
};

}  // namespace cpu_snap::sampling
