// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocprofsys::backends::hipfile
{

/// Slots in a hipFile Level-3 snapshot. Mirrors HIPFILE_MAX_GPUS, duplicated so this
/// header stays free of <hipfile.h>; wrapper.hpp static_asserts the two agree.
inline constexpr std::size_t MAX_GPUS = 16;

/**
 * @brief Per-GPU hipFile I/O counters, cumulative over the process lifetime.
 *
 * Only fields hipFile genuinely tracks are carried. Large parts of the Level-1 and
 * Level-3 structs are documented zero-filled stubs (batch_*, *_ops_per_sec, both size
 * histograms, n_p2p_*, n_dr_*, uuid, *_utilization); surfacing one would produce a
 * counter that reads a constant zero.
 *
 * The durations (read_duration_us / write_duration_us) are deliberately absent:
 * bandwidth is normalised to wall clock so it is comparable to the AMD SMI PCIe
 * bandwidth tracks, which makes hipFile's own I/O-time accounting unused.
 */
struct gpu_stats
{
    std::uint64_t read_bytes       = 0;
    std::uint64_t write_bytes      = 0;
    std::uint64_t read_ops         = 0;  ///< Reads across all backends
    std::uint64_t write_ops        = 0;  ///< Writes across all backends
    std::uint64_t fastpath_reads   = 0;  ///< Reads that took the AIS fastpath
    std::uint64_t fastpath_writes  = 0;  ///< Writes that took the AIS fastpath
    std::uint64_t fallback_reads   = 0;  ///< Reads that fell back to POSIX
    std::uint64_t fallback_writes  = 0;  ///< Writes that fell back to POSIX
    std::uint64_t unaligned_reads  = 0;
    std::uint64_t unaligned_writes = 0;
    std::uint64_t read_errors      = 0;
    std::uint64_t write_errors     = 0;
};

/**
 * @brief One hipFile Level-3 sample, indexed by GPU ordinal.
 *
 * @c per_gpu is not packed: entry @c i holds GPU @c i and slots for inactive GPUs stay
 * zero-filled, so consumers index by ordinal rather than iterating a dense prefix.
 */
struct stats_snapshot
{
    std::array<gpu_stats, MAX_GPUS> per_gpu{};
};

}  // namespace rocprofsys::backends::hipfile
