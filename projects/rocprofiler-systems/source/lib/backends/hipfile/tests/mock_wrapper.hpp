// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/backend.hpp"
#include "backends/hipfile/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocprofsys::backends::hipfile::testing
{

// ── Mock raw types ──────────────────────────────────────────────────────────
// Field names mirror hipFilePerGpuStats_t / hipFileStatsLevel3_t exactly so that
// backend<>::query() compiles against them unchanged. snake_case is intentional.
// Nothing here includes <hipfile.h>, which is the point: these tests build and run
// on machines with no hipFile package installed.
// NOLINTBEGIN(readability-identifier-naming) -- field names must match
// hipFilePerGpuStats_t / hipFileStatsLevel3_t

struct mock_per_gpu_stats_t
{
    std::uint64_t read_bytes         = 0;
    std::uint64_t read_duration_us   = 0;
    std::uint64_t n_total_reads      = 0;
    std::uint64_t n_nvfs_reads       = 0;
    std::uint64_t n_posix_reads      = 0;
    std::uint64_t n_unaligned_reads  = 0;
    std::uint64_t n_reads_err        = 0;
    std::uint64_t write_bytes        = 0;
    std::uint64_t write_duration_us  = 0;
    std::uint64_t n_total_writes     = 0;
    std::uint64_t n_nvfs_writes      = 0;
    std::uint64_t n_posix_writes     = 0;
    std::uint64_t n_unaligned_writes = 0;
    std::uint64_t n_writes_err       = 0;
    std::uint64_t n_mmap             = 0;
};

struct mock_stats_l3_t
{
    std::uint32_t                              num_gpus = 0;
    std::array<mock_per_gpu_stats_t, MAX_GPUS> per_gpu_stats{};
};

// NOLINTEND(readability-identifier-naming)

/**
 * @brief Test double for the hipFile wrapper policy.
 *
 * The wrapper policy is a set of static functions, so the fake is driven through
 * process-global state rather than gmock: tests set @c next_stats and @c query_succeeds,
 * then read @c call_count to assert how often hipFile was queried. That count is what
 * proves the backend's per-timestamp memoization actually holds.
 */
struct mock_wrapper
{
    using stats_l3_t = mock_stats_l3_t;

    static constexpr std::size_t MAX_GPU_SLOTS = MAX_GPUS;

    inline static stats_l3_t  next_stats{};
    inline static bool        query_succeeds    = true;
    inline static bool        version_supported = true;
    inline static std::size_t call_count        = 0;

    static void reset()
    {
        next_stats        = stats_l3_t{};
        query_succeeds    = true;
        version_supported = true;
        call_count        = 0;
    }

    /// Stands in for the cached hipFileGetVersion() check in the real wrapper.
    static bool runtime_version_supported() noexcept { return version_supported; }

    static bool get_stats_l3(stats_l3_t* out) noexcept
    {
        ++call_count;
        if(!query_succeeds)
        {
            return false;
        }
        *out = next_stats;
        return true;
    }
};

using mock_backend = backend<mock_wrapper>;

}  // namespace rocprofsys::backends::hipfile::testing
