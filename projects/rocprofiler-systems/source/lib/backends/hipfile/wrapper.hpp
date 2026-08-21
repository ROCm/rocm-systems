// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/types.hpp"

// The only file in the profiler that includes hipFile's public header. Everything
// above this layer works against backends::hipfile::stats_snapshot, so the collector
// and its tests build on machines with no hipFile package installed.
//
// The backend links libhipfile directly via the hip::hipfile imported target, matching
// how the profiler consumes amd_smi and other ROCm libraries. hipFile's stats query
// reads the calling process' own shared stats region, so this must run inside the
// profiled process; the dynamic linker guarantees a single libhipfile instance per
// process, so the profiler and the target application observe the same stats.
#include <hipfile.h>

#include <cstddef>

namespace rocprofsys::backends::hipfile
{

/**
 * @brief 1:1 thin wrapper around hipFile's public stats API.
 *
 * One method, one hipFile call, no error interpretation. This is a pure
 * dependency-injection seam so @c backend<Wrapper> can be exercised against a fake
 * without linking or including hipFile.
 */
struct wrapper
{
    using stats_l3_t = hipFileStatsLevel3_t;

    static constexpr std::size_t MAX_GPU_SLOTS = HIPFILE_MAX_GPUS;

    /**
     * @brief Snapshot the Level-3 (per-GPU) stats for the calling process.
     *
     * @param out [out] zero-initialized first, populated on success.
     * @return true on success. False means the target never initialized hipFile stats -
     *         it performed no hipFile I/O, or HIPFILE_STATS_LEVEL is disabled.
     */
    static bool get_stats_l3(stats_l3_t* out) noexcept
    {
        *out = stats_l3_t{};
        return hipFileGetStatsL3(out).err == hipFileSuccess;
    }
};

static_assert(wrapper::MAX_GPU_SLOTS == MAX_GPUS,
              "backends::hipfile::MAX_GPUS is out of sync with HIPFILE_MAX_GPUS; the "
              "snapshot would silently drop or over-read per-GPU slots");

}  // namespace rocprofsys::backends::hipfile
