// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/types.hpp"
#include "logger/debug.hpp"

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

// Supplied by the build (see source/lib/backends/hipfile/CMakeLists.txt). Checked
// explicitly because an undefined macro evaluates to 0 in #if, which would turn both
// version comparisons below into vacuous truths instead of failing loudly.
#if !defined(ROCPROFSYS_HIPFILE_MIN_VERSION_MAJOR) ||                                    \
    !defined(ROCPROFSYS_HIPFILE_MIN_VERSION_MINOR) ||                                    \
    !defined(ROCPROFSYS_HIPFILE_MIN_VERSION_PATCH)
#    error                                                                               \
        "ROCPROFSYS_HIPFILE_MIN_VERSION_{MAJOR,MINOR,PATCH} must be defined by the build"
#endif

namespace rocprofsys::backends::hipfile
{

/// @brief Collapse a version triple into one comparable number.
[[nodiscard]] constexpr unsigned long
version_ordinal(unsigned major, unsigned minor, unsigned patch) noexcept
{
    return (major * 1000000UL) + (minor * 1000UL) + patch;
}

/// @brief First hipFile release exposing the per-GPU stats API this backend needs.
inline constexpr unsigned long MIN_HIPFILE_VERSION = version_ordinal(
    ROCPROFSYS_HIPFILE_MIN_VERSION_MAJOR, ROCPROFSYS_HIPFILE_MIN_VERSION_MINOR,
    ROCPROFSYS_HIPFILE_MIN_VERSION_PATCH);

// find_package() already vetted the package, but it cannot vet the header that actually
// lands on the include path: a stale hipfile.h from another prefix would otherwise fail
// much later with a confusing "hipFileGetStatsL3 was not declared".
static_assert(version_ordinal(HIPFILE_VERSION_MAJOR, HIPFILE_VERSION_MINOR,
                              HIPFILE_VERSION_PATCH) >= MIN_HIPFILE_VERSION,
              "hipfile.h predates the per-GPU stats API (hipFileGetStatsL3); reconfigure "
              "against a newer hipFile or build with ROCPROFSYS_BUILD_HIPFILE=OFF");

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
     * @brief Whether the libhipfile loaded into this process is new enough to query.
     *
     * The headers are pinned at build time, but hipFile's SOVERSION is its major version,
     * which is 0 for every release so far: libhipfile.so.0 satisfies the link for 0.4 and
     * 0.5 alike. Running on a host with an older hipFile than the build machine would
     * therefore load a library with no stats API in it. Resolved once and cached, so the
     * per-sample cost is a load of a bool.
     */
    static bool runtime_version_supported() noexcept
    {
        static const bool _supported = []() {
            unsigned major = 0;
            unsigned minor = 0;
            unsigned patch = 0;

            if(hipFileGetVersion(&major, &minor, &patch).err != hipFileSuccess)
            {
                LOG_WARNING("hipFile telemetry unavailable: the hipFile runtime version "
                            "could not be queried");
                return false;
            }

            if(version_ordinal(major, minor, patch) < MIN_HIPFILE_VERSION)
            {
                LOG_WARNING(
                    "hipFile telemetry unavailable: the loaded hipFile runtime is "
                    "{}.{}.{}, but the per-GPU stats API requires {}.{}.{}",
                    major, minor, patch, ROCPROFSYS_HIPFILE_MIN_VERSION_MAJOR,
                    ROCPROFSYS_HIPFILE_MIN_VERSION_MINOR,
                    ROCPROFSYS_HIPFILE_MIN_VERSION_PATCH);
                return false;
            }

            return true;
        }();
        return _supported;
    }

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
