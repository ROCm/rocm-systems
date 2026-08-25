// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/types.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofsys::pmc::collectors::hipfile
{

// Data types are owned by the backend layer (the producer); re-exported here so pmc
// consumers keep their pmc::collectors::hipfile::* spellings.
namespace backend = ::rocprofsys::backends::hipfile;

using backend::gpu_stats;
using backend::MAX_GPUS;
using backend::stats_snapshot;

/**
 * @brief Bitfield for selecting which hipFile metrics to collect.
 *
 * Bit positions match the order of @c METRIC_TABLE below; @c metric_desc::bit is the
 * single source of truth tying a metric to its bit.
 */
union enabled_metrics
{
    struct
    {
        std::uint32_t read_bytes       : 1;
        std::uint32_t write_bytes      : 1;
        std::uint32_t read_ops         : 1;
        std::uint32_t write_ops        : 1;
        std::uint32_t fastpath_reads   : 1;
        std::uint32_t fastpath_writes  : 1;
        std::uint32_t fallback_reads   : 1;
        std::uint32_t fallback_writes  : 1;
        std::uint32_t unaligned_reads  : 1;
        std::uint32_t unaligned_writes : 1;
        std::uint32_t read_errors      : 1;
        std::uint32_t write_errors     : 1;
        std::uint32_t read_bandwidth   : 1;
        std::uint32_t write_bandwidth  : 1;
    } bits;
    std::uint32_t value = 0;
};

inline constexpr std::uint32_t HIPFILE_METRICS_COUNT = 14;
inline constexpr std::uint32_t ALL_HIPFILE_METRICS   = (1U << HIPFILE_METRICS_COUNT) - 1U;

/**
 * @brief One per-GPU hipFile sample.
 *
 * The twelve counters are raw cumulative totals, reported exactly as hipFile maintains
 * them. That matches every other byte counter in the profiler - AMD SMI's PCIe
 * bandwidth accumulator, the XGMI accumulators, and the NIC byte counters all publish
 * cumulative values under an ABSOLUTE value type - so a consumer reading `bytes` +
 * ABSOLUTE gets the same thing here as it does there. Perfetto's delta view recovers
 * the per-window signal for anyone who wants it.
 *
 * The two bandwidths are wall-clock rates over the sampling interval, matching AMD
 * SMI's instantaneous PCIe bandwidth. They are deliberately not derived from hipFile's
 * own read_bw_bytes_per_sec (lifetime-averaged, so it cannot show a burst) nor
 * normalised by read_duration_us (time inside I/O calls, which would put a number on
 * the timeline that does not correspond to the time axis it is drawn against).
 */
struct metrics
{
    std::uint64_t read_bytes       = 0;
    std::uint64_t write_bytes      = 0;
    std::uint64_t read_ops         = 0;
    std::uint64_t write_ops        = 0;
    std::uint64_t fastpath_reads   = 0;
    std::uint64_t fastpath_writes  = 0;
    std::uint64_t fallback_reads   = 0;
    std::uint64_t fallback_writes  = 0;
    std::uint64_t unaligned_reads  = 0;
    std::uint64_t unaligned_writes = 0;
    std::uint64_t read_errors      = 0;
    std::uint64_t write_errors     = 0;

    double read_bandwidth  = 0.0;  ///< bytes/sec over the sampling interval
    double write_bandwidth = 0.0;  ///< bytes/sec over the sampling interval

    /// Set when hipFile could not be queried, so the sample carries no measurement.
    /// Default false, which is what a value-initialized `metrics{}` needs: the paused
    /// collector emits exactly that to drop the counter tracks to zero.
    bool query_failed = false;
};

/**
 * @brief Describes one emitted metric: its track suffix, unit, bit, and accessor.
 *
 * Single source of truth for the metric set. Sampling, pause, metadata registration,
 * and the settings parser all iterate this table, so adding a metric is a one-line
 * change rather than an edit in four places.
 */
struct metric_desc
{
    const char*   suffix;  ///< Track name suffix, e.g. "Read Bytes"
    const char*   unit;    ///< Matches the AMD SMI conventions: bytes, bytes/s, count
    const char*   key;     ///< Group token for ROCPROFSYS_HIPFILE_METRICS
    std::uint32_t bit;     ///< Position in enabled_metrics
    double (*value)(const metrics&);  ///< Extractor, uniform over mixed field types
};

// Units follow the established collectors: `bytes` as AMD SMI's PCIe bandwidth
// accumulator and the NIC byte counters use, `bytes/s` as AMD SMI's instantaneous PCIe
// bandwidth uses, `count` as the CPU collector's context switches and page faults use.
inline constexpr std::array<metric_desc, HIPFILE_METRICS_COUNT> METRIC_TABLE{
    { { "Read Bytes", "bytes", "bytes", 0,
        [](const metrics& m) { return static_cast<double>(m.read_bytes); } },
      { "Write Bytes", "bytes", "bytes", 1,
        [](const metrics& m) { return static_cast<double>(m.write_bytes); } },
      { "Read Ops", "count", "ops", 2,
        [](const metrics& m) { return static_cast<double>(m.read_ops); } },
      { "Write Ops", "count", "ops", 3,
        [](const metrics& m) { return static_cast<double>(m.write_ops); } },
      { "Fastpath Reads", "count", "fastpath", 4,
        [](const metrics& m) { return static_cast<double>(m.fastpath_reads); } },
      { "Fastpath Writes", "count", "fastpath", 5,
        [](const metrics& m) { return static_cast<double>(m.fastpath_writes); } },
      { "Fallback Reads", "count", "fallback", 6,
        [](const metrics& m) { return static_cast<double>(m.fallback_reads); } },
      { "Fallback Writes", "count", "fallback", 7,
        [](const metrics& m) { return static_cast<double>(m.fallback_writes); } },
      { "Unaligned Reads", "count", "unaligned", 8,
        [](const metrics& m) { return static_cast<double>(m.unaligned_reads); } },
      { "Unaligned Writes", "count", "unaligned", 9,
        [](const metrics& m) { return static_cast<double>(m.unaligned_writes); } },
      { "Read Errors", "count", "errors", 10,
        [](const metrics& m) { return static_cast<double>(m.read_errors); } },
      { "Write Errors", "count", "errors", 11,
        [](const metrics& m) { return static_cast<double>(m.write_errors); } },
      { "Read Bandwidth", "bytes/s", "bandwidth", 12,
        [](const metrics& m) { return m.read_bandwidth; } },
      { "Write Bandwidth", "bytes/s", "bandwidth", 13,
        [](const metrics& m) { return m.write_bandwidth; } } }
};

/**
 * @brief Bits of every metric in @p group, or 0 when the group is unknown.
 *
 * A key names a read/write pair rather than a single track, mirroring how
 * ROCPROFSYS_AMD_SMI_METRICS groups its tokens (`power` covers current and average,
 * `temp` covers hotspot and edge), so users select "fastpath" rather than spelling out
 * both directions.
 */
[[nodiscard]] constexpr std::uint32_t
metric_group_mask(std::string_view group) noexcept
{
    std::uint32_t mask = 0;
    for(const auto& metric : METRIC_TABLE)
    {
        if(group == metric.key) mask |= (1U << metric.bit);
    }
    return mask;
}

/// @brief Perfetto/RocPD track name for a metric on a given GPU.
[[nodiscard]] inline std::string
track_name(std::size_t gpu_id, const char* suffix)
{
    return "GPU [" + std::to_string(gpu_id) + "] Storage " + suffix + " (S)";
}

/**
 * @brief RocPD PMC identifier for a metric, e.g. "device_storage_read_bytes".
 */
[[nodiscard]] inline std::string
pmc_name(const char* suffix)
{
    std::string out = "device_storage_";
    for(const char* c = suffix; *c != '\0'; ++c)
    {
        out += (*c == ' ')
                   ? '_'
                   : static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
    }
    return out;
}

}  // namespace rocprofsys::pmc::collectors::hipfile
