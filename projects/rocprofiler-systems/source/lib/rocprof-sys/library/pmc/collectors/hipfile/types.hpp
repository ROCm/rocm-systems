// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/types.hpp"
#include "common/string_utility.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
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
using backend::k_max_gpus;
using backend::stats_snapshot;

/**
 * @brief Bitfield for selecting which hipFile metrics to collect.
 *
 * Bit positions match the order of @c k_metric_table below; @c metric_desc::bit is the
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
 * The two bandwidths are rates over the sampling interval, matching AMD
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
 * Catalog for names, units, settings-group keys, bits, and the extractors used when a
 * sample is expanded into Perfetto/RocPD tracks. Sampling, pause, metadata registration,
 * and the settings parser iterate this table so those paths do not each hard-code the
 * list.
 *
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
inline constexpr std::array k_metric_table{
    metric_desc{ .suffix = "Read Bytes",
                 .unit   = "bytes",
                 .key    = "bytes",
                 .bit    = 0,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.read_bytes);
                     } },
    metric_desc{ .suffix = "Write Bytes",
                 .unit   = "bytes",
                 .key    = "bytes",
                 .bit    = 1,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.write_bytes);
                     } },
    metric_desc{ .suffix = "Read Ops",
                 .unit   = "count",
                 .key    = "ops",
                 .bit    = 2,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.read_ops);
                     } },
    metric_desc{ .suffix = "Write Ops",
                 .unit   = "count",
                 .key    = "ops",
                 .bit    = 3,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.write_ops);
                     } },
    metric_desc{ .suffix = "Fastpath Reads",
                 .unit   = "count",
                 .key    = "fastpath",
                 .bit    = 4,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.fastpath_reads);
                     } },
    metric_desc{ .suffix = "Fastpath Writes",
                 .unit   = "count",
                 .key    = "fastpath",
                 .bit    = 5,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.fastpath_writes);
                     } },
    metric_desc{ .suffix = "Fallback Reads",
                 .unit   = "count",
                 .key    = "fallback",
                 .bit    = 6,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.fallback_reads);
                     } },
    metric_desc{ .suffix = "Fallback Writes",
                 .unit   = "count",
                 .key    = "fallback",
                 .bit    = 7,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.fallback_writes);
                     } },
    metric_desc{ .suffix = "Unaligned Reads",
                 .unit   = "count",
                 .key    = "unaligned",
                 .bit    = 8,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.unaligned_reads);
                     } },
    metric_desc{ .suffix = "Unaligned Writes",
                 .unit   = "count",
                 .key    = "unaligned",
                 .bit    = 9,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.unaligned_writes);
                     } },
    metric_desc{ .suffix = "Read Errors",
                 .unit   = "count",
                 .key    = "errors",
                 .bit    = 10,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.read_errors);
                     } },
    metric_desc{ .suffix = "Write Errors",
                 .unit   = "count",
                 .key    = "errors",
                 .bit    = 11,
                 .value =
                     [](const metrics& sample_metrics) {
                         return static_cast<double>(sample_metrics.write_errors);
                     } },
    metric_desc{
        .suffix = "Read Bandwidth",
        .unit   = "bytes/s",
        .key    = "bandwidth",
        .bit    = 12,
        .value =
            [](const metrics& sample_metrics) { return sample_metrics.read_bandwidth; } },
    metric_desc{
        .suffix = "Write Bandwidth",
        .unit   = "bytes/s",
        .key    = "bandwidth",
        .bit    = 13,
        .value =
            [](const metrics& sample_metrics) { return sample_metrics.write_bandwidth; } }
};

/// @brief Mask with one bit set per entry in @c k_metric_table.
inline constexpr std::uint32_t k_all_hipfile_metrics = (1U << k_metric_table.size()) - 1U;

static_assert([]() constexpr {
    return std::ranges::all_of(k_metric_table, [](const auto& metric) constexpr {
        return metric.unit != nullptr && metric.unit[0] != '\0';
    });
}());

static_assert(k_metric_table.size() < 32,
              "enabled_metrics addresses k_metric_table through a 32-bit mask");

// k_all_hipfile_metrics assumes the bits run 0..size()-1 with no gaps, so a metric's bit
// is its index. Which metric sits at which index is arbitrary; only the correspondence
// is load-bearing.
static_assert([]() constexpr {
    for(std::size_t index = 0; index < k_metric_table.size(); ++index)
    {
        if(k_metric_table[index].bit != index) return false;
    }
    return true;
}());

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
    for(const auto& metric : k_metric_table)
    {
        if(group == metric.key)
        {
            mask |= (1U << metric.bit);
        }
    }
    return mask;
}

/**
 * @brief Bit mask of the metric whose track suffix is @p suffix, or 0 when unknown.
 *
 * Complements @c metric_group_mask: a group token selects a read/write pair, this
 * selects a single track by the suffix that appears in the Perfetto name.
 */
[[nodiscard]] constexpr std::uint32_t
metric_bit_mask(std::string_view suffix) noexcept
{
    for(const auto& metric : k_metric_table)
    {
        if(suffix == metric.suffix)
        {
            return 1U << metric.bit;
        }
    }
    return 0U;
}

static_assert(metric_bit_mask("Read Bytes") != 0U);
static_assert(metric_bit_mask("nonsense") == 0U);

/// @brief Perfetto/RocPD track name for a metric on a given GPU.
[[nodiscard]] inline std::string
track_name(std::size_t gpu_id, const char* suffix)
{
    return fmt::format("GPU [{}] Storage {} (S)", gpu_id, suffix);
}

/**
 * @brief RocPD PMC identifier for a metric, e.g. "device_storage_read_bytes".
 */
[[nodiscard]] inline std::string
pmc_name(const char* suffix)
{
    auto normalized = utility::string::to_lower(suffix);
    std::ranges::replace(normalized, ' ', '_');
    return fmt::format("device_storage_{}", normalized);
}

}  // namespace rocprofsys::pmc::collectors::hipfile
