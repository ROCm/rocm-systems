// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocprofsys::pmc::collectors::cpu
{

/**
 * @brief Bitfield union for selecting which CPU metrics to collect.
 *
 * Required by base::collector (Traits::enabled_metrics_t) and stored in the
 * serialized cpu_pmc_sample. Configurable via ROCPROFSYS_CPU_METRICS env var.
 *
 * Bit positions (for value access):
 *   - frequency     = 0   (per-CPU, from sysfs scaling_cur_freq, MHz)
 *   - load          = 1   (per-CPU, from /proc/stat, %)
 *   - page_rss      = 2   (process-level, physical memory RSS, bytes)
 *   - virt_mem      = 3   (process-level, virtual memory, bytes)
 *   - peak_rss      = 4   (process-level, peak memory HWM, bytes)
 *   - ctx_switches  = 5   (process-level, context switches, count)
 *   - page_faults   = 6   (process-level, page faults, count)
 *   - user_time     = 7   (process-level, user mode time, microseconds)
 *   - kernel_time   = 8   (process-level, kernel mode time, microseconds)
 */
union enabled_metrics
{
    struct
    {
        std::uint32_t frequency    : 1;
        std::uint32_t load         : 1;
        std::uint32_t page_rss     : 1;
        std::uint32_t virt_mem     : 1;
        std::uint32_t peak_rss     : 1;
        std::uint32_t ctx_switches : 1;
        std::uint32_t page_faults  : 1;
        std::uint32_t user_time    : 1;
        std::uint32_t kernel_time  : 1;
    } bits;
    std::uint32_t value = 0;
};

/// All 9 CPU metrics enabled (bits 0-8).
inline constexpr std::uint32_t ALL_CPU_METRICS = 0x1FF;

/**
 * @brief Per-CPU metric snapshot for a single CPU core.
 */
struct per_cpu_metrics
{
    size_t cpu_id    = 0;
    float  frequency = 0.0f;  // MHz, from /proc/cpuinfo
    double load      = 0.0;   // %, computed from /proc/stat deltas
};

/**
 * @brief Process-level resource usage metrics.
 *
 * Per-process, not per-CPU. Collected via getrusage() and /proc/self/statm.
 */
struct process_metrics
{
    std::int64_t page_rss         = 0;  // bytes
    std::int64_t virt_mem         = 0;  // bytes
    std::int64_t peak_rss         = 0;  // bytes
    std::int64_t context_switches = 0;  // count (voluntary + involuntary)
    std::int64_t page_faults      = 0;  // count (major + minor)
    std::int64_t user_mode_time   = 0;  // microseconds
    std::int64_t kernel_mode_time = 0;  // microseconds
};

/**
 * @brief Complete CPU metrics snapshot for one sample interval.
 *
 * Contains per-CPU data (frequency, load) for each monitored CPU,
 * plus a single process-level resource usage snapshot.
 */
struct metrics
{
    std::vector<per_cpu_metrics> cpu_data;
    process_metrics              process_data;
};

}  // namespace rocprofsys::pmc::collectors::cpu
