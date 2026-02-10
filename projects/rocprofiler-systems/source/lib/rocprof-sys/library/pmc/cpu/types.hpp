// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "library/pmc/common/types.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace cpu
{

/**
 * @brief Bitfield union for selecting which CPU metrics to collect.
 *
 * Bit positions (for value access):
 *   - frequency     = 0   (per-CPU, from /proc/cpuinfo, MHz)
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
        uint32_t frequency    : 1;
        uint32_t load         : 1;
        uint32_t page_rss     : 1;
        uint32_t virt_mem     : 1;
        uint32_t peak_rss     : 1;
        uint32_t ctx_switches : 1;
        uint32_t page_faults  : 1;
        uint32_t user_time    : 1;
        uint32_t kernel_time  : 1;
    } bits;
    uint32_t value = 0;
};

inline std::string
to_string(const enabled_metrics& metrics)
{
    std::stringstream ss;
    ss << "[CPU enabled metrics] ";
    ss << "Frequency: " << static_cast<bool>(metrics.bits.frequency)
       << ", Load: " << static_cast<bool>(metrics.bits.load)
       << ", Page RSS: " << static_cast<bool>(metrics.bits.page_rss)
       << ", Virtual memory: " << static_cast<bool>(metrics.bits.virt_mem)
       << ", Peak RSS: " << static_cast<bool>(metrics.bits.peak_rss)
       << ", Context switches: " << static_cast<bool>(metrics.bits.ctx_switches)
       << ", Page faults: " << static_cast<bool>(metrics.bits.page_faults)
       << ", User time: " << static_cast<bool>(metrics.bits.user_time)
       << ", Kernel time: " << static_cast<bool>(metrics.bits.kernel_time) << "\n";
    return ss.str();
}

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
 * These are per-process, not per-CPU. Collected via getrusage() and /proc/self/statm.
 */
struct process_metrics
{
    int64_t page_rss         = 0;  // bytes (physical memory RSS)
    int64_t virt_mem         = 0;  // bytes (virtual memory)
    int64_t peak_rss         = 0;  // bytes (peak memory HWM)
    int64_t context_switches = 0;  // count (voluntary + involuntary)
    int64_t page_faults      = 0;  // count (major + minor)
    int64_t user_mode_time   = 0;  // microseconds
    int64_t kernel_mode_time = 0;  // microseconds
};

/**
 * @brief Complete CPU metrics snapshot for one sample interval.
 *
 * Contains per-CPU data (frequency, load) for each monitored CPU,
 * plus a single process-level resource usage snapshot.
 */
struct metrics
{
    std::vector<per_cpu_metrics> cpu_data;      // one entry per monitored CPU
    process_metrics              process_data;  // single process-level snapshot
};

}  // namespace cpu
}  // namespace pmc
}  // namespace rocprofsys
