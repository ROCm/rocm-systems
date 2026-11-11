// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/units/data_size.hpp"
#include "core/categories.hpp"
#include "core/collection/metric_row.hpp"
#include "core/collection/schema.hpp"
#include "library/pmc/collectors/cpu/types.hpp"
#include "rocprofiler-systems/categories.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace rocprofsys::pmc::pollers::catalog::cpu
{
using enabled_t     = pmc::collectors::cpu::enabled_metrics;
using process_t     = pmc::collectors::cpu::process_metrics;
using core_t        = pmc::collectors::cpu::per_cpu_metrics;
using process_row_t = collection::metric_row<enabled_t, process_t>;
using core_row_t    = collection::metric_row<enabled_t, core_t>;

/** Bytes to MB. */
[[nodiscard]] inline double
to_megabytes(std::int64_t bytes_value) noexcept
{
    using common::units::bytes;
    using common::units::megabytes;
    return common::units::data_size_cast<megabytes>(
               bytes{ static_cast<double>(bytes_value) })
        .count();
}

/** Microseconds to seconds. */
[[nodiscard]] inline double
to_seconds(std::int64_t microseconds) noexcept
{
    return std::chrono::duration<double>{
        std::chrono::microseconds{ microseconds }
    }.count();
}

/** Process metrics: one column each, written under the first socket. */
inline constexpr auto k_process_metrics = std::to_array<process_row_t>({
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.page_rss != 0; },
      .read        = [](const process_t& info, std::size_t,
                 std::size_t) { return to_megabytes(info.page_rss); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_PAGE,
      .label       = "Memory Usage",
      .unit        = "MB",
      .symbol      = "Page RSS",
      .pmc_name    = {},
      .description = "Process Physical Memory (RSS)",
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.virt_mem != 0; },
      .read        = [](const process_t& info, std::size_t,
                 std::size_t) { return to_megabytes(info.virt_mem); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_VIRT,
      .label       = "Virtual Memory",
      .unit        = "MB",
      .symbol      = "Virt Mem",
      .pmc_name    = {},
      .description = "Process Virtual Memory",
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.peak_rss != 0; },
      .read        = [](const process_t& info, std::size_t,
                 std::size_t) { return to_megabytes(info.peak_rss); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_PEAK,
      .label       = "Peak Memory",
      .unit        = "MB",
      .symbol      = "Peak RSS",
      .pmc_name    = {},
      .description = "Process Peak Memory (HWM)",
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.ctx_switches != 0; },
      .read    = [](const process_t& info, std::size_t,
                 std::size_t) { return static_cast<double>(info.context_switches); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_CONTEXT_SWITCH,
      .label       = "Context Switches",
      .unit        = "count",
      .symbol      = "Ctx Switches",
      .pmc_name    = {},
      .description = "Context Switches",
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.page_faults != 0; },
      .read    = [](const process_t& info, std::size_t,
                 std::size_t) { return static_cast<double>(info.page_faults); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_PAGE_FAULT,
      .label       = "Page Faults",
      .unit        = "count",
      .symbol      = "Page Faults",
      .pmc_name    = {},
      .description = "Page Faults",
      .rocpd_track = {} },
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.user_time != 0; },
      .read        = [](const process_t& info, std::size_t,
                 std::size_t) { return to_seconds(info.user_mode_time); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_USER_MODE_TIME,
      .label       = "User Time",
      .unit        = "sec",
      .symbol      = "User Time",
      .pmc_name    = {},
      .description = "Process CPU Time in User Mode",
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.kernel_time != 0; },
      .read    = [](const process_t& info, std::size_t,
                 std::size_t) { return to_seconds(info.kernel_mode_time); },
      .category_id = ROCPROFSYS_CATEGORY_PROCESS_KERNEL_MODE_TIME,
      .label       = "Kernel Time",
      .unit        = "sec",
      .symbol      = "Kernel Time",
      .pmc_name    = {},
      .description = "Process CPU Time in Kernel Mode",
      .rocpd_track = {} },
});

/** Core metrics: one column per monitored core of the socket. */
inline constexpr auto k_core_metrics = std::to_array<core_row_t>({
    { .enabled     = [](const enabled_t& enabled) { return enabled.bits.frequency != 0; },
      .read        = [](const core_t& core, std::size_t,
                 std::size_t) { return static_cast<double>(core.frequency); },
      .category_id = ROCPROFSYS_CATEGORY_CPU_FREQ,
      .label       = "Frequency",
      .unit        = "MHz",
      .symbol      = {},
      .pmc_name    = {},
      .description = "CPU Core Frequency",
      .rocpd_track = {} },
    { .enabled = [](const enabled_t& enabled) { return enabled.bits.load != 0; },
      .read    = [](const core_t& core, std::size_t, std::size_t) { return core.load; },
      .category_id = ROCPROFSYS_CATEGORY_CPU_LOAD,
      .label       = "Load",
      .unit        = "%",
      .symbol      = {},
      .pmc_name    = {},
      .description = "CPU Core Load Percentage",
      .rocpd_track = {} },
});

/** The column of a process metric. */
[[nodiscard]] inline collection::column_spec
make_process_column(const process_row_t& row)
{
    const std::string pmc{ category_info(row.category_id).name };
    return { .symbol              = std::string{ row.symbol },
             .description         = std::string{ row.description },
             .unit                = std::string{ row.unit },
             .pmc_name            = pmc,
             .perfetto_track_name = fmt::format("CPU {} (S)", row.label),
             .rocpd_track_name    = pmc,
             .category_id         = row.category_id,
             .block               = {},
             .expression          = {} };
}

/** The column of a core metric of core `cpu_id` on socket `socket`. */
[[nodiscard]] inline collection::column_spec
make_core_column(const core_row_t& row, std::size_t socket, std::size_t cpu_id)
{
    const auto pmc = fmt::format("{} [{}] Core [{}]", category_info(row.category_id).name,
                                 socket, cpu_id);
    return { .symbol      = pmc,
             .description = std::string{ row.description },
             .unit        = std::string{ row.unit },
             .pmc_name    = pmc,
             .perfetto_track_name =
                 fmt::format("CPU [{}] Core [{}] {} (S)", socket, cpu_id, row.label),
             .rocpd_track_name = pmc,
             .category_id      = row.category_id,
             .block            = {},
             .expression       = {} };
}
}  // namespace rocprofsys::pmc::pollers::catalog::cpu
