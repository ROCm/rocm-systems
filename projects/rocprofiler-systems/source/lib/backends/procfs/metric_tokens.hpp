// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace rocprofsys::backends::procfs::cpu
{

enum class metric_kind : std::uint8_t
{
    frequency,
    load,
    memory,
    page_rss,
    virt_mem,
    peak_rss,
    context_switches,
    page_faults,
    cpu_time,
    user_time,
    kernel_time,
};

enum class metric_scope : std::uint8_t
{
    logical_cpu,
    process,
};

struct metric_token
{
    metric_kind      kind = metric_kind::frequency;
    std::string_view name;
    std::string_view description;
    std::string_view unit;
    metric_scope     scope = metric_scope::process;
};

inline constexpr std::string_view k_option_description =
    "--cpu-metrics lists procfs CPU and process sampling metrics this machine can "
    "collect.";
inline constexpr std::string_view k_selection_help =
    "Pass a comma-separated token list, or all (every supported token) or none "
    "(disable).";
inline constexpr std::string_view k_default_help =
    "Default when the flag is omitted: all.";
inline constexpr std::string_view k_default_tokens = "all";

inline constexpr std::uint32_t k_frequency_mask      = 1U << 0U;
inline constexpr std::uint32_t k_load_mask           = 1U << 1U;
inline constexpr std::uint32_t k_page_rss_mask       = 1U << 2U;
inline constexpr std::uint32_t k_virt_mem_mask       = 1U << 3U;
inline constexpr std::uint32_t k_peak_rss_mask       = 1U << 4U;
inline constexpr std::uint32_t k_context_switch_mask = 1U << 5U;
inline constexpr std::uint32_t k_page_faults_mask    = 1U << 6U;
inline constexpr std::uint32_t k_user_time_mask      = 1U << 7U;
inline constexpr std::uint32_t k_kernel_time_mask    = 1U << 8U;
inline constexpr std::uint32_t k_memory_mask =
    k_page_rss_mask | k_virt_mem_mask | k_peak_rss_mask;
inline constexpr std::uint32_t k_cpu_time_mask = k_user_time_mask | k_kernel_time_mask;

inline constexpr auto k_metric_tokens = std::array{
    metric_token{ .kind        = metric_kind::frequency,
                  .name        = "frequency",
                  .description = "Current logical CPU frequency",
                  .unit        = "MHz",
                  .scope       = metric_scope::logical_cpu },
    metric_token{ .kind        = metric_kind::load,
                  .name        = "load",
                  .description = "Logical CPU utilization",
                  .unit        = "%",
                  .scope       = metric_scope::logical_cpu },
    metric_token{ .kind        = metric_kind::memory,
                  .name        = "memory",
                  .description = "Process memory group (page_rss, virt_mem, peak_rss)",
                  .unit        = "bytes" },
    metric_token{ .kind        = metric_kind::page_rss,
                  .name        = "page_rss",
                  .description = "Process resident set size",
                  .unit        = "bytes" },
    metric_token{ .kind        = metric_kind::virt_mem,
                  .name        = "virt_mem",
                  .description = "Process virtual memory size",
                  .unit        = "bytes" },
    metric_token{ .kind        = metric_kind::peak_rss,
                  .name        = "peak_rss",
                  .description = "Process peak resident set size",
                  .unit        = "bytes" },
    metric_token{ .kind        = metric_kind::context_switches,
                  .name        = "ctx_switches",
                  .description = "Process voluntary and involuntary context switches",
                  .unit        = "count" },
    metric_token{ .kind        = metric_kind::page_faults,
                  .name        = "page_faults",
                  .description = "Process major and minor page faults",
                  .unit        = "count" },
    metric_token{ .kind        = metric_kind::cpu_time,
                  .name        = "cpu_time",
                  .description = "Process CPU time group (user_time, kernel_time)",
                  .unit        = "us" },
    metric_token{ .kind        = metric_kind::user_time,
                  .name        = "user_time",
                  .description = "Process user-mode CPU time",
                  .unit        = "us" },
    metric_token{ .kind        = metric_kind::kernel_time,
                  .name        = "kernel_time",
                  .description = "Process kernel-mode CPU time",
                  .unit        = "us" },
};

[[nodiscard]] constexpr std::uint32_t
metric_selection_mask(metric_kind kind) noexcept
{
    switch(kind)
    {
        case metric_kind::frequency: return k_frequency_mask;
        case metric_kind::load: return k_load_mask;
        case metric_kind::memory: return k_memory_mask;
        case metric_kind::page_rss: return k_page_rss_mask;
        case metric_kind::virt_mem: return k_virt_mem_mask;
        case metric_kind::peak_rss: return k_peak_rss_mask;
        case metric_kind::context_switches: return k_context_switch_mask;
        case metric_kind::page_faults: return k_page_faults_mask;
        case metric_kind::cpu_time: return k_cpu_time_mask;
        case metric_kind::user_time: return k_user_time_mask;
        case metric_kind::kernel_time: return k_kernel_time_mask;
    }
    return 0U;
}

}  // namespace rocprofsys::backends::procfs::cpu
