// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string_view>
#include <type_traits>

namespace rocprofsys::backends::amd_smi::gpu
{

enum class metric_group : std::uint8_t
{
    busy,
    temperature,
    power,
    memory_usage,
    sdma_usage,
    gfx_clock,
    memory_clock,
    vcn_activity,
    jpeg_activity,
    xgmi,
    pcie,
    count,
};

struct metric_token
{
    metric_group                    group = metric_group::busy;
    std::string_view                name;
    std::string_view                description;
    std::string_view                unit;
    std::array<std::string_view, 2> aliases;
    std::size_t                     alias_count = 0;
};

inline constexpr std::string_view k_option_description =
    "--gpu-metrics lists AMD SMI GPU sampling groups this machine can collect.";
inline constexpr std::string_view k_selection_help =
    "Pass a comma-separated token list, or all (every supported token) or none "
    "(disable).";
inline constexpr std::string_view k_default_help =
    "Default when the flag is omitted: busy,temp,power,mem_usage.";
inline constexpr std::string_view k_default_tokens = "busy,temp,power,mem_usage";

inline constexpr auto k_metric_tokens = std::array{
    metric_token{ .group       = metric_group::busy,
                  .name        = "busy",
                  .description = "GFX, UMC, and multimedia engine busy percentages",
                  .unit        = "%",
                  .aliases     = { "usage", "utilization" },
                  .alias_count = 2 },
    metric_token{ .group       = metric_group::temperature,
                  .name        = "temp",
                  .description = "GPU hotspot and edge temperatures",
                  .unit        = "C",
                  .aliases     = { "temperature", "" },
                  .alias_count = 1 },
    metric_token{ .group       = metric_group::power,
                  .name        = "power",
                  .description = "Current and average socket power",
                  .unit        = "W",
                  .aliases     = {} },
    metric_token{ .group       = metric_group::memory_usage,
                  .name        = "mem_usage",
                  .description = "GPU memory usage",
                  .unit        = "bytes",
                  .aliases     = { "memory", "" },
                  .alias_count = 1 },
    metric_token{ .group       = metric_group::sdma_usage,
                  .name        = "sdma_usage",
                  .description = "SDMA utilization percentage",
                  .unit        = "%",
                  .aliases     = {} },
    metric_token{ .group       = metric_group::gfx_clock,
                  .name        = "gfx_clock",
                  .description = "GFX clock frequency",
                  .unit        = "MHz",
                  .aliases     = {} },
    metric_token{ .group       = metric_group::memory_clock,
                  .name        = "mem_clock",
                  .description = "Memory clock frequency",
                  .unit        = "MHz",
                  .aliases     = {} },
    metric_token{ .group       = metric_group::vcn_activity,
                  .name        = "vcn_activity",
                  .description = "VCN activity reported per device or per GPU partition",
                  .unit        = "%",
                  .aliases     = {} },
    metric_token{ .group       = metric_group::jpeg_activity,
                  .name        = "jpeg_activity",
                  .description = "JPEG activity reported per device or per GPU partition",
                  .unit        = "%",
                  .aliases     = {} },
    metric_token{ .group = metric_group::xgmi,
                  .name  = "xgmi",
                  .description =
                      "XGMI link width, speed, and per-link read/write data accumulators",
                  .unit    = "",
                  .aliases = {} },
    metric_token{
        .group = metric_group::pcie,
        .name  = "pcie",
        .description =
            "PCIe link width, speed, and instantaneous and accumulated bandwidth",
        .unit    = "",
        .aliases = {} },
};

static_assert(k_metric_tokens.size() == static_cast<std::size_t>(metric_group::count));

struct metric_support
{
    std::array<bool, k_metric_tokens.size()> groups{};

    [[nodiscard]] constexpr bool operator[](metric_group group) const noexcept
    {
        return groups.at(static_cast<std::size_t>(group));
    }

    constexpr void set(metric_group group, bool supported = true) noexcept
    {
        groups.at(static_cast<std::size_t>(group)) = supported;
    }
};

template <std::integral T>
[[nodiscard]] constexpr bool
has_metric_value(T value) noexcept
{
    using value_type = std::remove_cvref_t<T>;
    return value != std::numeric_limits<value_type>::max();
}

template <std::ranges::input_range Range>
[[nodiscard]] constexpr bool
has_metric_value(const Range& values) noexcept
{
    return std::ranges::any_of(values,
                               [](const auto value) { return has_metric_value(value); });
}

template <typename Metrics>
[[nodiscard]] constexpr bool
has_vcn_value(const Metrics& metrics) noexcept
{
    return has_metric_value(metrics.vcn_activity) ||
           std::ranges::any_of(metrics.xcp_stats, [](const auto& xcp) {
               return has_metric_value(xcp.vcn_busy);
           });
}

template <typename Metrics>
[[nodiscard]] constexpr bool
has_jpeg_value(const Metrics& metrics) noexcept
{
    return has_metric_value(metrics.jpeg_activity) ||
           std::ranges::any_of(metrics.xcp_stats, [](const auto& xcp) {
               return has_metric_value(xcp.jpeg_busy);
           });
}

/**
 * Detects the CLI metric groups represented by an AMD SMI GPU metrics table.
 * Memory and SDMA support require separate AMD SMI APIs and remain false here.
 *
 * @param metrics Raw AMD SMI metrics table with unsupported values left as sentinels.
 * @return Supported metric groups discoverable from the table.
 */
template <typename Metrics>
[[nodiscard]] constexpr metric_support
detect_metric_support(const Metrics& metrics) noexcept
{
    metric_support result;
    result.set(metric_group::busy, has_metric_value(metrics.average_gfx_activity) ||
                                       has_metric_value(metrics.average_umc_activity) ||
                                       has_metric_value(metrics.average_mm_activity));
    result.set(metric_group::temperature, has_metric_value(metrics.temperature_hotspot) ||
                                              has_metric_value(metrics.temperature_edge));
    result.set(metric_group::power, has_metric_value(metrics.current_socket_power) ||
                                        has_metric_value(metrics.average_socket_power));
    result.set(metric_group::gfx_clock, has_metric_value(metrics.current_gfxclk));
    result.set(metric_group::memory_clock, has_metric_value(metrics.current_uclk));
    result.set(metric_group::vcn_activity, has_vcn_value(metrics));
    result.set(metric_group::jpeg_activity, has_jpeg_value(metrics));
    result.set(metric_group::xgmi, has_metric_value(metrics.xgmi_link_width) ||
                                       has_metric_value(metrics.xgmi_link_speed) ||
                                       has_metric_value(metrics.xgmi_read_data_acc) ||
                                       has_metric_value(metrics.xgmi_write_data_acc));
    result.set(metric_group::pcie, has_metric_value(metrics.pcie_link_width) ||
                                       has_metric_value(metrics.pcie_link_speed) ||
                                       has_metric_value(metrics.pcie_bandwidth_acc) ||
                                       has_metric_value(metrics.pcie_bandwidth_inst));
    return result;
}

}  // namespace rocprofsys::backends::amd_smi::gpu
