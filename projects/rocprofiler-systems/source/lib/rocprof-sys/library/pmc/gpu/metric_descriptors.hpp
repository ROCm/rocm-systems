// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file metric_descriptors.hpp
 * @brief Type-safe metric identifiers and enabled_metrics class for AMD SMI GPU metrics.
 *
 * @details This file provides the core metric infrastructure for GPU Performance
 * Monitoring Counters (PMC). It defines:
 *
 * - @ref metric_id : Enum class with explicit bit positions for each metric
 * - @ref enabled_metrics : Bitset wrapper providing type-safe metric operations
 * - @ref metric_masks : Pre-computed bitmasks for common metric groups
 * - @ref user_metric_aliases : User-facing metric names for environment variable parsing
 * - @ref perfetto_track_info : Perfetto track descriptors for visualization
 *
 * @section adding_metrics Adding a New Metric
 *
 * To add a new metric:
 * 1. Add to @ref metric_id enum with explicit bit position
 * 2. Update COUNT in @ref metric_id
 * 3. Add getter/setter methods to @ref enabled_metrics class
 * 4. Add mask constant to @ref metric_masks namespace
 * 5. Optionally add to @ref user_metric_aliases for user configuration
 * 6. Add collection logic in device.hpp
 * 7. Add output handling in perfetto_policy.hpp and cache_policy.hpp
 *
 * @section metric_types Metric Types
 *
 * | Category | Metrics | Hardware |
 * |----------|---------|----------|
 * | Power | current_socket_power, average_socket_power | All GPUs |
 * | Temperature | hotspot_temperature, edge_temperature | All GPUs |
 * | Activity | gfx_activity, umc_activity, mm_activity | All GPUs |
 * | VCN/JPEG | vcn_activity, jpeg_activity | Radeon (device-level) |
 * | VCN/JPEG XCP | xcp_vcn_activity, xcp_jpeg_activity | MI300 (per-XCP) |
 * | Interconnect | xgmi, pcie | Data center GPUs |
 *
 * @author AMD
 * @since 1.5.0
 */

#include <spdlog/fmt/fmt.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace pmc
{
namespace gpu
{

/**
 * @enum metric_id
 * @brief Bit positions for each GPU metric used in the enabled_metrics bitmask.
 * @note When adding new metrics, ensure the bit position is unique and update COUNT.
 */
enum class metric_id : uint8_t
{
    current_socket_power = 0,
    average_socket_power = 1,
    memory_usage         = 2,
    hotspot_temperature  = 3,
    edge_temperature     = 4,
    gfx_activity         = 5,
    umc_activity         = 6,
    mm_activity          = 7,
    vcn_activity         = 8,   ///< Device-level
    jpeg_activity        = 9,   ///< Device-level
    xcp_vcn_activity     = 10,  ///< Per-XCP partition
    xcp_jpeg_activity    = 11,  ///< Per-XCP partition
    xgmi                 = 12,
    pcie                 = 13,
    COUNT                = 14
};

constexpr size_t METRIC_COUNT = static_cast<size_t>(metric_id::COUNT);

/**
 * @class enabled_metrics
 * @brief Type-safe bitset wrapper for enabled/supported GPU metrics.
 *
 * @details Provides a type-safe interface for managing which GPU metrics are
 * enabled for collection or supported by a device. Replaces the old union-based
 * implementation with explicit accessors for each metric type.
 *
 * Features:
 * - Type-safe accessors via @ref metric_id enum
 * - Raw value access for serialization and bitwise operations
 * - Named convenience methods for common metric checks
 * - Bitwise operators for combining metric sets
 *
 * @par Example Usage:
 * @code
 * enabled_metrics user_requested;
 * user_requested.set_gfx_activity(true);
 * user_requested.set_memory_usage(true);
 *
 * enabled_metrics device_supported = device.get_supported_metrics();
 *
 * // Compute effective metrics (intersection)
 * enabled_metrics effective = user_requested & device_supported;
 *
 * if (effective.gfx_activity()) {
 *     // Collect GFX activity metric
 * }
 * @endcode
 */
class enabled_metrics
{
public:
    constexpr enabled_metrics() = default;

    /**
     * @brief Construct from raw bitmask value.
     * @param raw_value Bitmask where each bit corresponds to a metric_id
     */
    constexpr explicit enabled_metrics(uint32_t raw_value)
    : m_value(raw_value)
    {}

    /**
     * @brief Get the raw bitmask value.
     * @return The underlying bitmask
     */
    [[nodiscard]] constexpr uint32_t value() const noexcept { return m_value; }

    /**
     * @brief Set the raw bitmask value.
     * @param raw_value New bitmask value
     */
    constexpr void set_value(uint32_t raw_value) noexcept { m_value = raw_value; }

    /**
     * @brief Get mutable reference to raw value (for parse_value compatibility).
     * @return Reference to the underlying bitmask
     */
    constexpr uint32_t& value_ref() noexcept { return m_value; }

    /**
     * @brief Test if a specific metric is enabled.
     * @param id The metric to test
     * @return true if the metric is enabled
     */
    [[nodiscard]] constexpr bool test(metric_id id) const noexcept
    {
        return ((m_value >> static_cast<uint8_t>(id)) & 1u) != 0;
    }

    /**
     * @brief Enable a specific metric.
     * @param id The metric to enable
     */
    constexpr void set(metric_id id) noexcept
    {
        m_value |= (1u << static_cast<uint8_t>(id));
    }

    /**
     * @brief Disable a specific metric.
     * @param id The metric to disable
     */
    constexpr void reset(metric_id id) noexcept
    {
        m_value &= ~(1u << static_cast<uint8_t>(id));
    }

    /**
     * @brief Set a metric to a specific enabled/disabled state.
     * @param id The metric to modify
     * @param value true to enable, false to disable
     */
    constexpr void set(metric_id id, bool value) noexcept
    {
        if(value)
            set(id);
        else
            reset(id);
    }

    /**
     * @brief Enable all metrics.
     */
    constexpr void enable_all() noexcept { m_value = (1u << METRIC_COUNT) - 1; }

    /**
     * @brief Disable all metrics.
     */
    constexpr void disable_all() noexcept { m_value = 0; }

    /**
     * @brief Check if any metrics are enabled.
     * @return true if at least one metric is enabled
     */
    [[nodiscard]] constexpr bool any() const noexcept { return m_value != 0; }

    /**
     * @brief Check if no metrics are enabled.
     * @return true if all metrics are disabled
     */
    [[nodiscard]] constexpr bool none() const noexcept { return m_value == 0; }

    [[nodiscard]] constexpr bool current_socket_power() const noexcept
    {
        return test(metric_id::current_socket_power);
    }
    [[nodiscard]] constexpr bool average_socket_power() const noexcept
    {
        return test(metric_id::average_socket_power);
    }
    [[nodiscard]] constexpr bool memory_usage() const noexcept
    {
        return test(metric_id::memory_usage);
    }
    [[nodiscard]] constexpr bool hotspot_temperature() const noexcept
    {
        return test(metric_id::hotspot_temperature);
    }
    [[nodiscard]] constexpr bool edge_temperature() const noexcept
    {
        return test(metric_id::edge_temperature);
    }
    [[nodiscard]] constexpr bool gfx_activity() const noexcept
    {
        return test(metric_id::gfx_activity);
    }
    [[nodiscard]] constexpr bool umc_activity() const noexcept
    {
        return test(metric_id::umc_activity);
    }
    [[nodiscard]] constexpr bool mm_activity() const noexcept
    {
        return test(metric_id::mm_activity);
    }
    [[nodiscard]] constexpr bool vcn_activity() const noexcept
    {
        return test(metric_id::vcn_activity);
    }
    [[nodiscard]] constexpr bool jpeg_activity() const noexcept
    {
        return test(metric_id::jpeg_activity);
    }
    [[nodiscard]] constexpr bool xcp_vcn_activity() const noexcept
    {
        return test(metric_id::xcp_vcn_activity);
    }
    [[nodiscard]] constexpr bool xcp_jpeg_activity() const noexcept
    {
        return test(metric_id::xcp_jpeg_activity);
    }
    [[nodiscard]] constexpr bool xgmi() const noexcept { return test(metric_id::xgmi); }
    [[nodiscard]] constexpr bool pcie() const noexcept { return test(metric_id::pcie); }

    constexpr void set_current_socket_power(bool v) noexcept
    {
        set(metric_id::current_socket_power, v);
    }
    constexpr void set_average_socket_power(bool v) noexcept
    {
        set(metric_id::average_socket_power, v);
    }
    constexpr void set_memory_usage(bool v) noexcept { set(metric_id::memory_usage, v); }
    constexpr void set_hotspot_temperature(bool v) noexcept
    {
        set(metric_id::hotspot_temperature, v);
    }
    constexpr void set_edge_temperature(bool v) noexcept
    {
        set(metric_id::edge_temperature, v);
    }
    constexpr void set_gfx_activity(bool v) noexcept { set(metric_id::gfx_activity, v); }
    constexpr void set_umc_activity(bool v) noexcept { set(metric_id::umc_activity, v); }
    constexpr void set_mm_activity(bool v) noexcept { set(metric_id::mm_activity, v); }
    constexpr void set_vcn_activity(bool v) noexcept { set(metric_id::vcn_activity, v); }
    constexpr void set_jpeg_activity(bool v) noexcept
    {
        set(metric_id::jpeg_activity, v);
    }
    constexpr void set_xcp_vcn_activity(bool v) noexcept
    {
        set(metric_id::xcp_vcn_activity, v);
    }
    constexpr void set_xcp_jpeg_activity(bool v) noexcept
    {
        set(metric_id::xcp_jpeg_activity, v);
    }
    constexpr void set_xgmi(bool v) noexcept { set(metric_id::xgmi, v); }
    constexpr void set_pcie(bool v) noexcept { set(metric_id::pcie, v); }

    constexpr enabled_metrics operator&(const enabled_metrics& other) const noexcept
    {
        return enabled_metrics(m_value & other.m_value);
    }
    constexpr enabled_metrics operator|(const enabled_metrics& other) const noexcept
    {
        return enabled_metrics(m_value | other.m_value);
    }
    constexpr enabled_metrics& operator&=(const enabled_metrics& other) noexcept
    {
        m_value &= other.m_value;
        return *this;
    }
    constexpr enabled_metrics& operator|=(const enabled_metrics& other) noexcept
    {
        m_value |= other.m_value;
        return *this;
    }
    constexpr enabled_metrics operator~() const noexcept
    {
        return enabled_metrics(~m_value & ((1u << METRIC_COUNT) - 1));
    }
    constexpr bool operator==(const enabled_metrics& other) const noexcept
    {
        return m_value == other.m_value;
    }
    constexpr bool operator!=(const enabled_metrics& other) const noexcept
    {
        return m_value != other.m_value;
    }

private:
    uint32_t m_value = 0;
};

/**
 * @brief Create an enabled_metrics with specified metrics enabled.
 * @tparam Ids Variadic pack of metric_id values
 * @param ids The metrics to enable
 * @return enabled_metrics with the specified metrics set
 *
 * @par Example:
 * @code
 * auto metrics = make_enabled_metrics(metric_id::gfx_activity, metric_id::power);
 * @endcode
 */
template <typename... Ids>
constexpr enabled_metrics
make_enabled_metrics(Ids... ids)
{
    enabled_metrics result;
    (result.set(ids), ...);
    return result;
}

/**
 * @brief Create a bitmask from metric IDs (for static initialization).
 * @tparam Ids Variadic pack of metric_id values
 * @param ids The metrics to include in the mask
 * @return uint32_t bitmask with bits set for each metric
 *
 * @par Example:
 * @code
 * constexpr uint32_t power_mask = make_metric_mask(
 *     metric_id::current_socket_power,
 *     metric_id::average_socket_power);
 * @endcode
 */
template <typename... Ids>
constexpr uint32_t
make_metric_mask(Ids... ids)
{
    return ((1u << static_cast<uint8_t>(ids)) | ...);
}

/**
 * @brief Convert enabled_metrics to string for logging.
 * @param metrics The enabled_metrics to convert
 * @return String representation listing all metrics and their enabled state
 */
inline std::string
to_string(const enabled_metrics& metrics)
{
    return fmt::format(
        "[SMI enabled metrics] Current socket power: {}, Average socket power: {}, "
        "Memory usage: {}, Hotspot temperature: {}, Edge temperature: {}, "
        "GFX activity: {}, UMC activity: {}, MM activity: {}, "
        "VCN activity: {}, JPEG activity: {}, XCP VCN activity: {}, "
        "XCP JPEG activity: {}, XGMI: {}, PCIe: {}\n",
        metrics.current_socket_power(), metrics.average_socket_power(),
        metrics.memory_usage(), metrics.hotspot_temperature(), metrics.edge_temperature(),
        metrics.gfx_activity(), metrics.umc_activity(), metrics.mm_activity(),
        metrics.vcn_activity(), metrics.jpeg_activity(), metrics.xcp_vcn_activity(),
        metrics.xcp_jpeg_activity(), metrics.xgmi(), metrics.pcie());
}

/** @brief Pre-computed bitmasks for metric groups and individual metrics. */
namespace metric_masks
{

/** @brief Combined masks for metric groups */
constexpr uint32_t temperature =
    make_metric_mask(metric_id::hotspot_temperature, metric_id::edge_temperature);
constexpr uint32_t power =
    make_metric_mask(metric_id::current_socket_power, metric_id::average_socket_power);
constexpr uint32_t busy = make_metric_mask(
    metric_id::gfx_activity, metric_id::umc_activity, metric_id::mm_activity);

/** @brief Individual metric masks */
constexpr uint32_t gfx_activity      = make_metric_mask(metric_id::gfx_activity);
constexpr uint32_t umc_activity      = make_metric_mask(metric_id::umc_activity);
constexpr uint32_t mm_activity       = make_metric_mask(metric_id::mm_activity);
constexpr uint32_t memory_usage      = make_metric_mask(metric_id::memory_usage);
constexpr uint32_t vcn_activity      = make_metric_mask(metric_id::vcn_activity);
constexpr uint32_t jpeg_activity     = make_metric_mask(metric_id::jpeg_activity);
constexpr uint32_t xcp_vcn_activity  = make_metric_mask(metric_id::xcp_vcn_activity);
constexpr uint32_t xcp_jpeg_activity = make_metric_mask(metric_id::xcp_jpeg_activity);
constexpr uint32_t xgmi_mask         = make_metric_mask(metric_id::xgmi);
constexpr uint32_t pcie_mask         = make_metric_mask(metric_id::pcie);

constexpr uint32_t all  = (1u << METRIC_COUNT) - 1;
constexpr uint32_t none = 0;

}  // namespace metric_masks

/** @brief Maps user-friendly names to metric bitmasks for ROCPROFSYS_AMD_SMI_METRICS. */
struct metric_alias
{
    std::string_view name;
    uint32_t         mask;
};

constexpr std::array<metric_alias, 8> user_metric_aliases = { {
    { "temp", metric_masks::temperature },
    { "power", metric_masks::power },
    { "busy", metric_masks::busy },
    { "mem_usage", metric_masks::memory_usage },
    { "vcn_activity", metric_masks::vcn_activity },
    { "jpeg_activity", metric_masks::jpeg_activity },
    { "xgmi", metric_masks::xgmi_mask },
    { "pcie", metric_masks::pcie_mask },
} };

constexpr std::string_view metric_validation_pattern =
    R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie))"
    R"((?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie))*$)";

/** @brief Track descriptor for Perfetto output visualization. */
struct perfetto_track_info
{
    uint32_t         mask;
    std::string_view track_name;
    std::string_view units;
};

constexpr std::array<perfetto_track_info, 12> perfetto_tracks = { {
    { metric_masks::gfx_activity, "GFX Busy", "%" },
    { metric_masks::umc_activity, "UMC Busy", "%" },
    { metric_masks::mm_activity, "MM Busy", "%" },
    { metric_masks::temperature, "Temperature", "deg C" },
    { metric_masks::power, "Current Power", "watts" },
    { metric_masks::memory_usage, "Memory Usage", "megabytes" },
    { metric_masks::vcn_activity, "VCN Activity", "%" },
    { metric_masks::jpeg_activity, "JPEG Activity", "%" },
    { metric_masks::xcp_vcn_activity, "VCN Busy", "%" },
    { metric_masks::xcp_jpeg_activity, "JPEG Busy", "%" },
    { metric_masks::xgmi_mask, "XGMI", "" },
    { metric_masks::pcie_mask, "PCIe", "" },
} };

}  // namespace gpu
}  // namespace pmc
}  // namespace rocprofsys
