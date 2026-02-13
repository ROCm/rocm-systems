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

/**
 * @file metric_descriptors.hpp
 * @brief Type-safe metric identifiers and enabled_metrics class for AMD SMI GPU metrics.
 *
 * This file defines:
 * - metric_id: Enum class with explicit bit positions for each metric
 * - enabled_metrics: Bitset wrapper providing type-safe metric operations
 * - User-facing metric aliases for environment variable parsing
 * - Perfetto track descriptors
 *
 * To add a new metric:
 * 1. Add to metric_id enum with explicit bit position
 * 2. Update COUNT
 * 3. Add accessor method to enabled_metrics class
 * 4. Optionally add to user_metric_aliases
 * 5. Add collection logic in device.hpp
 * 6. Add output handling in perfetto_policy.hpp and cache_policy.hpp
 */

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace pmc
{
namespace gpu
{

/**
 * @brief Enum defining bit positions for each GPU metric.
 *
 * Each metric has an explicit bit position. Values can have gaps and
 * be in any order - the enum value IS the bit position.
 */
enum class metric_id : uint8_t
{
    // Power metrics
    current_socket_power = 0,
    average_socket_power = 1,

    // Memory metrics
    memory_usage = 2,

    // Temperature metrics
    hotspot_temperature = 3,
    edge_temperature    = 4,

    // Activity metrics
    gfx_activity = 5,
    umc_activity = 6,
    mm_activity  = 7,

    // VCN/JPEG metrics (device-level, Radeon GPUs)
    vcn_activity  = 8,
    jpeg_activity = 9,

    // VCN/JPEG metrics (per-XCP, MI300 series)
    vcn_busy  = 10,
    jpeg_busy = 11,

    // Interconnect metrics
    xgmi = 12,
    pcie = 13,

    // Total count (must be last, update when adding metrics)
    COUNT = 14
};

/**
 * @brief Number of metrics (for bitset sizing).
 */
constexpr size_t METRIC_COUNT = static_cast<size_t>(metric_id::COUNT);

/**
 * @brief Type-safe bitset wrapper for enabled/supported metrics.
 *
 * Replaces the old union-based enabled_metrics with a class that provides:
 * - Type-safe accessors via metric_id enum
 * - Raw value access for serialization and bitwise operations
 * - Named convenience methods for common checks
 */
class enabled_metrics
{
public:
    // =========================================================================
    // Constructors
    // =========================================================================

    constexpr enabled_metrics() = default;

    constexpr explicit enabled_metrics(uint32_t raw_value)
    : m_value(raw_value)
    {}

    // =========================================================================
    // Raw value access (for serialization, bitwise ops, etc.)
    // =========================================================================

    /// Get the raw bitmask value
    [[nodiscard]] constexpr uint32_t value() const noexcept { return m_value; }

    /// Set the raw bitmask value
    constexpr void set_value(uint32_t raw_value) noexcept { m_value = raw_value; }

    /// Get mutable reference to raw value (for parse_value compatibility)
    constexpr uint32_t& value_ref() noexcept { return m_value; }

    // =========================================================================
    // Type-safe metric accessors
    // =========================================================================

    /// Test if a metric is enabled
    [[nodiscard]] constexpr bool test(metric_id id) const noexcept
    {
        return ((m_value >> static_cast<uint8_t>(id)) & 1u) != 0;
    }

    /// Enable a metric
    constexpr void set(metric_id id) noexcept
    {
        m_value |= (1u << static_cast<uint8_t>(id));
    }

    /// Disable a metric
    constexpr void reset(metric_id id) noexcept
    {
        m_value &= ~(1u << static_cast<uint8_t>(id));
    }

    /// Set a metric to a specific value
    constexpr void set(metric_id id, bool value) noexcept
    {
        if(value)
            set(id);
        else
            reset(id);
    }

    /// Enable all metrics
    constexpr void set_all() noexcept { m_value = (1u << METRIC_COUNT) - 1; }

    /// Disable all metrics
    constexpr void reset_all() noexcept { m_value = 0; }

    /// Check if any metrics are enabled
    [[nodiscard]] constexpr bool any() const noexcept { return m_value != 0; }

    /// Check if no metrics are enabled
    [[nodiscard]] constexpr bool none() const noexcept { return m_value == 0; }

    // =========================================================================
    // Named convenience accessors (for readable code)
    // =========================================================================

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
    [[nodiscard]] constexpr bool vcn_busy() const noexcept
    {
        return test(metric_id::vcn_busy);
    }
    [[nodiscard]] constexpr bool jpeg_busy() const noexcept
    {
        return test(metric_id::jpeg_busy);
    }
    [[nodiscard]] constexpr bool xgmi() const noexcept { return test(metric_id::xgmi); }
    [[nodiscard]] constexpr bool pcie() const noexcept { return test(metric_id::pcie); }

    // Setters for each metric
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
    constexpr void set_vcn_busy(bool v) noexcept { set(metric_id::vcn_busy, v); }
    constexpr void set_jpeg_busy(bool v) noexcept { set(metric_id::jpeg_busy, v); }
    constexpr void set_xgmi(bool v) noexcept { set(metric_id::xgmi, v); }
    constexpr void set_pcie(bool v) noexcept { set(metric_id::pcie, v); }

    // =========================================================================
    // Bitwise operators
    // =========================================================================

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

// =============================================================================
// Helper functions
// =============================================================================

/**
 * @brief Create an enabled_metrics with specified metrics enabled.
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
 */
template <typename... Ids>
constexpr uint32_t
make_metric_mask(Ids... ids)
{
    return ((1u << static_cast<uint8_t>(ids)) | ...);
}

/**
 * @brief Convert enabled_metrics to string for logging.
 */
inline std::string
to_string(const enabled_metrics& metrics)
{
    std::stringstream ss;
    ss << "[SMI enabled metrics] ";
    ss << "Current socket power: " << metrics.current_socket_power()
       << ", Average socket power: " << metrics.average_socket_power()
       << ", Memory usage: " << metrics.memory_usage()
       << ", Hotspot temperature: " << metrics.hotspot_temperature()
       << ", Edge temperature: " << metrics.edge_temperature()
       << ", GFX activity: " << metrics.gfx_activity()
       << ", UMC activity: " << metrics.umc_activity()
       << ", MM activity: " << metrics.mm_activity()
       << ", VCN activity: " << metrics.vcn_activity()
       << ", JPEG activity: " << metrics.jpeg_activity()
       << ", VCN busy: " << metrics.vcn_busy() << ", JPEG busy: " << metrics.jpeg_busy()
       << ", XGMI: " << metrics.xgmi() << ", PCIE: " << metrics.pcie() << "\n";
    return ss.str();
}

// =============================================================================
// Pre-computed metric masks for common metric groups
// =============================================================================

namespace metric_masks
{
// Temperature: hotspot + edge
constexpr uint32_t temperature =
    make_metric_mask(metric_id::hotspot_temperature, metric_id::edge_temperature);

// Power: current + average
constexpr uint32_t power =
    make_metric_mask(metric_id::current_socket_power, metric_id::average_socket_power);

// Activity/Busy: gfx + umc + mm
constexpr uint32_t busy = make_metric_mask(
    metric_id::gfx_activity, metric_id::umc_activity, metric_id::mm_activity);

// Individual metrics
constexpr uint32_t gfx_activity  = make_metric_mask(metric_id::gfx_activity);
constexpr uint32_t umc_activity  = make_metric_mask(metric_id::umc_activity);
constexpr uint32_t mm_activity   = make_metric_mask(metric_id::mm_activity);
constexpr uint32_t memory_usage  = make_metric_mask(metric_id::memory_usage);
constexpr uint32_t vcn_activity  = make_metric_mask(metric_id::vcn_activity);
constexpr uint32_t jpeg_activity = make_metric_mask(metric_id::jpeg_activity);
constexpr uint32_t xgmi_mask     = make_metric_mask(metric_id::xgmi);
constexpr uint32_t pcie_mask     = make_metric_mask(metric_id::pcie);

// All metrics enabled
constexpr uint32_t all = (1u << METRIC_COUNT) - 1;

// No metrics enabled
constexpr uint32_t none = 0;

}  // namespace metric_masks

// =============================================================================
// User-facing metric aliases for environment variable parsing
// =============================================================================

/**
 * @brief Descriptor for a user-facing metric alias.
 */
struct metric_alias
{
    std::string_view name;  ///< User-facing name (e.g., "temp", "power")
    uint32_t         mask;  ///< Bitmask of metrics this alias enables
};

/**
 * @brief User-facing metric aliases for ROCPROFSYS_AMD_SMI_METRICS parsing.
 */
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

/**
 * @brief Regex pattern for validating user metric input.
 */
constexpr std::string_view metric_validation_pattern =
    R"(^(?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie))"
    R"((?:[,;](?:temp|power|busy|mem_usage|vcn_activity|jpeg_activity|xgmi|pcie))*$)";

// =============================================================================
// Perfetto track descriptors
// =============================================================================

/**
 * @brief Track descriptor for Perfetto output.
 */
struct perfetto_track_info
{
    uint32_t         mask;        ///< Metric mask this track represents
    std::string_view track_name;  ///< Display name in Perfetto
    std::string_view units;       ///< Units (%, watts, deg C, etc.)
};

/**
 * @brief Perfetto track descriptors for basic metrics.
 */
constexpr std::array<perfetto_track_info, 10> perfetto_tracks = { {
    { metric_masks::gfx_activity, "GFX Busy", "%" },
    { metric_masks::umc_activity, "UMC Busy", "%" },
    { metric_masks::mm_activity, "MM Busy", "%" },
    { metric_masks::temperature, "Temperature", "deg C" },
    { metric_masks::power, "Current Power", "watts" },
    { metric_masks::memory_usage, "Memory Usage", "megabytes" },
    { metric_masks::vcn_activity, "VCN Activity", "%" },
    { metric_masks::jpeg_activity, "JPEG Activity", "%" },
    { metric_masks::xgmi_mask, "XGMI", "" },
    { metric_masks::pcie_mask, "PCIe", "" },
} };

}  // namespace gpu
}  // namespace pmc
}  // namespace rocprofsys
