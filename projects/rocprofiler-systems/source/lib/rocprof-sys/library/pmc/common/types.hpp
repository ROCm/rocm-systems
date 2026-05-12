// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <set>
#include <string>

namespace rocprofsys
{
namespace pmc
{

/**
 * @brief Version information for device providers and drivers.
 */
struct version
{
    struct
    {
        std::uint32_t major   = 0;
        std::uint32_t minor   = 0;
        std::uint32_t release = 0;
    } numeric_representation;
    std::string string_representation;
};

/**
 * @brief Device type for provider enumeration.
 */
enum class device_type : std::uint8_t
{
    GPU,  ///< GPU device
    NIC   ///< Network interface device
};

/**
 * @brief Device selection mode for filtering devices.
 */
enum class device_selection_mode : std::uint8_t
{
    ALL,      ///< Include all devices
    NONE,     ///< Exclude all devices
    SPECIFIC  ///< Include only specific devices by index
};

/**
 * @brief Device filter configuration (index-based, for GPUs).
 */
struct device_filter
{
    device_selection_mode mode = device_selection_mode::ALL;
    std::set<size_t>      indices;  ///< Device indices when mode is SPECIFIC
};

/**
 * @brief NIC device filter configuration (name-based).
 *
 * NICs are filtered by network device name (e.g., "enp226s0", "eth0")
 * rather than index, since NIC indices are not as stable or meaningful.
 */
struct nic_device_filter
{
    device_selection_mode mode = device_selection_mode::ALL;
    std::set<std::string> names;  ///< Device names when mode is SPECIFIC
};

}  // namespace pmc
}  // namespace rocprofsys
