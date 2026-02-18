// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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
        uint32_t major   = 0;
        uint32_t minor   = 0;
        uint32_t release = 0;
    } numeric_representation;
    std::string string_representation;
};

/**
 * @brief Device selection mode for filtering devices.
 */
enum class device_selection_mode : uint8_t
{
    ALL,      ///< Include all devices
    NONE,     ///< Exclude all devices
    SPECIFIC  ///< Include only specific devices by index
};

/**
 * @brief Device filter configuration.
 */
struct device_filter
{
    device_selection_mode mode = device_selection_mode::ALL;
    std::set<size_t>      indices;  ///< Device indices when mode is SPECIFIC
};

}  // namespace pmc
}  // namespace rocprofsys
