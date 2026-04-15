// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace sdk_pmc
{

/**
 * @brief A single counter value from the device counting service.
 */
struct counter_value
{
    uint64_t    counter_id_handle = 0;
    std::string name;
    double      value = 0.0;
};

/**
 * @brief Container for SDK PMC metrics collected via device_counting_service.
 *
 * Unlike the GPU collector's fixed metrics struct, SDK PMC counters are dynamic
 * and user-specified. The counter list is determined at runtime from the
 * ROCPROFSYS_SDK_PMC_EVENTS setting.
 */
struct metrics
{
    std::vector<counter_value> counters;
};

/**
 * @brief Tracks which SDK PMC counters are enabled.
 *
 * For SDK PMC, counters are dynamically specified by name rather than a fixed
 * bitfield. The `value` field is non-zero when any counters are enabled,
 * providing compatibility with the base::collector template.
 */
struct enabled_metrics
{
    std::vector<std::string> counter_names;
    uint32_t                 value = 0;
};

}  // namespace sdk_pmc
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
