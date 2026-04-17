// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/**
 * @brief Position of a counter instance within a single dimension.
 */
struct dimension_position
{
    std::string name;      ///< Abbreviated dimension name (e.g. "WGP", "SA", "SE")
    size_t      position;  ///< Zero-based position within this dimension
};

/**
 * @brief Abbreviate a rocprofiler dimension name for display.
 *
 * Strips the "DIMENSION_" prefix and applies short aliases:
 *   SHADER_ENGINE → SE, SHADER_ARRAY → SA, INSTANCE → INST
 */
inline std::string
abbreviate_dimension_name(std::string_view dim_name)
{
    constexpr std::string_view prefix = "DIMENSION_";
    if(dim_name.substr(0, prefix.size()) == prefix) dim_name.remove_prefix(prefix.size());

    if(dim_name == "SHADER_ENGINE") return "SE";
    if(dim_name == "SHADER_ARRAY") return "SA";
    if(dim_name == "INSTANCE") return "INST";
    return std::string{ dim_name };
}

/**
 * @brief Build a qualified counter name with dimension positions.
 *
 * Returns e.g. "SQC_ICACHE_HITS[WGP=0,SA=0,SE=0]" for multi-dim counters,
 * or just the base name for scalar counters with no dimensions.
 */
inline std::string
make_qualified_name(const std::string&                     base_name,
                    const std::vector<dimension_position>& dims)
{
    if(dims.empty()) return base_name;
    std::string result = base_name + "[";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i > 0) result += ",";
        result += dims[i].name + "=" + std::to_string(dims[i].position);
    }
    result += "]";
    return result;
}

/**
 * @brief A single counter value from the device counting service.
 */
struct counter_value
{
    uint64_t    counter_id_handle = 0;
    std::string name;  ///< Qualified name including dimensions
    double      value = 0.0;
};

/**
 * @brief Container for SDK PMC metrics collected via device_counting_service.
 *
 * Unlike the GPU collector's fixed metrics struct, SDK PMC counters are dynamic
 * and user-specified. The counter list is determined at runtime from the
 * ROCPROFSYS_ROCM_EVENTS setting.
 */
struct metrics
{
    std::vector<counter_value> counters;
};

/**
 * @brief Per-counter capability info for SDK PMC.
 *
 * Derived from rocprofiler_counter_info_v1_t during device initialization,
 * exposing whether a counter is derived or constant.
 */
struct counter_capability
{
    std::string name;
    bool        is_derived  = false;
    bool        is_constant = false;
};

/**
 * @brief Tracks which SDK PMC counters are enabled.
 *
 * For SDK PMC, counters are dynamically specified by name rather than a fixed
 * bitfield. The `value` field is non-zero when any counters are enabled,
 * providing compatibility with the base::collector template. The `capabilities`
 * vector carries per-counter metadata from the SDK.
 */
struct enabled_metrics
{
    std::vector<std::string>        counter_names;
    std::vector<counter_capability> capabilities;
    uint32_t                        value = 0;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
