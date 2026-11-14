// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rocprofsys::collection
{
/** How the columns of an array metric are named; each catalog maps it to strings. */
enum class element_naming : std::uint8_t
{
    scalar,
    xcp_engine,
    engine,
    link
};

/** One catalog row: when the metric is on, how to read it, how to name it. Names and
 * descriptions default to category_info(category_id). */
template <typename Enabled, typename Reading>
struct metric_row
{
    /** Whether the metric is on for the effective (enabled and supported) metrics. */
    bool (*enabled)(const Enabled&) = nullptr;
    /** Reads element [outer][inner]; k_no_value when the reading has no value. */
    double (*read)(const Reading&, std::size_t outer, std::size_t inner) = nullptr;
    /** ROCPROFSYS_CATEGORY_* id of the metric. */
    std::uint32_t category_id = 0;
    /** Perfetto label inside the per-catalog track pattern. */
    std::string_view label;
    /** Unit of the values. */
    std::string_view unit;
    /** pmc_info symbol; empty for array rows (the pmc name is used). */
    std::string_view symbol;
    /** Empty: category_info(category_id).name. */
    std::string_view pmc_name;
    /** Empty: category_info(category_id).description. */
    std::string_view description;
    /** Empty: the pmc name. */
    std::string_view rocpd_track;
    /** XCPs, or 1. */
    std::size_t outer = 1;
    /** Engines or links, or 1. */
    std::size_t inner = 1;
    /** How the columns of the elements are named. */
    element_naming naming = element_naming::scalar;
};
}  // namespace rocprofsys::collection
