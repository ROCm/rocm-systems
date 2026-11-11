// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::collection
{
/** One output column: one metric of one device under its Perfetto and rocpd names. */
struct column_spec
{
    /** pmc_info symbol. */
    std::string symbol;
    std::string description;
    std::string unit;
    /** rocpd pmc_info name, unique per agent. */
    std::string pmc_name;
    std::string perfetto_track_name;
    std::string rocpd_track_name;
    /** ROCPROFSYS_CATEGORY_* id; the Perfetto category of the track. */
    std::uint32_t category_id = 0;
    // Hardware counters only; empty/false for the other sources.
    std::string block;
    std::string expression;
    bool        is_constant = false;
    bool        is_derived  = false;

    bool operator==(const column_spec&) const = default;
};

/** The columns one device writes per row. Stored once in the metadata registry. */
struct schema
{
    agent_type agent = agent_type::cpu;
    /** Agent type index used for the rocpd agent lookup. */
    std::uint32_t device = 0;
    /** pmc_info target_arch ("GPU", "CPU", "NIC"). */
    std::string target_arch;
    /** rocpd event category of the samples. */
    std::string              event_category;
    std::vector<column_spec> columns;

    bool operator==(const schema&) const = default;
};
}  // namespace rocprofsys::collection
