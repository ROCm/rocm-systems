// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/stack_frame.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/fmt/fmt.h>

#include <string>

namespace rocprofsys::sampling
{

// JSON serialization helpers for stack_frame. Single source of truth for the
// payload shape consumed by trace_cache::backtrace_region_sample and rocpd.
// NFR-P-6: byte/structurally equivalent to the legacy serialization.

inline std::string
make_call_stack_json(stack_frame const& frame)
{
    nlohmann::json j;
    j["name"] = frame.name.empty() ? fmt::format("{:X}", frame.address) : frame.name;
    j["pc"]   = fmt::format("{:X}", frame.address);
    j["file"] = frame.location;
    return j.dump();
}

inline std::string
make_line_info_json(stack_frame const& frame)
{
    nlohmann::json j;
    j["line_address"] = fmt::format("{:X}", frame.line_address);
    j["name"] = frame.name.empty() ? fmt::format("{:X}", frame.address) : frame.name;
    if(!frame.inlines.empty())
    {
        nlohmann::json inlined;
        auto const&    top  = frame.inlines.front();
        inlined["name"]     = top.name;
        inlined["location"] = top.location;
        // The integer line number is embedded in `location` as "file:line".
        // Consumers needing it as an int can parse the suffix on demand.
        j["inlined"] = inlined;
    }
    return j.dump();
}

// Locked extdata schema: { "depth": <int>, "cpu_ns": <int64> }.
inline std::string
make_extdata_json(int depth, int64_t cpu_ns = -1)
{
    nlohmann::json j;
    j["depth"] = depth;
    if(cpu_ns >= 0) j["cpu_ns"] = cpu_ns;
    return j.dump();
}

}  // namespace rocprofsys::sampling
