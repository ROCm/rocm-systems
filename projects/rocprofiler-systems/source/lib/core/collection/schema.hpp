// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::collection
{
// The Perfetto and rocpd outputs name the same metric differently, so both names are
// kept.
struct column_spec
{
    std::string   name;
    std::string   description;
    std::string   unit;
    std::string   pmc_name;
    std::string   perfetto_track_name;
    std::string   rocpd_track_name;
    std::uint32_t category_id = 0;

    std::string block;
    std::string expression;
    bool        is_constant = false;
    bool        is_derived  = false;
};

struct schema
{
    std::string              name;
    std::string              target_arch;
    agent_type               agent  = agent_type::gpu;
    std::uint32_t            device = 0;
    std::vector<column_spec> columns;
};
}  // namespace rocprofsys::collection
