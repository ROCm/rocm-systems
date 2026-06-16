// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk
{

using counter_id_t = std::uint64_t;

struct dimension_position
{
    std::string name;
    size_t      position{ 0 };
};

struct counter_metadata
{
    counter_id_t                    counter_id{ 0 };
    std::string                     name;
    std::string                     description;
    std::string                     block;
    std::string                     expression;
    bool                            is_constant = false;
    bool                            is_derived  = false;
    std::vector<dimension_position> dimensions;
};

}  // namespace rocprofsys::backends::rocprofiler_sdk
