// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace rocprofsys
{

struct domain_selection
{
    std::optional<std::string>              name;
    std::optional<std::string>              group;
    std::optional<std::vector<std::string>> operations;
};

}  // namespace rocprofsys
