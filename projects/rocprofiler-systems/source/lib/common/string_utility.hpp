// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace rocprofsys::utility::string
{

inline std::string
to_lower(std::string_view value)
{
    std::string str_copy{ value };

    std::ranges::transform(str_copy, str_copy.begin(), [](unsigned char chr) {
        return static_cast<char>(std::tolower(chr));
    });

    return str_copy;
}
}  // namespace rocprofsys::utility::string
