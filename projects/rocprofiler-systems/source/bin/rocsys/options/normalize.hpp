// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace rocprofsys::rocsys::options
{

/// @brief Lowercase @p token and map both '-' and '_' to '_'.
[[nodiscard]] inline std::string
normalize_choice(std::string_view token)
{
    std::string normalized;
    normalized.reserve(token.size());
    for(unsigned char character : token)
    {
        if(character == '-' || character == '_')
        {
            normalized.push_back('_');
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(character)));
    }
    return normalized;
}

/// @brief True when @p lhs and @p rhs are the same choice under @ref normalize_choice.
[[nodiscard]] inline bool
same_choice(std::string_view lhs, std::string_view rhs)
{
    return normalize_choice(lhs) == normalize_choice(rhs);
}

}  // namespace rocprofsys::rocsys::options
