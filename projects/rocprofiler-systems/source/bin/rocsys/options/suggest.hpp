// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/normalize.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::options
{
namespace detail
{
constexpr int k_suggest_limit = 2;

[[nodiscard]] inline int
edit_distance(std::string_view left, std::string_view right)
{
    if(left.size() > right.size() + static_cast<std::size_t>(k_suggest_limit) ||
       right.size() > left.size() + static_cast<std::size_t>(k_suggest_limit))
    {
        return k_suggest_limit + 1;
    }

    std::vector<int> previous(right.size() + 1);
    std::vector<int> current(right.size() + 1);
    for(std::size_t column = 0; column < previous.size(); ++column)
    {
        previous[column] = static_cast<int>(column);
    }
    for(std::size_t row = 1; row <= left.size(); ++row)
    {
        current[0] = static_cast<int>(row);
        for(std::size_t column = 1; column <= right.size(); ++column)
        {
            const int cost  = left[row - 1] == right[column - 1] ? 0 : 1;
            current[column] = std::min({ previous[column] + 1, current[column - 1] + 1,
                                         previous[column - 1] + cost });
        }
        std::swap(previous, current);
    }
    return previous[right.size()];
}

}  // namespace detail

/// @brief Closest choice within two edits, after @ref normalize_choice.
///        A tie, or a choice that already matches, returns nullopt.
[[nodiscard]] inline std::optional<std::string>
suggest_choice(std::string_view token, std::span<const std::string> choices)
{
    const auto  normalized = normalize_choice(token);
    int         best       = detail::k_suggest_limit + 1;
    std::size_t matches    = 0;
    std::string suggestion;
    for(const auto& choice : choices)
    {
        const int distance = detail::edit_distance(normalized, normalize_choice(choice));
        if(distance == 0 || distance > detail::k_suggest_limit)
        {
            continue;
        }
        if(distance < best)
        {
            best       = distance;
            matches    = 1;
            suggestion = choice;
            continue;
        }
        if(distance == best)
        {
            ++matches;
        }
    }
    if(matches != 1)
    {
        return std::nullopt;
    }
    return suggestion;
}

}  // namespace rocprofsys::rocsys::options
