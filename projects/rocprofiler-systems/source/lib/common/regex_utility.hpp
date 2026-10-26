// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <regex>
#include <string>

namespace rocprofsys::utility
{
/// @brief Check whether any regex in @p regexes matches (regex_search) @p name.
/// @tparam RangeT A range of std::regex.
template <typename RangeT>
[[nodiscard]] inline bool
regex_match_any(const std::string& name, const RangeT& regexes)
{
    return std::ranges::any_of(
        regexes, [&name](const std::regex& re) { return std::regex_search(name, re); });
}
}  // namespace rocprofsys::utility
