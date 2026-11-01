// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace rocprofsys::parser
{
/// Whole-string boolean. true/false, yes/no, on/off, y/n, t/f, and 1/0 match
/// case-insensitively. Any other integer, including an optional leading sign, is
/// true when it is not zero. An empty or whitespace-only string is not a boolean.
[[nodiscard]] std::optional<bool>
parse_boolean(std::string_view value);

/// Whole-string base-10 integer. Prefixes and trailing characters are rejected.
[[nodiscard]] std::optional<std::int64_t>
parse_integer(std::string_view value);

/// Whole-string finite number. Prefixes, trailing characters, NaN, and infinity are
/// rejected.
[[nodiscard]] std::optional<double>
parse_number(std::string_view value);

}  // namespace rocprofsys::parser
