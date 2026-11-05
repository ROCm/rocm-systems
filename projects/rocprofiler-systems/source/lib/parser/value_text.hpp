// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rocprofsys::parser
{
/// Whole-string boolean. true/false, yes/no, on/off, y/n, t/f, and 1/0 match
/// case-insensitively. Any other integer, including an optional leading sign, is
/// true when it is not zero. An empty string is not a boolean.
[[nodiscard]] std::optional<bool>
parse_boolean(std::string_view value);

/// Whole-string base-10 integer. Prefixes and trailing characters are rejected.
[[nodiscard]] std::optional<std::int64_t>
parse_integer(std::string_view value);

/// Whole-string finite number. Prefixes, trailing characters, NaN, and infinity are
/// rejected.
[[nodiscard]] std::optional<double>
parse_number(std::string_view value);

/// True when @p value is text the parameter accepts.
[[nodiscard]] bool
accepts_text(const schema_param& param, std::string_view value);

/// Parsed configuration value. Lists are already split.
using config_value_t =
    std::variant<bool, std::int64_t, double, std::string, std::vector<std::string>>;

/// Parameter id to the value parsed for it.
using typed_values_t = std::map<std::string, config_value_t, std::less<>>;

/// Parse @p text once as @p param. Returns nullopt when the text is not a value
/// the parameter accepts.
[[nodiscard]] std::optional<config_value_t>
parse_value(const schema_param& param, std::string_view text);

/// Canonical text. Booleans are true or false. Integers have no leading plus.
/// Lists are joined with a comma.
[[nodiscard]] std::string
format_value(const config_value_t& value);

}  // namespace rocprofsys::parser
