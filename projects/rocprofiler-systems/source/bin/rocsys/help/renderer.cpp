// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "help/renderer.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::help
{
namespace
{
constexpr std::size_t      k_min_name_column = 14;
constexpr std::string_view k_help_short      = "-h";
constexpr std::string_view k_help_alt_short  = "-?";
constexpr std::string_view k_help_long       = "--help";
constexpr std::string_view k_help_long_eq    = "--help=";

[[nodiscard]] bool
is_common(const options::option_spec& spec) noexcept
{
    return spec.required_caps == 0;
}

[[nodiscard]] bool
applies_to(const options::option_spec& spec, capability_set caps) noexcept
{
    return (caps & spec.required_caps) == spec.required_caps;
}

[[nodiscard]] bool
is_help_token(std::string_view token) noexcept
{
    return token == k_help_short || token == k_help_alt_short || token == k_help_long ||
           token.starts_with(k_help_long_eq);
}

[[nodiscard]] std::size_t
name_column(std::span<const options::option_spec* const> rows) noexcept
{
    auto width = k_min_name_column;
    for(const auto* spec : rows)
    {
        // "--" plus one space before the summary.
        width = std::max(width, spec->name.size() + 3);
    }
    return width;
}

void
write_rows(std::ostream& out, std::string_view heading,
           std::span<const options::option_spec* const> rows)
{
    out << heading << '\n';
    const auto column = name_column(rows);
    for(const auto* spec : rows)
    {
        const auto token = fmt::format("--{}", spec->name);
        out << fmt::format("  {:<{}}{}\n", token, column, spec->summary);
    }
}

[[nodiscard]] std::vector<const options::option_spec*>
matching_rows(std::span<const options::option_spec> table, capability_set caps,
              bool common_only)
{
    std::vector<const options::option_spec*> rows;
    for(const auto& spec : table)
    {
        const bool selected = common_only ? is_common(spec) : applies_to(spec, caps);
        if(selected)
        {
            rows.push_back(&spec);
        }
    }
    return rows;
}

void
write_rewrite_follow_up(std::ostream& out, const verb_spec& verb)
{
    if(!has_capability(verb.caps, capability::embeds_env_in_binary))
    {
        return;
    }
    out << "\nA rewritten binary keeps rocsys settings as defaults. Profile it with:\n"
           "  rocsys -- ./app.inst\n";
}

}  // namespace

void
write_common_options(std::ostream& out, std::span<const options::option_spec> table)
{
    const auto rows = matching_rows(table, {}, true);
    write_rows(out, "Common options:", rows);
}

void
write_verb_options(std::ostream& out, const verb_spec& verb,
                   std::span<const options::option_spec> table)
{
    const auto rows = matching_rows(table, verb.caps, false);
    write_rows(out, fmt::format("rocsys options for {}:", verb.name), rows);
    write_rewrite_follow_up(out, verb);
}

bool
requests_tool_help(std::span<const std::string> passthrough)
{
    return std::ranges::any_of(passthrough, is_help_token);
}

}  // namespace rocprofsys::rocsys::help
