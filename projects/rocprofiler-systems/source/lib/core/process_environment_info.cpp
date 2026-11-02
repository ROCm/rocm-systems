// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "process_environment_info.hpp"

#include "common/environment.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <span>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace
{
constexpr char             k_assignment     = '=';
constexpr char             k_path_separator = '/';
constexpr std::string_view k_home_alias     = "~";

// Single reviewable source of truth for which variables may be stored in a shared
// database. Everything not listed here is dropped, never filtered by keyword.
constexpr auto k_allowed_prefixes = std::to_array<std::string_view>({
    "ROCPROFSYS_",
    "LD_",
    "ROCM_",
});

constexpr auto k_allowed_names = std::to_array<std::string_view>({
    "PATH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "SLURM_PROCID",
    "SLURM_JOB_ID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
});

[[nodiscard]] constexpr bool
is_allowed(std::string_view name) noexcept
{
    const auto has_prefix = [name](std::string_view prefix) {
        return name.starts_with(prefix);
    };
    return std::ranges::find(k_allowed_names, name) != k_allowed_names.end() ||
           std::ranges::any_of(k_allowed_prefixes, has_prefix);
}

// A name character continues a path component, so a home directory that is followed or
// preceded by one is a different path ("/home/bobby", "/mnt/home/bob").
[[nodiscard]] constexpr bool
is_name_char(char character) noexcept
{
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.';
}

[[nodiscard]] constexpr bool
is_whole_component(std::string_view value, std::size_t pos, std::size_t length) noexcept
{
    const auto end = pos + length;
    return (pos == 0 || !is_name_char(value[pos - 1])) &&
           (end == value.size() || !is_name_char(value[end]));
}

// Replaces every occurrence of @p home that is a whole path component with "~". An empty
// @p home matches everywhere and is never replaced.
[[nodiscard]] std::string
redact_home(std::string_view value, std::string_view home)
{
    if(home.empty())
    {
        return std::string{ value };
    }

    std::string redacted;
    std::size_t copied = 0;  // value[0, copied) is already in redacted
    for(auto hit = value.find(home); hit != std::string_view::npos;
        hit      = value.find(home, hit + 1))
    {
        if(hit < copied || !is_whole_component(value, hit, home.size()))
        {
            continue;
        }
        redacted.append(value.substr(copied, hit - copied)).append(k_home_alias);
        copied = hit + home.size();
    }
    return redacted.append(value.substr(copied));
}
}  // namespace

nlohmann::json
filter_environment(std::span<const char* const> entries, std::string_view home)
{
    // Unset or "/" must not be redacted: it would rewrite every path.
    while(home.ends_with(k_path_separator))
    {
        home.remove_suffix(1);
    }

    auto environment = nlohmann::json::object();
    for(const std::string_view entry : entries)
    {
        const auto assignment = entry.find(k_assignment);
        if(assignment == std::string_view::npos || assignment == 0)
        {
            continue;
        }

        const auto name = entry.substr(0, assignment);
        if(!is_allowed(name))
        {
            continue;
        }

        environment[std::string{ name }] =
            redact_home(entry.substr(assignment + 1), home);
    }
    return environment;
}

nlohmann::json
capture_process_environment()
{
    if(environ == nullptr)
    {
        return nlohmann::json::object();
    }
    const auto end = std::ranges::find(environ, std::unreachable_sentinel, nullptr);
    return filter_environment(std::span<const char* const>{ environ, end },
                              get_env<std::string>("HOME", ""));
}

std::string
dump_environment(const nlohmann::json& environment)
{
    // Environment values are arbitrary bytes; the default strict handler would throw on
    // invalid UTF-8, and this runs during process start-up.
    return environment.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}
}  // namespace rocprofsys
