// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "delivery.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <span>
#include <string_view>
#include <utility>

namespace rocprofsys::rocsys
{
namespace
{
constexpr std::string_view k_separator       = "--";
constexpr std::string_view k_env_flag        = "--env";
constexpr std::string_view k_output_short    = "-o";
constexpr std::string_view k_output_long     = "--output";
constexpr std::string_view k_output_long_eq  = "--output=";
constexpr std::string_view k_output_short_eq = "-o=";

[[nodiscard]] std::vector<std::string>
payload_of(const std::vector<std::string>& argv, bool strip_subcommand)
{
    const auto start = strip_subcommand ? std::size_t{ 2 } : std::size_t{ 1 };
    if(argv.size() <= start)
    {
        return {};
    }
    return { argv.begin() + static_cast<std::ptrdiff_t>(start), argv.end() };
}

[[nodiscard]] std::size_t
separator_index(const std::vector<std::string>& payload)
{
    const auto found = std::ranges::find(payload, k_separator);
    return static_cast<std::size_t>(std::distance(payload.begin(), found));
}

[[nodiscard]] bool
is_output_option(std::string_view arg)
{
    return arg == k_output_short || arg == k_output_long ||
           arg.starts_with(k_output_long_eq) || arg.starts_with(k_output_short_eq);
}

// Stop at "--". A token after the separator belongs to the application.
[[nodiscard]] bool
output_option_present(std::span<const std::string> args)
{
    for(const auto& arg : args)
    {
        if(arg.empty() || arg == k_separator)
        {
            return false;
        }
        if(is_output_option(arg))
        {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool
has_embedded(const env_plan& ops)
{
    return std::ranges::any_of(ops, &env_op::embed);
}

[[nodiscard]] env_plan
inherited_ops(const env_plan& ops)
{
    env_plan kept;
    for(const auto& op : ops)
    {
        if(!op.embed)
        {
            kept.push_back(op);
        }
    }
    return kept;
}

void
append_embedded(std::vector<std::string>& argv, const env_plan& ops)
{
    for(const auto& op : ops)
    {
        if(!op.embed)
        {
            continue;
        }
        argv.emplace_back(k_env_flag);
        argv.push_back(op.key + "=" + op.value);
    }
}

[[nodiscard]] diagnostic
embed_not_supported()
{
    return { .error = "this command cannot embed environment variables in a binary",
             .hint  = "profile, instrument, python, and attach inherit the environment" };
}

void
append_implicit_output_flag(std::vector<std::string>&    argv,
                            std::span<const std::string> before_separator,
                            const verb_spec&             verb)
{
    if(!has_capability(verb.caps, capability::embeds_env_in_binary))
    {
        return;
    }
    if(output_option_present(before_separator))
    {
        return;
    }
    argv.emplace_back(k_output_short);
}

}  // namespace

result<launch_plan>
plan(const verb_spec& verb, bool strip_subcommand, const std::vector<std::string>& argv,
     const env_plan& ops)
{
    if(argv.empty())
    {
        return diagnostic{ .error = "missing command arguments" };
    }
    if(has_embedded(ops) && !has_capability(verb.caps, capability::embeds_env_in_binary))
    {
        return embed_not_supported();
    }

    const bool  in_process = has_capability(verb.caps, capability::in_process);
    launch_plan planned{ .in_process = in_process,
                         .program =
                             in_process ? argv.front() : std::string{ verb.binary_name },
                         .env  = inherited_ops(ops),
                         .mode = verb.mode };
    planned.argv.push_back(planned.program);
    append_embedded(planned.argv, ops);

    const auto payload = payload_of(argv, strip_subcommand);
    const auto split   = separator_index(payload);
    const auto before  = std::span<const std::string>{ payload.data(), split };
    planned.argv.insert(planned.argv.end(), before.begin(), before.end());
    append_implicit_output_flag(planned.argv, before, verb);
    planned.argv.insert(planned.argv.end(),
                        payload.begin() + static_cast<std::ptrdiff_t>(split),
                        payload.end());
    return planned;
}

}  // namespace rocprofsys::rocsys
