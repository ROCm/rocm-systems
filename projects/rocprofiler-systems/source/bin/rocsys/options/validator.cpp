// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "options/validator.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::rocsys::options
{
namespace
{
[[nodiscard]] bool
starts_with_flag(std::string_view token)
{
    return !token.empty() && token.front() == '-';
}

[[nodiscard]] std::optional<diagnostic>
check_capability(const parsed_command& command, const parsed_option& option)
{
    const auto required = option.spec->required_caps;
    if(command.verb == nullptr || (command.verb->caps & required) == required)
    {
        return std::nullopt;
    }
    return diagnostic{ .error = fmt::format("--{} is not available for '{}'",
                                            option.spec->name, command.verb->name),
                       .hint  = "this command does not support that option" };
}

[[nodiscard]] std::optional<diagnostic>
check_min_values(const parsed_option& option)
{
    if(option.values.size() >= option.spec->min_values)
    {
        return std::nullopt;
    }
    return diagnostic{ .error =
                           fmt::format("missing value for --{}", option.spec->name) };
}

[[nodiscard]] bool
spelling_matches(std::string_view token, std::string_view spelling)
{
    if(token == spelling)
    {
        return true;
    }
    return token.size() > spelling.size() && token.starts_with(spelling) &&
           token[spelling.size()] == '=';
}

[[nodiscard]] std::string_view
attached_value(std::string_view token, std::string_view spelling)
{
    const auto prefix = spelling.size() + 1;
    if(token.size() <= prefix || token[spelling.size()] != '=')
    {
        return {};
    }
    return token.substr(prefix);
}

[[nodiscard]] std::optional<diagnostic>
conflict_on_token(const option_spec& spec, const legacy_conflict& conflict,
                  std::string_view token, std::string_view following, bool has_following)
{
    for(const auto spelling : conflict.spellings)
    {
        if(!spelling_matches(token, spelling))
        {
            continue;
        }
        const auto value = token == spelling && has_following
                               ? following
                               : attached_value(token, spelling);
        if(conflict.matches_value != nullptr && !conflict.matches_value(value))
        {
            continue;
        }
        return diagnostic{ .error =
                               fmt::format("{} conflicts with --{}", spelling, spec.name),
                           .hint = std::string{ conflict.reason } };
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<diagnostic>
check_conflicts(const parsed_command& command, const parsed_option& option)
{
    if(command.verb == nullptr)
    {
        return std::nullopt;
    }
    for(const auto& conflict : option.spec->conflicts)
    {
        if(conflict.binary != command.verb->binary_name)
        {
            continue;
        }
        for(std::size_t index = 0; index < command.passthrough.size(); ++index)
        {
            const bool has_following = index + 1 < command.passthrough.size() &&
                                       !starts_with_flag(command.passthrough[index + 1]);
            const auto following =
                has_following ? std::string_view{ command.passthrough[index + 1] }
                              : std::string_view{};
            if(auto error =
                   conflict_on_token(*option.spec, conflict, command.passthrough[index],
                                     following, has_following))
            {
                return error;
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] bool
same_op(const env_op& lhs, const env_op& rhs)
{
    return lhs.value == rhs.value && lhs.mode == rhs.mode && lhs.embed == rhs.embed;
}

[[nodiscard]] result<env_plan>
fold_same_key(env_plan ops)
{
    env_plan folded;
    for(auto& op : ops)
    {
        const auto found = std::ranges::find(folded, op.key, &env_op::key);
        if(found == folded.end())
        {
            folded.push_back(std::move(op));
            continue;
        }
        if(!same_op(*found, op))
        {
            return diagnostic{
                .error = fmt::format("{} and {} both set {}", found->origin, op.origin,
                                     op.key),
                .hint  = fmt::format("use only one option for {}", op.key),
            };
        }
    }
    return folded;
}

}  // namespace

result<parsed_command>
validate(parsed_command command)
{
    for(const auto& option : command.options)
    {
        if(option.spec == nullptr)
        {
            return diagnostic{ .error = "missing option specification" };
        }
        if(auto error = check_capability(command, option))
        {
            return *error;
        }
        if(auto error = check_min_values(option))
        {
            return *error;
        }
        if(auto error = check_conflicts(command, option))
        {
            return *error;
        }
    }
    return command;
}

result<env_plan>
resolve(const parsed_command& command, const context& ctx)
{
    if(command.verb == nullptr)
    {
        return diagnostic{ .error = "missing command" };
    }
    env_plan plan;
    for(const auto& option : command.options)
    {
        if(option.spec == nullptr || option.spec->resolve == nullptr)
        {
            continue;
        }
        if(auto error = option.spec->resolve(option.values, *command.verb, ctx, plan))
        {
            return *error;
        }
    }
    return fold_same_key(std::move(plan));
}

}  // namespace rocprofsys::rocsys::options
