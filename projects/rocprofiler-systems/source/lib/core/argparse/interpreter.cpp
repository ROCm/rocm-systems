// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Interim engine: the interpreter dispatches descriptors onto
// tim::argparse::argument_parser. The descriptor table itself is
// engine-agnostic; replacing tim::argparse with another parser library
// (e.g. CLI11) is a focused change confined to this translation unit.

#include "interpreter.hpp"
#include "exception.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace argparse
{
namespace
{
std::string_view
strip_dashes(std::string_view name) noexcept
{
    while(!name.empty() && name.front() == '-')
        name.remove_prefix(1);
    return name;
}

std::string
key_from(const flag_descriptor& descriptor)
{
    if(descriptor.names.empty())
        throw exception<std::runtime_error>("flag_descriptor has no names");
    return std::string{ strip_dashes(descriptor.names.back()) };
}

std::vector<std::string>
to_strvec(const std::vector<std::string_view>& source)
{
    std::vector<std::string> result;
    result.reserve(source.size());
    std::transform(source.begin(), source.end(), std::back_inserter(result),
                   [](std::string_view value) { return std::string{ value }; });
    return result;
}

std::set<std::string>
to_strset(const std::vector<std::string_view>& source)
{
    std::set<std::string> result;
    for(auto value : source)
        result.emplace(value);
    return result;
}

const char*
delimiter_for(join_with join) noexcept
{
    switch(join)
    {
        case join_with::space: return " ";
        case join_with::comma: return ",";
        case join_with::colon: return ":";
        case join_with::none:
        default: return " ";
    }
}

std::string
read_value(parser_t& parser, const std::string& key, const flag_descriptor& descriptor)
{
    switch(descriptor.kind)
    {
        case value_kind::flag: return parser.get<bool>(key) ? "true" : "false";

        case value_kind::scalar: return parser.get<std::string>(key);

        case value_kind::list:
        {
            auto values = parser.get<std::vector<std::string>>(key);
            return fmt::format("{}", fmt::join(values, delimiter_for(descriptor.join)));
        }
    }
    return {};
}

void
emit_env(parser_data& data, const flag_descriptor& descriptor, const std::string& value)
{
    for(auto env_var : descriptor.env_vars)
    {
        common::update_env(data.env.current, env_var, value, descriptor.mode,
                           delimiter_for(descriptor.join), data.env.updated,
                           data.env.initial);
    }
}

void
remember_processed(parser_data& data, const std::string& key,
                   const flag_descriptor& descriptor)
{
    data.reg.processed_environs.emplace(key);
    for(auto alias : descriptor.aliased_env)
        data.reg.processed_environs.emplace(std::string{ alias });
}

void
apply_count(parser_t::argument& arg, const count_spec& count) noexcept
{
    if(count.exact >= 0) arg.count(count.exact);
    if(count.min >= 0) arg.min_count(count.min);
    if(count.max >= 0) arg.max_count(count.max);
}
}  // namespace

void
register_flag(parser_t& parser, parser_data& data, const flag_descriptor& descriptor)
{
    auto key = key_from(descriptor);
    if(!data.reg.environ_filter(key, data)) return;

    auto& arg =
        parser.add_argument(to_strvec(descriptor.names), std::string{ descriptor.help });

    apply_count(arg, descriptor.count);
    if(!descriptor.dtype.empty()) arg.dtype(std::string{ descriptor.dtype });
    if(!descriptor.choices.empty()) arg.choices(to_strset(descriptor.choices));
    if(!descriptor.conflicts.empty()) arg.conflicts(to_strset(descriptor.conflicts));
    if(!descriptor.requires_.empty()) arg.required(to_strvec(descriptor.requires_));

    arg.action([&data, descriptor, key](parser_t& parser_ref) {
        if(descriptor.custom != nullptr)
        {
            descriptor.custom(parser_ref, data);
            return;
        }
        emit_env(data, descriptor, read_value(parser_ref, key, descriptor));
    });

    remember_processed(data, key, descriptor);
}

void
register_group(parser_t& parser, parser_data& data, const flag_group& group)
{
    parser.start_group(std::string{ group.title }, std::string{ group.subtitle });
    for(const auto& descriptor : group.flags)
        register_flag(parser, data, descriptor);
}

}  // namespace argparse
}  // namespace rocprofsys
