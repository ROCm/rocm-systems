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
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace argparse
{
namespace
{
[[nodiscard]] std::string_view
strip_dashes(std::string_view name) noexcept
{
    while(!name.empty() && name.front() == '-')
        name.remove_prefix(1);
    return name;
}

struct flag_keys
{
    std::string parser_key;  // for parser.get<T>(key) — uses hyphens, e.g. "log-level"
    std::string env_key;  // for processed_environs dedup — uses underscores, "log_level"
};

[[nodiscard]] flag_keys
keys_from(const flag_descriptor& descriptor)
{
    if(descriptor.names.empty())
        throw exception<std::runtime_error>("flag_descriptor has no names");
    auto parser_key = std::string{ strip_dashes(descriptor.names.back()) };
    auto env_key    = parser_key;
    std::replace(env_key.begin(), env_key.end(), '-', '_');
    return { std::move(parser_key), std::move(env_key) };
}

template <typename Container>
[[nodiscard]] Container
to_container(const std::vector<std::string_view>& source)
{
    Container result;
    if constexpr(std::is_same_v<Container, std::vector<std::string>>)
    {
        result.reserve(source.size());
        for(auto value : source)
            result.emplace_back(value);
    }
    else
    {
        for(auto value : source)
            result.emplace(std::string{ value });
    }
    return result;
}

[[nodiscard]] std::string_view
delimiter_for(join_with join) noexcept
{
    switch(join)
    {
        case join_with::space: return " ";
        case join_with::comma: return ",";
        case join_with::colon: return ":";
        case join_with::none: return {};
    }
    return {};
}

[[nodiscard]] std::string
read_value(parser_t& parser, const std::string& key, const flag_descriptor& descriptor)
{
    switch(descriptor.kind)
    {
        case value_kind::flag: return parser.get<bool>(key) ? "true" : "false";
        case value_kind::scalar: return parser.get<std::string>(key);
        case value_kind::scalar_int: return std::to_string(parser.get<int64_t>(key));
        case value_kind::scalar_double: return std::to_string(parser.get<double>(key));
        case value_kind::list:
        {
            // List values must declare a join. `join_with::none` is a caller
            // bug — emit_env's delimiter would be empty and produce concatenated
            // garbage like "abc-def" instead of "abc,def".
            assert(descriptor.join != join_with::none &&
                   "value_kind::list requires an explicit join_with");
            auto values = parser.get<std::vector<std::string>>(key);
            return fmt::format("{}", fmt::join(values, delimiter_for(descriptor.join)));
        }
    }
    return {};
}

void
emit_env(parser_data& data, const flag_descriptor& descriptor, const std::string& value)
{
    auto delim = delimiter_for(descriptor.join);
    if(delim.empty()) delim = ":";  // safe scalar default for env-list joiners
    for(auto env_var : descriptor.env_vars)
        common::update_env(data.env.current, env_var, value, descriptor.mode, delim,
                           data.env.updated, data.env.initial);
}

void
remember_processed(parser_data& data, const std::string& env_key,
                   const flag_descriptor& descriptor)
{
    data.reg.processed_environs.emplace(env_key);
    for(auto alias : descriptor.dedup_keys)
        data.reg.processed_environs.emplace(std::string{ alias });
}

void
apply_count(parser_t::argument& arg, const count_spec& count) noexcept
{
    if(count.exact >= 0) arg.count(count.exact);
    if(count.min >= 0) arg.min_count(count.min);
    if(count.max >= 0) arg.max_count(count.max);
}

void
register_flag(parser_t& parser, parser_data& data, const flag_descriptor& descriptor)
{
    auto keys = keys_from(descriptor);
    if(!data.reg.environ_filter(keys.env_key, data)) return;

    auto& arg =
        parser.add_argument(to_container<std::vector<std::string>>(descriptor.names),
                            std::string{ descriptor.help });

    apply_count(arg, descriptor.count);
    if(!descriptor.dtype.empty()) arg.dtype(std::string{ descriptor.dtype });
    if(!descriptor.choices.empty())
        arg.choices(to_container<std::set<std::string>>(descriptor.choices));
    if(!descriptor.conflicts.empty())
        arg.conflicts(to_container<std::set<std::string>>(descriptor.conflicts));
    if(!descriptor.requires_.empty())
        arg.required(to_container<std::vector<std::string>>(descriptor.requires_));

    arg.action([&data, descriptor,
                parser_key = std::move(keys.parser_key)](parser_t& parser_ref) {
        if(descriptor.custom != nullptr)
        {
            descriptor.custom(parser_ref, data);
            return;
        }
        emit_env(data, descriptor, read_value(parser_ref, parser_key, descriptor));
    });

    remember_processed(data, keys.env_key, descriptor);
}
}  // namespace

void
register_group(parser_t& parser, parser_data& data, const flag_group& group)
{
    parser.start_group(std::string{ group.title }, std::string{ group.subtitle });
    for(const auto& descriptor : group.flags)
        register_flag(parser, data, descriptor);
}

}  // namespace argparse
}  // namespace rocprofsys
