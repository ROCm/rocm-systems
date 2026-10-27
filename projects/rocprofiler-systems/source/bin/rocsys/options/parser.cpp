// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "options/parser.hpp"

#include "options/suggest.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <optional>
#include <string_view>
#include <utility>

namespace rocprofsys::rocsys::options
{
namespace
{
constexpr std::string_view k_separator = "--";

[[nodiscard]] bool
is_flag_token(std::string_view token)
{
    return !token.empty() && token.front() == '-';
}

[[nodiscard]] std::string_view
trim_spaces(std::string_view token)
{
    while(!token.empty() && token.front() == ' ')
    {
        token.remove_prefix(1);
    }
    while(!token.empty() && token.back() == ' ')
    {
        token.remove_suffix(1);
    }
    return token;
}

[[nodiscard]] std::vector<std::string>
split_commas(std::string_view token)
{
    std::vector<std::string> pieces;
    while(true)
    {
        const auto comma = token.find(',');
        if(comma == std::string_view::npos)
        {
            pieces.emplace_back(trim_spaces(token));
            break;
        }
        pieces.emplace_back(trim_spaces(token.substr(0, comma)));
        token.remove_prefix(comma + 1);
    }
    return pieces;
}

struct long_option
{
    std::string_view                name = {};
    std::optional<std::string_view> inline_value;
};

[[nodiscard]] std::optional<long_option>
match_long_option(std::string_view token)
{
    constexpr std::string_view k_prefix = "--";
    if(!token.starts_with(k_prefix) || token.size() <= k_prefix.size())
    {
        return std::nullopt;
    }
    const auto body  = token.substr(k_prefix.size());
    const auto equal = body.find('=');
    if(equal == std::string_view::npos)
    {
        return long_option{ body, std::nullopt };
    }
    return long_option{ body.substr(0, equal), body.substr(equal + 1) };
}

[[nodiscard]] const option_spec*
find_option(std::span<const option_spec> options, std::string_view name)
{
    for(const auto& spec : options)
    {
        if(spec.name == name)
        {
            return &spec;
        }
    }
    return nullptr;
}

[[nodiscard]] diagnostic
unknown_value(const option_spec& spec, std::string_view token, const context& ctx)
{
    auto hint = std::string{ "missing '--' before the application?" };
    if(spec.choices != nullptr)
    {
        const auto choices = spec.choices(ctx);
        if(const auto guess = suggest_choice(token, choices))
        {
            hint = fmt::format("did you mean '{}'? {}", *guess, hint);
        }
    }
    return { .error = fmt::format("unknown value '{}' for --{}", token, spec.name),
             .hint  = std::move(hint) };
}

[[nodiscard]] diagnostic
missing_value(const option_spec& spec)
{
    return { .error = fmt::format("missing value for --{}", spec.name) };
}

[[nodiscard]] diagnostic
flag_does_not_take_a_value(const option_spec& spec)
{
    return { .error = fmt::format("--{} does not take a value", spec.name) };
}

void
merge_option(parsed_command& command, const option_spec& spec,
             std::vector<std::string> values)
{
    for(auto& existing : command.options)
    {
        if(existing.spec == &spec)
        {
            existing.values.insert(existing.values.end(),
                                   std::make_move_iterator(values.begin()),
                                   std::make_move_iterator(values.end()));
            return;
        }
    }
    command.options.push_back(parsed_option{ &spec, std::move(values) });
}

class token_parser
{
public:
    explicit token_parser(const parse_request& request)
    : m_request(request)
    {}

    [[nodiscard]] result<parsed_command> run()
    {
        if(m_request.verb == nullptr || m_request.ctx == nullptr)
        {
            return diagnostic{ .error = "missing command arguments" };
        }
        m_command.verb = m_request.verb;
        split_payload();
        while(m_index < m_head.size())
        {
            if(auto error = consume_one())
            {
                return *error;
            }
        }
        return std::move(m_command);
    }

private:
    void split_payload()
    {
        const auto start =
            m_request.strip_subcommand ? std::size_t{ 2 } : std::size_t{ 1 };
        if(m_request.argv.size() <= start)
        {
            return;
        }
        const auto payload = m_request.argv.subspan(start);
        const auto found   = std::ranges::find(payload, std::string{ k_separator });
        const auto split   = static_cast<std::size_t>(found - payload.begin());
        m_head.assign(payload.begin(),
                      payload.begin() + static_cast<std::ptrdiff_t>(split));
        m_command.app.assign(payload.begin() + static_cast<std::ptrdiff_t>(split),
                             payload.end());
    }

    [[nodiscard]] std::optional<diagnostic> consume_one()
    {
        const auto token   = std::string_view{ m_head[m_index] };
        const auto matched = match_long_option(token);
        if(!matched)
        {
            m_command.passthrough.push_back(m_head[m_index]);
            ++m_index;
            return std::nullopt;
        }
        const auto* spec = find_option(m_request.options, matched->name);
        if(spec == nullptr)
        {
            m_command.passthrough.push_back(m_head[m_index]);
            ++m_index;
            return std::nullopt;
        }
        ++m_index;
        return consume_option(*spec, *matched);
    }

    [[nodiscard]] std::optional<diagnostic> consume_option(const option_spec& spec,
                                                           const long_option& matched)
    {
        switch(spec.kind)
        {
            case value_kind::flag: return consume_flag(spec, matched);
            case value_kind::single: return consume_single(spec, matched);
            case value_kind::token_list: return consume_list(spec, matched);
        }
        return diagnostic{ .error = fmt::format("unsupported option --{}", spec.name) };
    }

    [[nodiscard]] std::optional<diagnostic> consume_flag(const option_spec& spec,
                                                         const long_option& matched)
    {
        if(matched.inline_value.has_value())
        {
            return flag_does_not_take_a_value(spec);
        }
        merge_option(m_command, spec, {});
        return std::nullopt;
    }

    [[nodiscard]] std::optional<diagnostic> consume_single(const option_spec& spec,
                                                           const long_option& matched)
    {
        std::string value;
        if(matched.inline_value.has_value())
        {
            value = std::string{ *matched.inline_value };
        }
        else if(m_index < m_head.size() && !is_flag_token(m_head[m_index]))
        {
            value = m_head[m_index];
            ++m_index;
        }
        else
        {
            return missing_value(spec);
        }
        if(auto error = reject_unless_accepted(spec, value))
        {
            return error;
        }
        merge_option(m_command, spec, { std::move(value) });
        return std::nullopt;
    }

    [[nodiscard]] std::optional<diagnostic> consume_list(const option_spec& spec,
                                                         const long_option& matched)
    {
        std::vector<std::string> values;
        if(matched.inline_value.has_value())
        {
            if(auto error = append_pieces(spec, *matched.inline_value, values))
            {
                return error;
            }
        }
        while(m_index < m_head.size() && !is_flag_token(m_head[m_index]))
        {
            if(auto error = append_pieces(spec, m_head[m_index], values))
            {
                return error;
            }
            ++m_index;
        }
        merge_option(m_command, spec, std::move(values));
        return std::nullopt;
    }

    [[nodiscard]] std::optional<diagnostic> append_pieces(
        const option_spec& spec, std::string_view token, std::vector<std::string>& values)
    {
        for(auto& piece : split_commas(token))
        {
            if(piece.empty())
            {
                return diagnostic{ .error =
                                       fmt::format("empty value in --{}", spec.name) };
            }
            if(auto error = reject_unless_accepted(spec, piece))
            {
                return error;
            }
            values.push_back(std::move(piece));
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<diagnostic> reject_unless_accepted(
        const option_spec& spec, std::string_view token)
    {
        if(spec.accepts == nullptr || spec.accepts(token, *m_request.ctx))
        {
            return std::nullopt;
        }
        return unknown_value(spec, token, *m_request.ctx);
    }

    const parse_request&     m_request;
    parsed_command           m_command;
    std::vector<std::string> m_head;
    std::size_t              m_index = 0;
};

}  // namespace

result<parsed_command>
parse(const parse_request& request)
{
    return token_parser{ request }.run();
}

std::vector<std::string>
command_argv(std::string_view argv0, const parsed_command& command)
{
    std::vector<std::string> argv;
    argv.reserve(1 + command.passthrough.size() + command.app.size());
    argv.emplace_back(argv0);
    argv.insert(argv.end(), command.passthrough.begin(), command.passthrough.end());
    argv.insert(argv.end(), command.app.begin(), command.app.end());
    return argv;
}

}  // namespace rocprofsys::rocsys::options
