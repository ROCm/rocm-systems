// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
constexpr std::string_view k_option_end{ "--" };
constexpr std::size_t      k_min_option_length = 2;
constexpr std::size_t      k_min_bundle_length = 3;
constexpr std::size_t      k_single_value      = 1;

struct token_limit
{
    std::size_t minimum = 0;
    std::size_t maximum = 0;
};

std::string_view
option_name(const schema_param& param)
{
    return param.arg ? std::string_view{ param.arg->name } : std::string_view{ param.id };
}

token_limit
limits_for(const schema_param& param)
{
    const bool is_flag  = param.arg && param.arg->form == arg_form::flag;
    const auto fallback = is_flag ? std::size_t{ 0 } : k_single_value;
    const auto minimum  = param.arg ? param.arg->min_count.value_or(fallback) : fallback;
    if(param.type == param_type::enumeration_list && param.arg && !param.arg->max_count)
    {
        return { .minimum = minimum, .maximum = std::numeric_limits<std::size_t>::max() };
    }
    const auto maximum = param.arg ? param.arg->max_count.value_or(minimum) : minimum;
    return { .minimum = minimum, .maximum = maximum };
}

[[nodiscard]] bool
starts_option(std::string_view token)
{
    return token.size() >= k_min_option_length && token.front() == '-';
}

[[nodiscard]] bool
is_signed_number(const schema_param& param, std::string_view token)
{
    if(param.type == param_type::integer)
    {
        return parse_integer(token).has_value();
    }
    if(param.type == param_type::number)
    {
        return parse_number(token).has_value();
    }
    return false;
}

[[nodiscard]] bool
blocks_value(const schema_param& param, std::string_view token)
{
    return starts_option(token) && !is_signed_number(param, token);
}

[[nodiscard]] bool
accepts_list_token(const schema_param& param, std::string_view token)
{
    if(param.type != param_type::enumeration_list)
    {
        return true;
    }
    return is_valid_value(param, canonical_choice(param, token));
}

std::string
join_tokens(const std::vector<std::string>& tokens, std::size_t start, std::size_t count,
            std::string_view separator)
{
    std::string joined;
    const auto  end = start + count;
    for(std::size_t index = start; index < end; ++index)
    {
        if(index != start)
        {
            joined.append(separator);
        }
        joined.append(tokens[index]);
    }
    return joined;
}

std::vector<std::string>
tail_from(const std::vector<std::string>& args, std::size_t index)
{
    if(index >= args.size())
    {
        return {};
    }
    return { std::next(args.begin(), static_cast<std::ptrdiff_t>(index)), args.end() };
}

class argument_cursor
{
public:
    argument_cursor(const schema& config_schema, const std::vector<std::string>& args,
                    unknown_option unknown)
    : m_schema(config_schema)
    , m_args(args)
    , m_unknown(unknown)
    {}

    parse_result run()
    {
        while(m_index < m_args.size())
        {
            parse_next();
        }
        return std::move(m_result);
    }

private:
    void parse_next()
    {
        const auto& token = m_args[m_index];
        if(token == k_option_end)
        {
            finish(m_result.command, m_index + 1);
            return;
        }
        if(!starts_option(token))
        {
            m_result.errors.push_back(fmt::format("unexpected argument '{}'", token));
            ++m_index;
            return;
        }
        if(!consume_option(token))
        {
            reject_unknown(token);
            return;
        }
        ++m_index;
    }

    void reject_unknown(const std::string& token)
    {
        if(m_unknown == unknown_option::stop)
        {
            finish(m_result.unparsed, m_index);
            return;
        }
        m_result.errors.push_back(fmt::format("unknown option '{}'", token));
        ++m_index;
    }

    void finish(std::vector<std::string>& destination, std::size_t index)
    {
        destination = tail_from(m_args, index);
        m_index     = m_args.size();
    }

    [[nodiscard]] bool consume_option(std::string_view token)
    {
        if(!starts_option(token))
        {
            return false;
        }
        const auto  equals_at = token.find('=');
        const auto  name      = token.substr(0, equals_at);
        const auto* param     = m_schema.find_param_by_arg(name);
        if(param == nullptr)
        {
            return store_bundle(token);
        }
        if(equals_at != std::string_view::npos)
        {
            store_target(*param, param->id, token.substr(equals_at + 1));
            return true;
        }
        return consume_counted(*param);
    }

    [[nodiscard]] bool consume_counted(const schema_param& param)
    {
        const auto limit  = limits_for(param);
        const auto tokens = take_tokens(param, limit.maximum);
        if(tokens.size() < limit.minimum)
        {
            report_missing(param);
            return true;
        }
        if(tokens.empty())
        {
            if(!store_bare_flag(param) && limit.minimum > 0)
            {
                report_missing(param);
            }
            return true;
        }
        const auto binds = param.arg.value_or(arg_spec{}).binds;
        if(!binds.empty())
        {
            store_binds(param, binds, tokens);
            return true;
        }
        store_target(param, param.id,
                     join_tokens(tokens, 0, tokens.size(), list_separator(param)));
        return true;
    }

    [[nodiscard]] bool store_bare_flag(const schema_param& param)
    {
        const auto form = param.arg.value_or(arg_spec{}).form;
        if(form != arg_form::flag)
        {
            return false;
        }
        store_target(param, param.id, "true");
        return true;
    }

    [[nodiscard]] std::vector<std::string> take_tokens(const schema_param& param,
                                                       std::size_t         maximum)
    {
        std::vector<std::string> tokens;
        while(tokens.size() < maximum)
        {
            const auto next = m_index + 1;
            if(next >= m_args.size() || blocks_value(param, m_args[next]) ||
               !accepts_list_token(param, m_args[next]))
            {
                break;
            }
            tokens.push_back(m_args[next]);
            m_index = next;
        }
        return tokens;
    }

    void report_missing(const schema_param& param)
    {
        m_result.errors.push_back(
            fmt::format("option '{}' requires a value", option_name(param)));
    }

    void store_binds(const schema_param& option, const std::vector<std::string>& binds,
                     const std::vector<std::string>& tokens)
    {
        std::size_t start = 0;
        for(std::size_t slot = 0; slot < binds.size() && start < tokens.size(); ++slot)
        {
            const bool last  = slot + 1 == binds.size();
            const auto count = last ? tokens.size() - start : k_single_value;
            const auto value = join_tokens(tokens, start, count, list_separator(option));
            store_target(option, binds[slot], value);
            start += count;
        }
    }

    void store_target(const schema_param& option, std::string_view target_id,
                      std::string_view value)
    {
        const auto* target = m_schema.find_param(target_id);
        if(target == nullptr)
        {
            m_result.errors.push_back(
                fmt::format("unknown parameter '{}' for option '{}'", target_id,
                            option_name(option)));
            return;
        }
        m_result.store_if_valid(*target, value);
    }

    [[nodiscard]] bool store_bundle(std::string_view token)
    {
        if(token.size() < k_min_bundle_length || token[1] == '-')
        {
            return false;
        }
        std::vector<const schema_param*> flags;
        flags.reserve(token.size() - 1);
        for(std::size_t index = 1; index < token.size(); ++index)
        {
            const auto  name  = std::string{ '-' } + token[index];
            const auto* param = m_schema.find_param_by_arg(name);
            if(param == nullptr || !param->arg || param->arg->form != arg_form::flag)
            {
                return false;
            }
            flags.push_back(param);
        }
        for(const auto* param : flags)
        {
            store_target(*param, param->id, "true");
        }
        return true;
    }

    const schema&                   m_schema;
    const std::vector<std::string>& m_args;
    unknown_option                  m_unknown;
    std::size_t                     m_index = 0;
    parse_result                    m_result{};
};
}  // namespace

cli_parser::cli_parser(std::vector<std::string> args, unknown_option unknown)
: m_args(std::move(args))
, m_unknown(unknown)
{}

std::vector<std::string>
cli_parser::arguments_from(int argc, char** argv)
{
    std::vector<std::string> args;
    if(argv == nullptr || argc < 2)
    {
        return args;
    }
    args.reserve(static_cast<std::size_t>(argc - 1));
    for(int index = 1; index < argc; ++index)
    {
        if(argv[index] != nullptr)
        {
            args.emplace_back(argv[index]);
        }
    }
    return args;
}

parse_result
cli_parser::parse(const schema& config_schema) const
{
    return argument_cursor{ config_schema, m_args, m_unknown }.run();
}

}  // namespace rocprofsys::parser
