// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <iterator>
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

std::string_view
option_name(const schema_param& param)
{
    return param.arg ? std::string_view{ param.arg->name } : std::string_view{ param.id };
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
        if(token.size() < k_min_option_length || token.front() != '-')
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
            store_value(*param, token.substr(equals_at + 1));
            return true;
        }
        if(param->arg && param->arg->form == arg_form::flag)
        {
            store_value(*param, "true");
            return true;
        }
        return consume_separate_value(*param);
    }

    [[nodiscard]] bool consume_separate_value(const schema_param& param)
    {
        const auto next = m_index + 1;
        if(next >= m_args.size() || m_args[next] == k_option_end)
        {
            m_result.errors.push_back(
                fmt::format("option '{}' requires a value", option_name(param)));
            return true;
        }
        store_value(param, m_args[next]);
        m_index = next;
        return true;
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
            store_value(*param, "true");
        }
        return true;
    }

    void store_value(const schema_param& param, std::string_view value)
    {
        if(!is_valid_value(param, value))
        {
            m_result.errors.push_back(fmt::format("invalid value '{}' for option '{}'",
                                                  value, option_name(param)));
            return;
        }
        m_result.values.insert_or_assign(param.id, std::string{ value });
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
