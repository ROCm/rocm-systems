// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/cli_parser.hpp"
#include "parser/parse_result.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"
#include "parser/value_text.hpp"

#include "common/string_utility.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
    return param.arg ? std::string_view{ *param.arg } : std::string_view{ param.id };
}

token_limit
limits_for(const schema_param& param)
{
    const auto fallback = param.flag ? std::size_t{ 0 } : k_single_value;
    const auto minimum  = param.arg_min_count.value_or(fallback);
    if(param.type == param_type::enumeration_list && param.arg && !param.arg_max_count)
    {
        return { .minimum = minimum, .maximum = std::numeric_limits<std::size_t>::max() };
    }
    const auto maximum = param.arg_max_count.value_or(minimum);
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

/// The tokens of one list option as one list value. A token boundary separates two
/// names, and a separator already at the boundary is not doubled, so "hip," "mpi" and
/// "hip" "," "mpi" are both hip,mpi, while "," alone stays a value that names nothing.
std::string
list_text(const std::vector<std::string>& tokens, std::string_view separator)
{
    std::string text;
    for(const auto& token : tokens)
    {
        const auto part = utility::string::trim(token);
        if(part.empty())
        {
            continue;
        }
        if(!text.empty() && !text.ends_with(separator) && !part.starts_with(separator))
        {
            text.append(separator);
        }
        text.append(part);
    }
    return text;
}

/// The part of a refused list @p text to name: the first element that is not a valid
/// choice, or the whole text when the list itself is malformed, as in "a,,b".
std::string
refused_part(const schema_param& param, const std::string& text)
{
    const auto parts = split_list(text, list_separator(param));
    if(parts)
    {
        const auto found =
            std::ranges::find_if(*parts, [&param](const std::string& part) {
                return !part.empty() && !is_valid_value(param, part);
            });
        if(found != parts->end())
        {
            return *found;
        }
    }
    return text;
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
            if(m_unknown == unknown_option::forward)
            {
                forward(token);
                return;
            }
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
        if(m_unknown == unknown_option::forward)
        {
            forward(token);
            return;
        }
        m_result.errors.push_back(fmt::format("unknown option '{}'", token));
        ++m_index;
    }

    void forward(const std::string& token)
    {
        m_result.unparsed.push_back(token);
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
            store_tokens(*param, { std::string{ token.substr(equals_at + 1) } });
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
        store_tokens(param, tokens);
        return true;
    }

    /// A list reads its tokens as one value and adds its elements to the ones given
    /// earlier; nothing more is stored when that value is refused. Any other option
    /// stores its tokens joined, and a repeat replaces the earlier value.
    void store_tokens(const schema_param& param, const std::vector<std::string>& tokens)
    {
        const auto separator = list_separator(param);
        if(param.type != param_type::enumeration_list)
        {
            m_result.store_if_valid(param,
                                    join_tokens(tokens, 0, tokens.size(), separator));
            return;
        }
        const auto text   = list_text(tokens, separator);
        const auto parsed = parse_value(param, text);
        if(!parsed)
        {
            m_result.errors.push_back(
                invalid_value_message(refused_part(param, text), param.id));
            return;
        }
        auto        elements = earlier_elements(param);
        const auto& parts    = std::get<std::vector<std::string>>(*parsed);
        elements.insert(elements.end(), parts.begin(), parts.end());
        m_result.store_typed(param, std::move(elements));
    }

    [[nodiscard]] std::vector<std::string> earlier_elements(
        const schema_param& param) const
    {
        const auto found = m_result.values.find(param.id);
        if(found == m_result.values.end())
        {
            return {};
        }
        const auto* list = std::get_if<std::vector<std::string>>(&found->second);
        return list == nullptr ? std::vector<std::string>{} : *list;
    }

    [[nodiscard]] bool store_bare_flag(const schema_param& param)
    {
        if(!param.flag)
        {
            return false;
        }
        m_result.store_if_valid(param, "true");
        return true;
    }

    [[nodiscard]] std::vector<std::string> take_tokens(const schema_param& param,
                                                       std::size_t         maximum)
    {
        std::vector<std::string> tokens;
        while(tokens.size() < maximum)
        {
            const auto next = m_index + 1;
            if(next >= m_args.size() || blocks_value(param, m_args[next]))
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
            if(param == nullptr || !param->flag)
            {
                return false;
            }
            flags.push_back(param);
        }
        for(const auto* param : flags)
        {
            m_result.store_if_valid(*param, "true");
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
