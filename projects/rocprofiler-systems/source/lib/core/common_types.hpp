// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rocprofsys
{
/// Field delimiter for the serialized function/region argument wire format:
///   <arg_number>;;<arg_type>;;<arg_name>;;<arg_value>;;
/// Single source of truth shared by the producer (get_args_string) and the
/// consumer (process_arguments_string) below.
inline constexpr std::string_view ARG_DELIMITER = ";;";

/**
 * @brief Structure for function/region argument information
 *
 * Represents metadata about function or region arguments including
 * their position, type, name, and value information.
 */
struct argument_info
{
    std::uint32_t arg_number = 0;   ///< Argument position/index
    std::string   arg_type   = {};  ///< Argument type (e.g., "int", "float*")
    std::string   arg_name   = {};  ///< Argument name
    std::string   arg_value  = {};  ///< Argument value as string
};

using function_args_t = std::vector<argument_info>;

/// Append a single field to the wire stream, percent-escaping the two characters
/// that would otherwise collide with the field grammar: '%' (the escape introducer)
/// and ';' (so a value can never reconstruct the ";;" delimiter). Escaping is
/// lossless and reversed by unescape_field().
inline void
append_escaped_field(std::string& out, std::string_view field)
{
    // fast path: nothing to escape (the overwhelmingly common case)
    if(field.find_first_of("%;") == std::string_view::npos)
    {
        out.append(field);
        return;
    }

    for(char ch : field)
    {
        switch(ch)
        {
            case '%': out.append("%25"); break;
            case ';': out.append("%3B"); break;
            default: out.push_back(ch); break;
        }
    }
}

/// Inverse of append_escaped_field(): decode the two escape sequences this format
/// produces (%25 -> '%', %3B -> ';'). Any other byte is copied verbatim.
inline std::string
unescape_field(std::string_view field)
{
    if(field.find('%') == std::string_view::npos)
    {
        return std::string{ field };
    }

    std::string out;
    out.reserve(field.size());
    for(std::size_t i = 0; i < field.size(); ++i)
    {
        if(field[i] == '%' && i + 2 < field.size())
        {
            if(field[i + 1] == '2' && field[i + 2] == '5')
            {
                out.push_back('%');
                i += 2;
                continue;
            }
            if(field[i + 1] == '3' && field[i + 2] == 'B')
            {
                out.push_back(';');
                i += 2;
                continue;
            }
        }
        out.push_back(field[i]);
    }
    return out;
}

inline std::string
get_args_string(const function_args_t& args)
{
    std::string args_str;
    std::for_each(args.begin(), args.end(), [&args_str](const argument_info& arg) {
        // arg_number is a uint32 -> never contains an escapable character
        args_str.append(std::to_string(arg.arg_number)).append(ARG_DELIMITER);
        append_escaped_field(args_str, arg.arg_type);
        args_str.append(ARG_DELIMITER);
        append_escaped_field(args_str, arg.arg_name);
        args_str.append(ARG_DELIMITER);
        append_escaped_field(args_str, arg.arg_value);
        args_str.append(ARG_DELIMITER);
    });
    return args_str;
}

inline function_args_t
process_arguments_string(const std::string& arg_str)
{
    function_args_t   args;
    const std::string delimiter{ ARG_DELIMITER };

    auto split = [](const std::string& str, const std::string& _delimiter) {
        std::vector<std::string> tokens;
        size_t                   start = 0;
        size_t                   end   = str.find(_delimiter);

        while(end != std::string::npos)
        {
            tokens.push_back(str.substr(start, end - start));
            start = end + _delimiter.length();
            end   = str.find(_delimiter, start);
        }

        return tokens;
    };

    auto tokens = split(arg_str, delimiter);

    // Ensure the number of tokens is a multiple of 4
    if(tokens.size() % 4 != 0)
    {
        throw std::invalid_argument("Malformed argument string.");
    }

    for(auto it = tokens.begin(); it != tokens.end(); it += 4)
    {
        const std::string& number_token = *it;
        std::uint32_t      arg_number   = 0;
        auto [ptr, ec]                  = std::from_chars(
            number_token.data(), number_token.data() + number_token.size(), arg_number);
        if(ec != std::errc{} || ptr != number_token.data() + number_token.size())
        {
            throw std::invalid_argument("Malformed argument string.");
        }

        argument_info arg = { arg_number, unescape_field(*(it + 1)),
                              unescape_field(*(it + 2)), unescape_field(*(it + 3)) };
        args.push_back(arg);
    }

    return args;
}

}  // namespace rocprofsys
