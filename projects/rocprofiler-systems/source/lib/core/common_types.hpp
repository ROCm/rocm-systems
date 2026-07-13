// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <sstream>
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

/// A serialized record is four delimiter-terminated fields: idx;;type;;name;;value;;
inline constexpr std::size_t fields_per_record = 4;

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

/// Append a single field to the wire stream, percent-escaping the ';' and '%'
/// characters. Escaping is lossless and reversed by unescape_field().
inline void
append_escaped_field(std::string& out, std::string_view field)
{
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

/// Inverse of append_escaped_field(): decode %25 -> '%' and %3B -> ';'. Any other byte
/// (including a stray '%' that is not part of a recognized escape) is copied verbatim.
inline std::string
unescape_field(std::string_view field)
{
    // fast path: no escape introducer means there is nothing to decode
    if(field.find('%') == std::string_view::npos)
    {
        return std::string{ field };
    }

    constexpr std::size_t escape_seq_len = 3;
    constexpr std::size_t code_len       = 2;

    std::string out;
    out.reserve(field.size());
    for(std::size_t i = 0; i < field.size();)
    {
        // Only treat '%' as an escape when a full escape sequence still fits
        const bool has_full_escape =
            (field[i] == '%' && i + escape_seq_len <= field.size());
        if(has_full_escape)
        {
            const auto code = field.substr(i + 1, code_len);
            if(code == "25")
            {
                out += '%';
                i += escape_seq_len;
                continue;
            }
            if(code == "3B")
            {
                out += ';';
                i += escape_seq_len;
                continue;
            }
        }
        out += field[i];
        ++i;
    }
    return out;
}

inline std::string
get_args_string(const function_args_t& args)
{
    std::string args_str;
    std::for_each(args.begin(), args.end(), [&args_str](const argument_info& arg) {
        // arg_number is a uint32 and never contains an escapable character
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

    // Ensure the number of tokens is a whole number of records
    if(tokens.size() % fields_per_record != 0)
    {
        throw std::invalid_argument("Malformed argument string.");
    }

    for(auto it = tokens.begin(); it != tokens.end(); it += fields_per_record)
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
