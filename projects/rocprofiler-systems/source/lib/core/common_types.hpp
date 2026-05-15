// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys
{
/**
 * @brief Structure for function/region argument information
 *
 * Represents metadata about function or region arguments including
 * their position, type, name, and value information.
 */
struct argument_info
{
    uint32_t    arg_number = 0;   ///< Argument position/index
    std::string arg_type   = {};  ///< Argument type (e.g., "int", "float*")
    std::string arg_name   = {};  ///< Argument name
    std::string arg_value  = {};  ///< Argument value as string
};

using function_args_t = std::vector<argument_info>;

inline std::string
get_args_string(const function_args_t& args)
{
    std::string args_str;
    std::for_each(args.begin(), args.end(), [&args_str](const argument_info& arg) {
        const auto*       delimiter = ";;";
        std::stringstream ss;
        ss << arg.arg_number << delimiter << arg.arg_type << delimiter << arg.arg_name
           << delimiter << arg.arg_value << delimiter;
        args_str.append(ss.str());
    });
    return args_str;
}

inline function_args_t
process_arguments_string(const std::string& arg_str)
{
    function_args_t   args;
    const std::string delimiter = ";;";

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
        argument_info arg = { static_cast<uint32_t>(std::stoi(*it)), *(it + 1), *(it + 2),
                              *(it + 3) };
        args.push_back(arg);
    }

    return args;
}

}  // namespace rocprofsys
