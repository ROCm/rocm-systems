// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/traits.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rocprofsys::inline common
{
// Join args into one string separated by `delim`. Bools render as true/false.
template <typename... Args>
[[nodiscard]] inline std::string
join(std::string_view delim, Args&&... args)
{
    std::ostringstream oss;
    oss << std::boolalpha;
    std::string_view sep = "";

    ((oss << sep << args, sep = delim), ...);

    return oss.str();
}

// Like join(), but string-type args are wrapped in double quotes.
template <typename... Args>
[[nodiscard]] inline std::string
join_with_strings_quoted(std::string_view delim, Args&&... args)
{
    auto quote_if_string = [](auto&& arg) -> decltype(auto) {
        using decayed_arg_type = std::decay_t<decltype(arg)>;
        if constexpr(traits::string_literal<decayed_arg_type>)
        {
            // Guard against nullptr char* - passing it to operator<< is UB.
            if constexpr(std::is_pointer_v<decayed_arg_type>)
            {
                return '"' + std::string{ arg ? arg : "" } + '"';
            }
            else
            {
                return '"' + std::string{ arg } + '"';
            }
        }
        else
        {
            return std::forward<decltype(arg)>(arg);
        }
    };

    return join(delim, quote_if_string(std::forward<Args>(args))...);
}
}  // namespace rocprofsys::inline common
