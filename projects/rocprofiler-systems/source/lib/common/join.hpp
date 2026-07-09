// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/traits.hpp"

#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofsys
{
inline namespace common
{
/// @brief Join arbitrary args into one string separated by @p delim.
///
/// Sets @c std::boolalpha so bool args render as "true"/"false" (matching the
/// removed timemory join). When @p QuoteStrings is true, string-type args are
/// wrapped in double quotes (a nullptr @c char* is treated as empty). Used by the
/// logging macro (unquoted, space-separated) and the invoke() diagnostics
/// (quoted, comma-separated). Prefer @c fmt::join for plain ranges; this exists
/// for the heterogeneous / per-arg-quoting cases fmt cannot express directly.
/// @tparam QuoteStrings Whether to quote string-type args.
/// @param  delim        Separator inserted between args.
/// @param  args         Values to join.
/// @return The joined string.
template <bool QuoteStrings = false, typename... Args>
inline std::string
join_args(std::string_view delim, Args&&... args)
{
    std::ostringstream oss;
    oss << std::boolalpha;

    // Take the arg by forwarding reference and stream it without adding const:
    // some third-party types (e.g. Dyninst's BPatch_*) only provide a non-const
    // operator<<(std::ostream&, T&), so decaying to const here would drop their
    // overload and leave the call ambiguous against generic templated operator<<.
    bool first  = true;
    auto append = [&](auto&& arg) {
        if(!first) oss << delim;
        first = false;

        using decayed_arg_type = std::decay_t<decltype(arg)>;
        if constexpr(QuoteStrings && traits::string_literal<decayed_arg_type>)
        {
            // Prevent passing nullptr char* arg to operator<< - it would cause UB
            if constexpr(std::is_pointer_v<decayed_arg_type>)
            {
                oss << '"' << (arg ? arg : "") << '"';
            }
            else
            {
                oss << '"' << arg << '"';
            }
        }
        else
        {
            oss << arg;
        }
    };

    (append(std::forward<Args>(args)), ...);

    return oss.str();
}
}  // namespace common
}  // namespace rocprofsys
