// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// addc/detail/format.hpp
//
// Minimal, dependency-free replacement for std::format, sufficient for the
// format-string grammar used by ADDC Base.
//
// Why this exists: libstdc++ did not ship <format> until GCC 13.1, but the
// Yocto Kirkstone toolchain is GCC 11. Rather than take an external dependency
// (e.g. fmtlib), this header implements exactly the subset of the std::format
// grammar in use so that `addc::format(...)` can be swapped for
// `addc::format(...)` mechanically, preserving every format string verbatim.
//
// Supported grammar (everything used in these repos, nothing more):
//   {}            default: integers -> decimal, strings -> passthrough
//   {:x} {:X}     hex, lower / upper case, no padding
//   {:0Nx} {:0NX} zero-padded hex to width N          (N in
//   2,3,4,6,8,12,16,...)
//   {:Nx} {:NX}   width-padded hex (space fill)
//   {:d}          decimal
//   {:0Nd} {:0N}  zero-padded decimal to width N      (e.g. timestamps)
//   {:Nd} {:N}    width-padded decimal (space fill)
//   {{  }}        literal brace escapes
//
// NOT supported (intentionally — none appear in the codebase): floats, sign
// flags, '#' alt-form, positional/named indices, alignment, non-'0' fill,
// string field widths. Using an unsupported spec throws std::runtime_error so
// mistakes surface loudly rather than silently mis-formatting.

#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace addc
{
namespace detail
{

// Copy literal text into `out`, collapsing "{{" -> "{" and "}}" -> "}".
inline void append_literal(std::string& out, std::string_view s)
{
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        const char c = s[i];
        if ((c == '{' || c == '}') && i + 1 < s.size() && s[i + 1] == c)
        {
            out += c;
            ++i;
        }
        else
        {
            out += c;
        }
    }
}

// Render one integer value according to `spec` (the text between ':' and '}').
// Grammar: [ '0' ] [ width-digits ] [ 'x' | 'X' | 'd' ]
inline std::string render_int(unsigned long long magnitude, bool negative,
                              bool is_signed, std::string_view spec)
{
    bool zero_pad = false;
    int width = 0;
    char type = 'd';

    std::size_t i = 0;
    if (i < spec.size() && spec[i] == '0')
    {
        zero_pad = true;
        ++i;
    }
    while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9')
    {
        width = width * 10 + (spec[i] - '0');
        ++i;
    }
    if (i < spec.size() && (spec[i] == 'x' || spec[i] == 'X' || spec[i] == 'd'))
    {
        type = spec[i];
        ++i;
    }
    if (i != spec.size())
    {
        throw std::runtime_error("addc::format: unsupported integer spec '" +
                                 std::string(spec) + "'");
    }

    // Decimal follows the signedness of the source type; hex is always
    // unsigned. Build a printf conversion spec dynamically, using the ll length
    // modifier with '*' width so we never risk buffer/width mismatches.
    const char conv_type = (type == 'd' && is_signed) ? 'd'
                           : (type == 'd')            ? 'u'
                                                      : type; // 'x' or 'X'
    char conv[16];
    std::snprintf(conv, sizeof(conv), "%%%s*ll%c", zero_pad ? "0" : "",
                  conv_type);

    char buf[80];
    if (conv_type == 'd')
    {
        // Signed decimal: reapply the sign we stripped for the unsigned path.
        long long signed_value = negative ? -static_cast<long long>(magnitude)
                                          : static_cast<long long>(magnitude);
        std::snprintf(buf, sizeof(buf), conv, width, signed_value);
    }
    else
    {
        std::snprintf(buf, sizeof(buf), conv, width, magnitude);
    }
    return std::string(buf);
}

// Append a single argument, dispatching on its type.
template <typename T>
void append_value(std::string& out, std::string_view spec, const T& value)
{
    using U = std::remove_cv_t<std::remove_reference_t<T>>;

    if constexpr (std::is_same_v<U, std::string> ||
                  std::is_same_v<U, std::string_view> ||
                  std::is_same_v<U, const char*> || std::is_same_v<U, char*>)
    {
        if (!spec.empty())
        {
            throw std::runtime_error(
                "addc::format: format spec not supported for strings");
        }
        out += value;
    }
    else if constexpr (std::is_enum_v<U>)
    {
        append_value(out, spec, static_cast<std::underlying_type_t<U>>(value));
    }
    else if constexpr (std::is_integral_v<U>)
    {
        if constexpr (std::is_signed_v<U>)
        {
            const bool negative = value < 0;
            // Take magnitude via unsigned to avoid UB on the most-negative
            // value.
            using UU = std::make_unsigned_t<U>;
            unsigned long long magnitude =
                negative ? static_cast<unsigned long long>(
                               ~static_cast<UU>(value) + 1u)
                         : static_cast<unsigned long long>(value);
            out += render_int(magnitude, negative, /*is_signed=*/true, spec);
        }
        else
        {
            out += render_int(static_cast<unsigned long long>(value), false,
                              /*is_signed=*/false, spec);
        }
    }
    else
    {
        static_assert(sizeof(T) == 0,
                      "addc::format: unsupported argument type");
    }
}

// Advance through `fmt` up to and including the next placeholder, appending
// literal text to `out`, and return the spec (text after ':') plus the
// remaining format string after '}'. Sets `found` to false if no placeholder
// remains (all leading text is emitted as literal).
struct NextPlaceholder
{
    bool found;
    std::string_view spec;
    std::string_view rest;
};

inline NextPlaceholder next_placeholder(std::string& out, std::string_view fmt)
{
    std::size_t i = 0;
    while (i < fmt.size())
    {
        const char c = fmt[i];
        if (c == '{')
        {
            if (i + 1 < fmt.size() && fmt[i + 1] == '{')
            {
                out += '{';
                i += 2;
                continue;
            }
            // Real placeholder. Find the closing '}'.
            std::size_t close = fmt.find('}', i + 1);
            if (close == std::string_view::npos)
            {
                throw std::runtime_error(
                    "addc::format: unmatched '{' in format string");
            }
            std::string_view inside = fmt.substr(i + 1, close - (i + 1));
            std::string_view spec;
            if (!inside.empty())
            {
                if (inside[0] != ':')
                {
                    throw std::runtime_error(
                        "addc::format: unsupported placeholder '{" +
                        std::string(inside) + "}'");
                }
                spec = inside.substr(1);
            }
            return {true, spec, fmt.substr(close + 1)};
        }
        if (c == '}')
        {
            if (i + 1 < fmt.size() && fmt[i + 1] == '}')
            {
                out += '}';
                i += 2;
                continue;
            }
            throw std::runtime_error(
                "addc::format: stray '}' in format string");
        }
        out += c;
        ++i;
    }
    return {false, {}, {}};
}

inline void format_into(std::string& out, std::string_view fmt)
{
    // No more args: emit remaining text (with brace unescaping); any leftover
    // placeholder is an error.
    NextPlaceholder np = next_placeholder(out, fmt);
    if (np.found)
    {
        throw std::runtime_error(
            "addc::format: more placeholders than arguments");
    }
}

template <typename T, typename... Rest>
void format_into(std::string& out, std::string_view fmt, const T& value,
                 const Rest&... rest)
{
    NextPlaceholder np = next_placeholder(out, fmt);
    if (!np.found)
    {
        throw std::runtime_error(
            "addc::format: more arguments than placeholders");
    }
    append_value(out, np.spec, value);
    format_into(out, np.rest, rest...);
}

} // namespace detail

// Drop-in replacement for the std::format subset used in this codebase.
template <typename... Args>
std::string format(std::string_view fmt, const Args&... args)
{
    std::string out;
    detail::format_into(out, fmt, args...);
    return out;
}

} // namespace addc
