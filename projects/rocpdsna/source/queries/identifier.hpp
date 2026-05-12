// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <string_view>

namespace rocpdsna::queries
{

/**
 * Wrap a SQL identifier in backticks so non-alphanumeric characters
 * (e.g. hyphens in uuid-derived table suffixes) are not parsed as
 * SQL operators. If the caller already passed a backtick-wrapped name,
 * it is returned unchanged.
 */
inline std::string
quote_identifier(std::string_view name)
{
    if(name.size() >= 2 && name.front() == '`' && name.back() == '`')
    {
        return std::string{ name };
    }
    std::string out;
    out.reserve(name.size() + 2);
    out.push_back('`');
    out.append(name.data(), name.size());
    out.push_back('`');
    return out;
}

}  // namespace rocpdsna::queries
