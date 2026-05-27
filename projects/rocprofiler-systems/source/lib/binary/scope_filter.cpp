// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scope_filter.hpp"
#include "core/exception.hpp"

#include <regex>

namespace rocprofsys::binary
{
bool
scope_filter::operator()(std::string_view _value) const
{
    if(mode == FILTER_INCLUDE)
    {
        return (expression.empty())
                   ? true
                   : std::regex_search(_value.data(), std::regex{ expression });
    }
    if(mode == FILTER_EXCLUDE)
    {
        return (expression.empty())
                   ? false
                   : !std::regex_search(_value.data(), std::regex{ expression });
    }
    throw exception<std::runtime_error>{ "invalid scope filter mode" };
}
}  // namespace rocprofsys::binary
