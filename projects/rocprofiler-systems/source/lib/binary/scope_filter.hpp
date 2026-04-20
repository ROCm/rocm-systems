// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/defines.hpp"

#include <cstdint>
#include <string>

namespace rocprofsys
{
namespace binary
{
struct scope_filter
{
    enum filter_mode : uint8_t
    {
        FILTER_INCLUDE = 0,
        FILTER_EXCLUDE
    };

    enum filter_scope : uint8_t
    {
        UNIVERSAL_FILTER = (1 << 0),
        BINARY_FILTER    = (1 << 1),
        SOURCE_FILTER    = (1 << 2),
        FUNCTION_FILTER  = (1 << 3)
    };

    filter_mode  mode       = FILTER_INCLUDE;
    filter_scope scope      = UNIVERSAL_FILTER;
    std::string  expression = {};

    bool operator()(std::string_view _value) const;

    template <typename ContainerT>
    static bool satisfies_filter(const ContainerT&, filter_scope,
                                 std::string_view) ROCPROFSYS_PURE;
};

template <typename ContainerT>
inline bool
scope_filter::satisfies_filter(const ContainerT& _filters, filter_scope _scope,
                               std::string_view _value)
{
    for(const auto& itr : _filters)  // NOLINT
    {
        // if the filter is for the specified scope and itr does not satisfy the
        // include/exclude mode, return false
        if((itr.scope & _scope) > 0 && !itr(_value)) return false;
    }
    return true;
}
}  // namespace binary
}  // namespace rocprofsys
