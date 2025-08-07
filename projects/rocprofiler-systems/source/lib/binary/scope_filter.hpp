// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
