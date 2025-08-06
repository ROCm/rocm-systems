// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

#include "lib/common/mpl.hpp"

#include <fmt/format.h>

#include <string>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace tool
{
struct output_key
{
    output_key(std::string _key, std::string _val, std::string _desc = {});

    template <typename Tp,
              typename Up                                                    = Tp,
              std::enable_if_t<!common::mpl::is_string_type<Up>::value, int> = 0>
    output_key(std::string _key, Tp&& _val, std::string _desc = {});

    operator std::pair<std::string, std::string>() const;

    std::string key         = {};
    std::string value       = {};
    std::string description = {};
};

template <typename Tp, typename Up, std::enable_if_t<!common::mpl::is_string_type<Up>::value, int>>
output_key::output_key(std::string _key, Tp&& _val, std::string _desc)
: key{std::move(_key)}
, value{fmt::format("{}", std::forward<Tp>(_val))}
, description{std::move(_desc)}
{}

std::vector<output_key>
output_keys(std::string _tag = {});
}  // namespace tool
}  // namespace rocprofiler
