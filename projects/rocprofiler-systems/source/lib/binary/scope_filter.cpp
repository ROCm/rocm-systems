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

#include "scope_filter.hpp"
#include "core/exception.hpp"

#include <regex>

namespace rocprofsys
{
namespace binary
{
bool
scope_filter::operator()(std::string_view _value) const
{
    if(mode == FILTER_INCLUDE)
        return (expression.empty())
                   ? true
                   : std::regex_search(_value.data(), std::regex{ expression });
    else if(mode == FILTER_EXCLUDE)
        return (expression.empty())
                   ? false
                   : !std::regex_search(_value.data(), std::regex{ expression });
    throw exception<std::runtime_error>{ "invalid scope filter mode" };
}
}  // namespace binary
}  // namespace rocprofsys
