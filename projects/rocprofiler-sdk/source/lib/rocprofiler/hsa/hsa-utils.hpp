// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace utils
{
struct join_args
{
    std::string_view prefix    = {};
    std::string_view suffix    = {};
    std::string_view separator = {};
};

template <typename Tp>
auto
join_impl(const Tp& _v)
{
    return _v;
}

template <typename LhsT, typename RhsT>
auto
join_impl(const std::pair<LhsT, RhsT>& _v)
{
    auto _ss = std::stringstream{};
    _ss << _v.first << "=" << _v.second;
    return _ss.str();
}

template <typename... Args>
auto
join(join_args ja, Args... args)
{
    auto _content = std::string{};
    {
        auto _ss = std::stringstream{};
        ((_ss << ja.separator << join_impl(args)), ...);
        auto _v = _ss.str();
        if(_v.length() > ja.separator.length()) _content = _v.substr(2);
    }

    return (std::stringstream{} << ja.prefix << _content << ja.suffix).str();
}

template <typename Tp>
auto
stringize_impl(const Tp& _v)
{
    auto _ss = std::stringstream{};
    _ss << _v;
    return _ss.str();
}

template <typename LhsT, typename RhsT>
auto
stringize_impl(const std::pair<LhsT, RhsT>& _v)
{
    return std::make_pair(stringize_impl(_v.first), stringize_impl(_v.second));
}

template <typename... Args>
auto
stringize(Args... args)
{
    return std::vector<std::pair<std::string, std::string>>{stringize_impl(args)...};
}
}  // namespace utils
}  // namespace hsa
}  // namespace rocprofiler
