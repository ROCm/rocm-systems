// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/version.h>

#include "lib/common/mpl.hpp"
#include "lib/common/stringize_arg.hpp"
#include "lib/rocprofiler-sdk/rccl/details/fmt.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <sstream>
#include <type_traits>

namespace rocprofiler
{
namespace rccl
{
namespace utils
{
template <typename Tp>
auto
stringize_impl(const Tp& _v)
{
    using value_type = std::decay_t<Tp>;

    if constexpr(fmt::is_formattable<value_type>::value && !std::is_pointer<value_type>::value)
    {
        return fmt::format("{}", _v);
    }
    else
    {
        auto _ss = std::stringstream{};
        _ss << _v;
        return _ss.str();
    }
}

template <typename... Args>
auto
stringize(int32_t max_deref, Args... args)
{
    using array_type = common::stringified_argument_array_t<sizeof...(Args)>;
    return array_type{common::stringize_arg(
        max_deref, args, [](const auto& _v) { return stringize_impl(_v); })...};
}
}  // namespace utils
}  // namespace rccl
}  // namespace rocprofiler
