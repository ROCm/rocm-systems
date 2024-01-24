// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/rocprofiler-sdk/hip/details/ostream.hpp"

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace utils
{
inline static std::ostream&
operator<<(std::ostream& out, const hipDeviceProp_tR0000& v)
{
    return ::rocprofiler::hip::detail::operator<<(out, v);
}

template <typename Tp>
auto
stringize_impl(const Tp& _v)
{
    using nonpointer_type = typename std::remove_pointer_t<Tp>;

    if constexpr(common::mpl::is_pair<Tp>::value)
    {
        return std::make_pair(stringize_impl(_v.first), stringize_impl(_v.second));
    }
    else if constexpr(std::is_constructible<std::string_view, Tp>::value)
    {
        auto _ss = std::stringstream{};
        _ss << _v;
        return _ss.str();
    }
    else if constexpr(fmt::is_formattable<Tp>::value && !std::is_pointer<Tp>::value)
    {
        return fmt::format("{}", _v);
    }
    else if constexpr(std::is_pointer<Tp>::value && !std::is_pointer<nonpointer_type>::value &&
                      common::mpl::is_type_complete_v<nonpointer_type> &&
                      !std::is_void<nonpointer_type>::value)
    {
        if(_v)
        {
            return stringize_impl(*_v);
        }
        else
        {
            auto _ss = std::stringstream{};
            _ss << _v;
            return _ss.str();
        }
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
stringize(Args... args)
{
    return std::vector<std::pair<std::string, std::string>>{stringize_impl(args)...};
}
}  // namespace utils
}  // namespace hip
}  // namespace rocprofiler
