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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/common/mpl.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace common
{
struct stringified_argument
{
    int32_t     indirection_level = 0;
    int32_t     dereference_count = 0;
    const char* type              = nullptr;
    const char* name              = nullptr;
    std::string value             = {};
};

template <size_t N>
using stringified_argument_array_t =
    container::small_vector<stringified_argument, std::min<size_t>(N, 6)>;

template <typename Tp, typename FuncT>
auto
stringize_arg_impl(const Tp& _v, const int32_t max_deref, int32_t& deref_cnt, FuncT&& impl)
{
    using value_type      = std::decay_t<Tp>;
    using nonpointer_type = std::remove_pointer_t<Tp>;

    if constexpr(common::mpl::is_string_type<value_type>::value &&
                 !std::is_pointer<nonpointer_type>::value)
    {
        if constexpr(std::is_pointer<value_type>::value)
        {
            if(!_v) return std::string{"(null)"};
        }

        return std::string{_v};
    }
    else if constexpr(fmt::is_formattable<value_type>::value && !std::is_pointer<value_type>::value)
    {
        return fmt::format("{}", _v);
    }
    else if constexpr(std::is_pointer<value_type>::value &&
                      !std::is_pointer<nonpointer_type>::value &&
                      common::mpl::is_type_complete_v<nonpointer_type> &&
                      !std::is_void<nonpointer_type>::value)
    {
        if(_v && deref_cnt < max_deref)
            return stringize_arg_impl(*_v, max_deref, ++deref_cnt, std::forward<FuncT>(impl));
        else if(_v)
            return std::forward<FuncT>(impl)(_v);
        else
            return std::string{"(null)"};
    }
    else if constexpr(std::is_pointer<value_type>::value && std::is_pointer<nonpointer_type>::value)
    {
        using next_nonpointer_type = std::remove_pointer_t<nonpointer_type>;

        if(_v)
        {
            if constexpr(!std::is_void<next_nonpointer_type>::value)
            {
                if(deref_cnt < max_deref)
                    return stringize_arg_impl(
                        *_v, max_deref, ++deref_cnt, std::forward<FuncT>(impl));
                else
                    return std::forward<FuncT>(impl)(_v);
            }
            else
            {
                return std::forward<FuncT>(impl)(_v);
            }
        }
        else
        {
            return std::string{"(null)"};
        }
    }
    else
    {
        return std::forward<FuncT>(impl)(_v);
    }
}

template <typename Tp, typename FuncT>
common::stringified_argument
stringize_arg(int32_t max_deref, const std::pair<const char*, Tp>& arg, FuncT&& impl)
{
    auto _arg              = common::stringified_argument{};
    _arg.indirection_level = mpl::indirection_level<Tp>::value;
    _arg.type              = typeid(Tp).name();
    _arg.name              = arg.first;
    _arg.value             = stringize_arg_impl(
        arg.second, max_deref, _arg.dereference_count, std::forward<FuncT>(impl));
    return _arg;
}
}  // namespace common
}  // namespace rocprofiler
