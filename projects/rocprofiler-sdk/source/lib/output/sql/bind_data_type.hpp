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

#include "lib/common/mpl.hpp"

#include <sqlite3.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace tool
{
namespace sql
{
template <typename Tp, typename CondT = void>
struct bind_data_type;

template <typename Tp>
struct bind_data_type<Tp, std::enable_if_t<std::is_integral<Tp>::value>>
{
    static constexpr auto value = SQLITE_INTEGER;

    auto operator()(sqlite3_stmt* stmt, int32_t col, Tp val) const
    {
        if constexpr(std::is_signed<Tp>::value)
        {
            if constexpr(sizeof(Tp) > sizeof(int32_t))
                return sqlite3_bind_int64(stmt, col, val);
            else
                return sqlite3_bind_int(stmt, col, val);
        }
        else
        {
            return sqlite3_bind_int64(stmt, col, val);
        }
    }
};

template <typename Tp>
struct bind_data_type<Tp, std::enable_if_t<std::is_floating_point<Tp>::value>>
{
    static constexpr auto value = SQLITE_FLOAT;

    auto operator()(sqlite3_stmt* stmt, int32_t col, Tp val) const
    {
        return sqlite3_bind_double(stmt, col, val);
    }
};

template <typename Tp>
struct bind_data_type<Tp, std::enable_if_t<common::mpl::is_string_type<Tp>::value>>
{
    static constexpr auto value = SQLITE_TEXT;

    auto operator()(sqlite3_stmt* stmt, int32_t col, Tp val) const
    {
        auto text_property = SQLITE_STATIC;
        if constexpr(std::is_same_v<Tp, std::string>) text_property = SQLITE_TRANSIENT;

        return sqlite3_bind_text(stmt, col, val.data(), -1, text_property);
    }
};
}  // namespace sql
}  // namespace tool
}  // namespace rocprofiler
