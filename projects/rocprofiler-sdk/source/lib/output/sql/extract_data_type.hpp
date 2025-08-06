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
struct extract_data_type;

template <typename Tp, int ExpectedV>
bool
column_data_is_null(sqlite3_stmt* stmt, int32_t col);

template <typename Tp>
struct extract_data_type<Tp, std::enable_if_t<std::is_integral<Tp>::value>>
{
    static constexpr auto value = SQLITE_INTEGER;

    std::optional<Tp> operator()(sqlite3_stmt* stmt, int32_t col) const
    {
        if(column_data_is_null<Tp, value>(stmt, col)) return std::nullopt;

        if constexpr(std::is_signed<Tp>::value)
        {
            if constexpr(sizeof(Tp) > sizeof(int32_t))
                return Tp{sqlite3_column_int64(stmt, col)};
            else
                return Tp{sqlite3_column_int(stmt, col)};
        }
        else
        {
            auto val = sqlite3_column_int64(stmt, col);
            return static_cast<Tp>(val);
        }
    }
};

template <typename Tp>
struct extract_data_type<Tp, std::enable_if_t<std::is_floating_point<Tp>::value>>
{
    static constexpr auto value = SQLITE_FLOAT;

    std::optional<Tp> operator()(sqlite3_stmt* stmt, int32_t col) const
    {
        if(column_data_is_null<Tp, value>(stmt, col)) return std::nullopt;

        return Tp{sqlite3_column_double(stmt, col)};
    }
};

template <typename Tp>
struct extract_data_type<Tp, std::enable_if_t<common::mpl::is_string_type<Tp>::value>>
{
    static constexpr auto value = SQLITE_TEXT;

    std::optional<Tp> operator()(sqlite3_stmt* stmt, int32_t col) const
    {
        if(column_data_is_null<Tp, value>(stmt, col)) return std::nullopt;

        const auto* ret = reinterpret_cast<const char*>(sqlite3_column_text(stmt, col));

        if constexpr(std::is_constructible<Tp, const char*>::value) return Tp{ret};

        return ret;
    }
};

template <typename Tp, size_t N>
struct extract_data_type<
    std::array<Tp, N>,
    std::enable_if_t<std::is_integral<Tp>::value && sizeof(Tp) == sizeof(uint8_t)>>
{
    static constexpr auto value = SQLITE_BLOB;

    using value_type = std::array<Tp, N>;
    std::optional<value_type> operator()(sqlite3_stmt* stmt, int32_t col) const
    {
        if(column_data_is_null<value_type, value>(stmt, col)) return std::nullopt;

        const auto* val = reinterpret_cast<const Tp*>(sqlite3_column_blob(stmt, col));

        auto ret = value_type{};
        ret.fill(0);

        uint64_t nbytes = std::min<uint64_t>(sqlite3_column_bytes(stmt, col), ret.size());
        for(uint64_t i = 0; i < nbytes; ++i)
            ret.at(i) = val[i];

        return ret;
    }
};

template <typename Tp>
struct extract_data_type<Tp, std::enable_if_t<std::is_same<Tp, std::nullptr_t>::value>>
{
    static constexpr auto value = SQLITE_TEXT;

    std::optional<Tp> operator()(sqlite3_stmt* stmt, int32_t col) const
    {
        if(column_data_is_null<Tp, value>(stmt, col)) return std::nullopt;
        return std::nullptr_t{};
    }
};
}  // namespace sql
}  // namespace tool
}  // namespace rocprofiler
