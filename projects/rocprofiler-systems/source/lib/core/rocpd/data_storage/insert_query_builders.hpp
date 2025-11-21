// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "common/traits.hpp"

#include <sstream>
#include <string>
#include <type_traits>

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
namespace queries
{

namespace query_builders
{

struct query_value_builder
{
    query_value_builder(std::stringstream& ss)
    : _ss{ ss }
    {}

    template <typename... Values>
    query_value_builder& set_values(Values&&... values)
    {
        auto i = sizeof...(values);
        _ss << "( ";
        ((process_value(values) << (i-- > 1 ? ", " : " ")), ...);
        _ss << ")";
        return *this;
    }

    std::string get_query_string() { return _ss.str(); }

private:
    template <typename T>
    std::enable_if_t<common::traits::is_string_literal<T>(), std::stringstream&>
    process_value(T& value)
    {
        _ss << "\"" << value << "\"";
        return _ss;
    }

    template <typename T>
    std::enable_if_t<common::traits::is_optional_v<std::decay_t<T>>, std::stringstream&>
    process_value(T& value)
    {
        if(value.has_value())
        {
            _ss << value.value();
        }
        else
        {
            _ss << "NULL";
        }
        return _ss;
    }

    template <typename T>
    std::enable_if_t<!common::traits::is_string_literal<T>() &&
                         !common::traits::is_optional_v<std::decay_t<T>>,
                     std::stringstream&>
    process_value(T& value)
    {
        _ss << value;
        return _ss;
    }

private:
    std::stringstream& _ss;
};

struct query_columns_builder
{
    query_columns_builder(std::stringstream& ss)
    : _ss{ ss }
    , _query_value_builder{ _ss }
    {}

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    query_value_builder& set_columns(Columns&... columns)
    {
        auto i = sizeof...(columns);
        _ss << "( ";
        ((_ss << columns << (i-- > 1 ? ", " : " ")), ...) << ") VALUES ";
        return _query_value_builder;
    }

private:
    std::stringstream&  _ss;
    query_value_builder _query_value_builder;
};

}  // namespace query_builders
}  // namespace queries
}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
