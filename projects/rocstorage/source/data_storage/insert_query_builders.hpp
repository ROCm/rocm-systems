// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "traits.hpp"

#include <sstream>
#include <string>
#include <type_traits>

namespace rocstorage
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
    : m_stream{ ss }
    {}

    template <typename... Values>
    query_value_builder& set_values(Values&&... values)
    {
        auto i = sizeof...(values);
        m_stream << "( ";
        ((process_value(values) << (i-- > 1 ? ", " : " ")), ...);
        m_stream << ")";
        return *this;
    }

    std::string get_query_string() { return m_stream.str(); }

private:
    template <typename T>
    std::enable_if_t<common::traits::is_string_literal<T>(), std::stringstream&>
    process_value(T& value)
    {
        m_stream << "\"" << value << "\"";
        return m_stream;
    }

    template <typename T>
    std::enable_if_t<common::traits::is_optional_v<std::decay_t<T>>, std::stringstream&>
    process_value(T& value)
    {
        if(value.has_value())
        {
            m_stream << value.value();
        }
        else
        {
            m_stream << "NULL";
        }
        return m_stream;
    }

    template <typename T>
    std::enable_if_t<!common::traits::is_string_literal<T>() &&
                         !common::traits::is_optional_v<std::decay_t<T>>,
                     std::stringstream&>
    process_value(T& value)
    {
        m_stream << value;
        return m_stream;
    }

private:
    std::stringstream& m_stream;
};

struct query_columns_builder
{
    query_columns_builder(std::stringstream& ss)
    : m_stream{ ss }
    , m_value_builder{ m_stream }
    {}

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    query_value_builder& set_columns(Columns&... columns)
    {
        auto i = sizeof...(columns);
        m_stream << "( ";
        ((m_stream << columns << (i-- > 1 ? ", " : " ")), ...) << ") VALUES ";
        return m_value_builder;
    }

private:
    std::stringstream&  m_stream;
    query_value_builder m_value_builder;
};

}  // namespace query_builders
}  // namespace queries
}  // namespace data_storage
}  // namespace rocstorage
