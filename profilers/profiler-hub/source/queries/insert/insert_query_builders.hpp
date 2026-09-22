// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "traits.hpp"

#include <sstream>
#include <string>
#include <type_traits>

namespace profiler_hub::queries::insert
{

struct query_value_builder
{
    explicit query_value_builder(std::stringstream& ss)
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

    [[nodiscard]] std::string get_query_string() const { return m_stream.str(); }

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

    std::stringstream& m_stream;
};

struct query_columns_builder
{
    explicit query_columns_builder(std::stringstream& ss)
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

}  // namespace profiler_hub::queries::insert
