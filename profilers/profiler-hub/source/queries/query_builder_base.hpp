// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "traits.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace profiler_hub::queries
{

class query_builder_base
{
public:
    explicit query_builder_base(std::stringstream& ss);
    virtual ~query_builder_base() = default;

    query_builder_base(const query_builder_base&)            = default;
    query_builder_base& operator=(const query_builder_base&) = delete;
    query_builder_base(query_builder_base&&)                 = default;
    query_builder_base& operator=(query_builder_base&&)      = delete;

    [[nodiscard]] std::string get_query_string() const;

protected:
    std::stringstream& m_stream;

    void append_keyword(std::string_view keyword);
    void append_raw(std::string_view text);

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    void append_columns(Columns&&... columns)
    {
        auto count = sizeof...(columns);
        ((m_stream << columns << (--count > 0 ? ", " : "")), ...);
    }

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    void append_columns_parenthesized(Columns&&... columns)
    {
        m_stream << "( ";
        auto count = sizeof...(columns);
        ((m_stream << columns << (--count > 0 ? ", " : " ")), ...);
        m_stream << ")";
    }
};

}  // namespace profiler_hub::queries
