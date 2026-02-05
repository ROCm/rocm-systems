// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "../query_builder_base.hpp"
#include "../query_common.hpp"
#include "traits.hpp"

#include <cstddef>
#include <sstream>
#include <string_view>
#include <type_traits>

namespace rocstorage::queries::select
{

// Forward declarations - ordered by builder chain position
class limit_clause_builder;
class order_by_clause_builder;
class having_clause_builder;
class group_by_clause_builder;
class where_clause_builder;
class join_clause_builder;
class from_clause_builder;
class select_columns_builder;

class limit_clause_builder : public query_builder_base
{
public:
    explicit limit_clause_builder(std::stringstream& ss);

    limit_clause_builder& limit(size_t count);
    limit_clause_builder& offset(size_t count);
};

class order_by_clause_builder : public query_builder_base
{
public:
    explicit order_by_clause_builder(std::stringstream& ss);

    order_by_clause_builder& order_by(std::string_view column,
                                      sort_order       order = sort_order::ascending);

    limit_clause_builder& limit(size_t count);
    limit_clause_builder& offset(size_t count);

private:
    bool                 m_has_order_by{ false };
    limit_clause_builder m_limit_builder;
};

class having_clause_builder : public query_builder_base
{
public:
    explicit having_clause_builder(std::stringstream& ss);

    having_clause_builder& having(std::string_view condition);

    order_by_clause_builder& order_by(std::string_view column,
                                      sort_order       order = sort_order::ascending);
    limit_clause_builder&    limit(size_t count);

private:
    order_by_clause_builder m_order_by_builder;
};

class group_by_clause_builder : public query_builder_base
{
public:
    explicit group_by_clause_builder(std::stringstream& ss);

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    having_clause_builder& group_by(Columns&&... columns)
    {
        m_stream << " GROUP BY ";
        append_columns(std::forward<Columns>(columns)...);
        return m_having_builder;
    }

    having_clause_builder&   having(std::string_view condition);
    order_by_clause_builder& order_by(std::string_view column,
                                      sort_order       order = sort_order::ascending);
    limit_clause_builder&    limit(size_t count);

private:
    having_clause_builder m_having_builder;
};

class where_clause_builder : public query_builder_base
{
public:
    explicit where_clause_builder(std::stringstream& ss);

    where_clause_builder& where(std::string_view condition);
    where_clause_builder& and_where(std::string_view condition);
    where_clause_builder& or_where(std::string_view condition);

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    having_clause_builder& group_by(Columns&&... columns)
    {
        return m_group_by_builder.group_by(std::forward<Columns>(columns)...);
    }

    having_clause_builder&   having(std::string_view condition);
    order_by_clause_builder& order_by(std::string_view column,
                                      sort_order       order = sort_order::ascending);
    limit_clause_builder&    limit(size_t count);

private:
    bool                    m_has_where{ false };
    group_by_clause_builder m_group_by_builder;
};

class join_clause_builder : public query_builder_base
{
public:
    explicit join_clause_builder(std::stringstream& ss);

    join_clause_builder& inner_join(std::string_view table,
                                    std::string_view on_condition);
    join_clause_builder& inner_join(std::string_view table,
                                    std::string_view alias,
                                    std::string_view on_condition);
    join_clause_builder& left_join(std::string_view table, std::string_view on_condition);
    join_clause_builder& left_join(std::string_view table,
                                   std::string_view alias,
                                   std::string_view on_condition);
    join_clause_builder& right_join(std::string_view table,
                                    std::string_view on_condition);
    join_clause_builder& right_join(std::string_view table,
                                    std::string_view alias,
                                    std::string_view on_condition);

    where_clause_builder& where(std::string_view condition);

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    having_clause_builder& group_by(Columns&&... columns)
    {
        return m_where_builder.group_by(std::forward<Columns>(columns)...);
    }

    having_clause_builder&   having(std::string_view condition);
    order_by_clause_builder& order_by(std::string_view column,
                                      sort_order       order = sort_order::ascending);
    limit_clause_builder&    limit(size_t count);

private:
    where_clause_builder m_where_builder;
};

class from_clause_builder : public query_builder_base
{
public:
    explicit from_clause_builder(std::stringstream& ss);

    join_clause_builder& from(std::string_view table);
    join_clause_builder& from(std::string_view table, std::string_view alias);

private:
    join_clause_builder m_join_builder;
};

class select_columns_builder : public query_builder_base
{
public:
    explicit select_columns_builder(std::stringstream& ss);

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    from_clause_builder& select(Columns&&... columns)
    {
        m_stream << "SELECT ";
        if(m_distinct)
        {
            m_stream << "DISTINCT ";
        }
        append_columns(std::forward<Columns>(columns)...);
        return m_from_builder;
    }

    from_clause_builder& select_all();

    select_columns_builder& distinct();

private:
    bool                m_distinct{ false };
    from_clause_builder m_from_builder;
};

}  // namespace rocstorage::queries::select
