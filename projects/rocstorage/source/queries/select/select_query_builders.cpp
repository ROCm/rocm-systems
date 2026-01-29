// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "select_query_builders.hpp"

namespace rocstorage::queries::select
{

limit_clause_builder::limit_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
{}

limit_clause_builder&
limit_clause_builder::limit(size_t count)
{
    m_stream << " LIMIT " << count;
    return *this;
}

limit_clause_builder&
limit_clause_builder::offset(size_t count)
{
    m_stream << " OFFSET " << count;
    return *this;
}

order_by_clause_builder::order_by_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_limit_builder{ ss }
{}

order_by_clause_builder&
order_by_clause_builder::order_by(std::string_view column, sort_order order)
{
    if(!m_has_order_by)
    {
        m_stream << " ORDER BY ";
        m_has_order_by = true;
    }
    else
    {
        m_stream << ", ";
    }

    m_stream << column;
    m_stream << (order == sort_order::asc ? " ASC" : " DESC");
    return *this;
}

limit_clause_builder&
order_by_clause_builder::limit(size_t count)
{
    return m_limit_builder.limit(count);
}

limit_clause_builder&
order_by_clause_builder::offset(size_t count)
{
    return m_limit_builder.offset(count);
}

having_clause_builder::having_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_order_by_builder{ ss }
{}

having_clause_builder&
having_clause_builder::having(std::string_view condition)
{
    m_stream << " HAVING " << condition;
    return *this;
}

order_by_clause_builder&
having_clause_builder::order_by(std::string_view column, sort_order order)
{
    return m_order_by_builder.order_by(column, order);
}

limit_clause_builder&
having_clause_builder::limit(size_t count)
{
    return m_order_by_builder.limit(count);
}

group_by_clause_builder::group_by_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_having_builder{ ss }
{}

having_clause_builder&
group_by_clause_builder::having(std::string_view condition)
{
    return m_having_builder.having(condition);
}

order_by_clause_builder&
group_by_clause_builder::order_by(std::string_view column, sort_order order)
{
    return m_having_builder.order_by(column, order);
}

limit_clause_builder&
group_by_clause_builder::limit(size_t count)
{
    return m_having_builder.limit(count);
}

where_clause_builder::where_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_group_by_builder{ ss }
{}

where_clause_builder&
where_clause_builder::where(std::string_view condition)
{
    if(!m_has_where)
    {
        m_stream << " WHERE " << condition;
        m_has_where = true;
    }
    else
    {
        m_stream << " AND " << condition;
    }
    return *this;
}

where_clause_builder&
where_clause_builder::and_where(std::string_view condition)
{
    if(!m_has_where)
    {
        return where(condition);
    }
    m_stream << " AND " << condition;
    return *this;
}

where_clause_builder&
where_clause_builder::or_where(std::string_view condition)
{
    if(!m_has_where)
    {
        return where(condition);
    }
    m_stream << " OR " << condition;
    return *this;
}

having_clause_builder&
where_clause_builder::having(std::string_view condition)
{
    return m_group_by_builder.having(condition);
}

order_by_clause_builder&
where_clause_builder::order_by(std::string_view column, sort_order order)
{
    return m_group_by_builder.order_by(column, order);
}

limit_clause_builder&
where_clause_builder::limit(size_t count)
{
    return m_group_by_builder.limit(count);
}

join_clause_builder::join_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_where_builder{ ss }
{}

join_clause_builder&
join_clause_builder::inner_join(std::string_view table, std::string_view on_condition)
{
    m_stream << " INNER JOIN " << table << " ON " << on_condition;
    return *this;
}

join_clause_builder&
join_clause_builder::inner_join(std::string_view table,
                                std::string_view alias,
                                std::string_view on_condition)
{
    m_stream << " INNER JOIN " << table << " AS " << alias << " ON " << on_condition;
    return *this;
}

join_clause_builder&
join_clause_builder::left_join(std::string_view table, std::string_view on_condition)
{
    m_stream << " LEFT JOIN " << table << " ON " << on_condition;
    return *this;
}

join_clause_builder&
join_clause_builder::left_join(std::string_view table,
                               std::string_view alias,
                               std::string_view on_condition)
{
    m_stream << " LEFT JOIN " << table << " AS " << alias << " ON " << on_condition;
    return *this;
}

join_clause_builder&
join_clause_builder::right_join(std::string_view table, std::string_view on_condition)
{
    m_stream << " RIGHT JOIN " << table << " ON " << on_condition;
    return *this;
}

join_clause_builder&
join_clause_builder::right_join(std::string_view table,
                                std::string_view alias,
                                std::string_view on_condition)
{
    m_stream << " RIGHT JOIN " << table << " AS " << alias << " ON " << on_condition;
    return *this;
}

where_clause_builder&
join_clause_builder::where(std::string_view condition)
{
    m_where_builder.where(condition);
    return m_where_builder;
}

having_clause_builder&
join_clause_builder::having(std::string_view condition)
{
    return m_where_builder.having(condition);
}

order_by_clause_builder&
join_clause_builder::order_by(std::string_view column, sort_order order)
{
    return m_where_builder.order_by(column, order);
}

limit_clause_builder&
join_clause_builder::limit(size_t count)
{
    return m_where_builder.limit(count);
}

select_columns_builder::select_columns_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_join_builder{ ss }
{}

join_clause_builder&
select_columns_builder::select_all()
{
    m_stream << " SELECT ";
    if(m_distinct)
    {
        m_stream << "DISTINCT ";
    }
    m_stream << "*";
    return m_join_builder;
}

select_columns_builder&
select_columns_builder::distinct()
{
    m_distinct = true;
    return *this;
}

from_clause_builder::from_clause_builder(std::stringstream& ss)
: query_builder_base{ ss }
, m_select_builder{ ss }
{}

select_columns_builder&
from_clause_builder::from(std::string_view table)
{
    m_stream << " FROM " << table;
    return m_select_builder;
}

select_columns_builder&
from_clause_builder::from(std::string_view table, std::string_view alias)
{
    m_stream << " FROM " << table << " " << alias;
    return m_select_builder;
}

}  // namespace rocstorage::queries::select
