// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "select_query_builders.hpp"

#include <sstream>
#include <string_view>

namespace profiler_hub::queries::select
{

/**
 * @brief Fluent SQL SELECT query builder.
 *
 * Provides a type-safe, chainable interface for constructing SELECT queries.
 * The builder enforces SQL clause ordering at compile time.
 *
 * ## Supported Clauses
 * - SELECT / SELECT DISTINCT (entry point)
 * - FROM (with optional table alias)
 * - JOIN (INNER, LEFT, RIGHT with optional AS alias)
 * - WHERE / AND / OR conditions
 * - GROUP BY
 * - HAVING
 * - ORDER BY (ASC/DESC)
 * - LIMIT / OFFSET
 *
 * ## Usage Examples
 *
 * ### Basic select all:
 * @code
 * table_select_query query;
 * auto sql = query.select_all().from("users").get_query_string();
 * // Result: "SELECT * FROM users"
 * @endcode
 *
 * ### Select specific columns:
 * @code
 * auto sql = query.select("id", "name", "email").from("users").get_query_string();
 * // Result: "SELECT id, name, email FROM users"
 * @endcode
 *
 * ### With WHERE clause (use ? for parameter binding):
 * @code
 * auto sql = query.select("id", "name")
 *                 .from("users")
 *                 .where("status = ?")
 *                 .and_where("age > ?")
 *                 .get_query_string();
 * // Result: "SELECT id, name FROM users WHERE status = ? AND age > ?"
 * @endcode
 *
 * ### With DISTINCT:
 * @code
 * auto sql = query.distinct()
 *                 .select("customer_id")
 *                 .from("orders")
 *                 .get_query_string();
 * // Result: "SELECT DISTINCT customer_id FROM orders"
 * @endcode
 *
 * ### With JOINs:
 * @code
 * auto sql = query.select("o.id", "c.name")
 *                 .from("orders", "o")
 *                 .inner_join("customers", "c", "o.customer_id = c.id")
 *                 .left_join("products", "p", "o.product_id = p.id")
 *                 .get_query_string();
 * // Result: "SELECT o.id, c.name FROM orders o
 * //          INNER JOIN customers AS c ON o.customer_id = c.id
 * //          LEFT JOIN products AS p ON o.product_id = p.id"
 * @endcode
 *
 * ### With GROUP BY and HAVING:
 * @code
 * auto sql = query.select("customer_id", "SUM(amount) as total")
 *                 .from("orders")
 *                 .group_by("customer_id")
 *                 .having("total > ?")
 *                 .get_query_string();
 * // Result: "SELECT customer_id, SUM(amount) as total FROM orders
 * //          GROUP BY customer_id HAVING total > ?"
 * @endcode
 *
 * ### With ORDER BY, LIMIT, and OFFSET:
 * @code
 * auto sql = query.select("id", "name")
 *                 .from("users")
 *                 .order_by("created_at", sort_order::desc)
 *                 .order_by("name", sort_order::asc)
 *                 .limit(10)
 *                 .offset(20)
 *                 .get_query_string();
 * // Result: "SELECT id, name FROM users ORDER BY created_at DESC, name ASC
 * //          LIMIT 10 OFFSET 20"
 * @endcode
 *
 * ### Full query with all clauses:
 * @code
 * auto sql = query.distinct()
 *                 .select("c.name", "SUM(o.amount) as total")
 *                 .from("orders", "o")
 *                 .inner_join("customers", "c", "o.customer_id = c.id")
 *                 .where("o.status = ?")
 *                 .group_by("c.name")
 *                 .having("total > ?")
 *                 .order_by("total", sort_order::desc)
 *                 .limit(10)
 *                 .get_query_string();
 * @endcode
 *
 * @note Use ? placeholders for parameter binding with prepared statements.
 */
struct table_select_query
{
    table_select_query()
    : m_select_builder{ m_ss }
    {}

    /**
     * Starts a SELECT query with specific columns.
     * @param columns Column names to select.
     * @return from_clause_builder for chaining .from() call.
     */
    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    from_clause_builder& select(Columns&&... columns)
    {
        m_ss.str("");
        m_select_builder.reset();
        return m_select_builder.select(std::forward<Columns>(columns)...);
    }

    /**
     * Starts a SELECT * query.
     * @return from_clause_builder for chaining .from() call.
     */
    from_clause_builder& select_all()
    {
        m_ss.str("");
        m_select_builder.reset();
        return m_select_builder.select_all();
    }

    /**
     * Enables DISTINCT modifier for the SELECT query.
     * Must be called before select() or select_all().
     * @return select_columns_builder for chaining .select() call.
     */
    select_columns_builder& distinct()
    {
        m_ss.str("");
        m_select_builder.reset();
        return m_select_builder.distinct();
    }

private:
    std::stringstream      m_ss;
    select_columns_builder m_select_builder;
};

}  // namespace profiler_hub::queries::select
