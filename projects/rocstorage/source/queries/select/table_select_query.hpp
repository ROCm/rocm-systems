// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "select_query_builders.hpp"

#include <sstream>
#include <string_view>

namespace rocstorage::queries::select
{

/**
 * @brief Fluent SQL SELECT query builder.
 *
 * Provides a type-safe, chainable interface for constructing SELECT queries.
 * The builder enforces SQL clause ordering at compile time.
 *
 * ## Supported Clauses
 * - FROM (with optional table alias)
 * - SELECT / SELECT DISTINCT
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
 * auto sql = query.from("users").select_all().get_query_string();
 * // Result: " FROM users SELECT *"
 * @endcode
 *
 * ### Select specific columns:
 * @code
 * auto sql = query.from("users").select("id", "name", "email").get_query_string();
 * // Result: " FROM users SELECT id, name, email"
 * @endcode
 *
 * ### With WHERE clause (use ? for parameter binding):
 * @code
 * auto sql = query.from("users")
 *                 .select("id", "name")
 *                 .where("status = ?")
 *                 .and_where("age > ?")
 *                 .get_query_string();
 * // Result: " FROM users SELECT id, name WHERE status = ? AND age > ?"
 * @endcode
 *
 * ### With DISTINCT:
 * @code
 * auto sql = query.from("orders")
 *                 .distinct()
 *                 .select("customer_id")
 *                 .get_query_string();
 * // Result: " FROM orders SELECT DISTINCT customer_id"
 * @endcode
 *
 * ### With JOINs:
 * @code
 * auto sql = query.from("orders", "o")
 *                 .select("o.id", "c.name")
 *                 .inner_join("customers", "c", "o.customer_id = c.id")
 *                 .left_join("products", "p", "o.product_id = p.id")
 *                 .get_query_string();
 * // Result: " FROM orders o SELECT o.id, c.name
 * //          INNER JOIN customers AS c ON o.customer_id = c.id
 * //          LEFT JOIN products AS p ON o.product_id = p.id"
 * @endcode
 *
 * ### With GROUP BY and HAVING:
 * @code
 * auto sql = query.from("orders")
 *                 .select("customer_id", "SUM(amount) as total")
 *                 .group_by("customer_id")
 *                 .having("total > ?")
 *                 .get_query_string();
 * // Result: " FROM orders SELECT customer_id, SUM(amount) as total
 * //          GROUP BY customer_id HAVING total > ?"
 * @endcode
 *
 * ### With ORDER BY, LIMIT, and OFFSET:
 * @code
 * auto sql = query.from("users")
 *                 .select("id", "name")
 *                 .order_by("created_at", sort_order::desc)
 *                 .order_by("name", sort_order::asc)
 *                 .limit(10)
 *                 .offset(20)
 *                 .get_query_string();
 * // Result: " FROM users SELECT id, name ORDER BY created_at DESC, name ASC
 * //          LIMIT 10 OFFSET 20"
 * @endcode
 *
 * ### Full query with all clauses:
 * @code
 * auto sql = query.from("orders", "o")
 *                 .distinct()
 *                 .select("c.name", "SUM(o.amount) as total")
 *                 .inner_join("customers", "c", "o.customer_id = c.id")
 *                 .where("o.status = ?")
 *                 .group_by("c.name")
 *                 .having("total > ?")
 *                 .order_by("total", sort_order::desc)
 *                 .limit(10)
 *                 .get_query_string();
 * @endcode
 *
 * @note The query string starts with a space for easy concatenation.
 * @note Use ? placeholders for parameter binding with prepared statements.
 */
struct table_select_query
{
    table_select_query()
    : m_from_builder{ m_ss }
    {}

    select_columns_builder& from(std::string_view table)
    {
        m_ss.str("");
        return m_from_builder.from(table);
    }

    select_columns_builder& from(std::string_view table, std::string_view alias)
    {
        m_ss.str("");
        return m_from_builder.from(table, alias);
    }

private:
    std::stringstream   m_ss;
    from_clause_builder m_from_builder;
};

}  // namespace rocstorage::queries::select
