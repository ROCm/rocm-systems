// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "query_common.hpp"
#include "select/table_select_query.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

using namespace profiler_hub::queries::select;
using profiler_hub::queries::sort_order;

class table_select_query_test : public ::testing::Test
{
protected:
    table_select_query m_query;
};

TEST_F(table_select_query_test, simple_select_all)
{
    auto query_string = m_query.select_all().from("users").get_query_string();
    EXPECT_EQ(query_string, "SELECT * FROM users");
}

TEST_F(table_select_query_test, select_specific_columns)
{
    auto query_string =
        m_query.select("id", "name", "email").from("users").get_query_string();
    EXPECT_EQ(query_string, "SELECT id, name, email FROM users");
}

TEST_F(table_select_query_test, select_with_table_alias)
{
    auto query_string =
        m_query.select("u.id", "u.name").from("users", "u").get_query_string();
    EXPECT_EQ(query_string, "SELECT u.id, u.name FROM users u");
}

TEST_F(table_select_query_test, select_distinct)
{
    auto query_string =
        m_query.distinct().select("name").from("users").get_query_string();
    EXPECT_EQ(query_string, "SELECT DISTINCT name FROM users");
}

TEST_F(table_select_query_test, select_with_where)
{
    auto query_string =
        m_query.select("id", "name").from("users").where("active = ?").get_query_string();
    EXPECT_EQ(query_string, "SELECT id, name FROM users WHERE active = ?");
}

TEST_F(table_select_query_test, select_with_multiple_where_conditions)
{
    auto query_string = m_query.select("id", "name")
                            .from("users")
                            .where("active = ?")
                            .and_where("age > ?")
                            .or_where("role = ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT id, name FROM users WHERE active = ? AND age > ? OR role = ?");
}

TEST_F(table_select_query_test, select_with_inner_join)
{
    auto query_string = m_query.select("u.id", "o.total")
                            .from("users", "u")
                            .inner_join("orders", "o", "u.id = o.user_id")
                            .get_query_string();
    EXPECT_EQ(
        query_string,
        "SELECT u.id, o.total FROM users u INNER JOIN orders AS o ON u.id = o.user_id");
}

TEST_F(table_select_query_test, select_with_left_join)
{
    auto query_string = m_query.select("users.id", "orders.total")
                            .from("users")
                            .left_join("orders", "users.id = orders.user_id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT users.id, orders.total FROM users LEFT JOIN orders ON users.id = "
              "orders.user_id");
}

TEST_F(table_select_query_test, select_with_right_join_and_alias)
{
    auto query_string = m_query.select("u.id", "o.total")
                            .from("users", "u")
                            .right_join("orders", "o", "u.id = o.user_id")
                            .get_query_string();
    EXPECT_EQ(
        query_string,
        "SELECT u.id, o.total FROM users u RIGHT JOIN orders AS o ON u.id = o.user_id");
}

TEST_F(table_select_query_test, select_with_order_by)
{
    auto query_string = m_query.select("id", "name")
                            .from("users")
                            .order_by("name", sort_order::ascending)
                            .get_query_string();
    EXPECT_EQ(query_string, "SELECT id, name FROM users ORDER BY name ASC");
}

TEST_F(table_select_query_test, select_with_order_by_desc)
{
    auto query_string = m_query.select("id", "created_at")
                            .from("users")
                            .order_by("created_at", sort_order::descending)
                            .get_query_string();
    EXPECT_EQ(query_string, "SELECT id, created_at FROM users ORDER BY created_at DESC");
}

TEST_F(table_select_query_test, select_with_multiple_order_by)
{
    auto query_string = m_query.select("id", "name", "age")
                            .from("users")
                            .order_by("name", sort_order::ascending)
                            .order_by("age", sort_order::descending)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT id, name, age FROM users ORDER BY name ASC, age DESC");
}

TEST_F(table_select_query_test, select_with_limit)
{
    auto query_string =
        m_query.select("id", "name").from("users").limit(10).get_query_string();
    EXPECT_EQ(query_string, "SELECT id, name FROM users LIMIT 10");
}

TEST_F(table_select_query_test, select_with_limit_and_offset)
{
    auto query_string = m_query.select("id", "name")
                            .from("users")
                            .limit(10)
                            .offset(20)
                            .get_query_string();
    EXPECT_EQ(query_string, "SELECT id, name FROM users LIMIT 10 OFFSET 20");
}

TEST_F(table_select_query_test, select_with_group_by)
{
    auto query_string = m_query.select("customer_id", "COUNT(*)")
                            .from("orders")
                            .group_by("customer_id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id");
}

TEST_F(table_select_query_test, select_with_group_by_and_having)
{
    auto query_string = m_query.select("customer_id", "COUNT(*)")
                            .from("orders")
                            .group_by("customer_id")
                            .having("COUNT(*) > ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING "
              "COUNT(*) > ?");
}

TEST_F(table_select_query_test, complex_query)
{
    auto query_string = m_query.select("u.id", "u.name", "COUNT(o.id)")
                            .from("users", "u")
                            .left_join("orders", "o", "u.id = o.user_id")
                            .where("u.active = ?")
                            .group_by("u.id", "u.name")
                            .having("COUNT(o.id) > ?")
                            .order_by("u.name", sort_order::ascending)
                            .limit(100)
                            .get_query_string();

    EXPECT_EQ(query_string,
              "SELECT u.id, u.name, COUNT(o.id) FROM users u"
              " LEFT JOIN orders AS o ON u.id = o.user_id"
              " WHERE u.active = ?"
              " GROUP BY u.id, u.name"
              " HAVING COUNT(o.id) > ?"
              " ORDER BY u.name ASC"
              " LIMIT 100");
}

TEST_F(table_select_query_test, reuse_query_builder)
{
    auto query1 = m_query.select("id").from("users").get_query_string();

    auto query2 = m_query.select("id", "total")
                      .from("orders")
                      .where("total > ?")
                      .get_query_string();

    EXPECT_EQ(query1, "SELECT id FROM users");
    EXPECT_EQ(query2, "SELECT id, total FROM orders WHERE total > ?");
}

TEST_F(table_select_query_test, select_all_with_where_and_limit)
{
    auto query_string = m_query.select_all()
                            .from("products")
                            .where("price > ?")
                            .limit(50)
                            .get_query_string();
    EXPECT_EQ(query_string, "SELECT * FROM products WHERE price > ? LIMIT 50");
}

TEST_F(table_select_query_test, select_distinct_with_order_by_and_limit)
{
    auto query_string = m_query.distinct()
                            .select("name")
                            .from("categories")
                            .order_by("name", sort_order::ascending)
                            .limit(100)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT DISTINCT name FROM categories ORDER BY name ASC LIMIT 100");
}

TEST_F(table_select_query_test, multiple_joins)
{
    auto query_string = m_query.select("o.id", "c.name", "p.title")
                            .from("orders", "o")
                            .inner_join("customers", "c", "o.customer_id = c.id")
                            .left_join("products", "p", "o.product_id = p.id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT o.id, c.name, p.title FROM orders o"
              " INNER JOIN customers AS c ON o.customer_id = c.id"
              " LEFT JOIN products AS p ON o.product_id = p.id");
}

TEST_F(table_select_query_test, multiple_joins_with_where)
{
    auto query_string = m_query.select("o.id", "c.name", "p.title", "o.total")
                            .from("orders", "o")
                            .inner_join("customers", "c", "o.customer_id = c.id")
                            .left_join("products", "p", "o.product_id = p.id")
                            .where("o.status = ?")
                            .and_where("o.total > ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT o.id, c.name, p.title, o.total FROM orders o"
              " INNER JOIN customers AS c ON o.customer_id = c.id"
              " LEFT JOIN products AS p ON o.product_id = p.id"
              " WHERE o.status = ? AND o.total > ?");
}

TEST_F(table_select_query_test, join_with_group_by_having_order_limit)
{
    auto query_string = m_query.select("s.product_id", "p.name", "SUM(s.amount)")
                            .from("sales", "s")
                            .inner_join("products", "p", "s.product_id = p.id")
                            .group_by("s.product_id", "p.name")
                            .having("SUM(s.amount) > ?")
                            .order_by("SUM(s.amount)", sort_order::descending)
                            .limit(10)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT s.product_id, p.name, SUM(s.amount) FROM sales s"
              " INNER JOIN products AS p ON s.product_id = p.id"
              " GROUP BY s.product_id, p.name"
              " HAVING SUM(s.amount) > ?"
              " ORDER BY SUM(s.amount) DESC"
              " LIMIT 10");
}

TEST_F(table_select_query_test, distinct_with_join_and_where)
{
    auto query_string = m_query.distinct()
                            .select("c.country")
                            .from("orders", "o")
                            .left_join("customers", "c", "o.customer_id = c.id")
                            .where("o.year = ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT DISTINCT c.country FROM orders o"
              " LEFT JOIN customers AS c ON o.customer_id = c.id"
              " WHERE o.year = ?");
}

TEST_F(table_select_query_test, where_with_or_conditions_and_order_by)
{
    auto query_string = m_query.select("id", "name", "role")
                            .from("users")
                            .where("role = ?")
                            .or_where("role = ?")
                            .or_where("role = ?")
                            .order_by("name", sort_order::ascending)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT id, name, role FROM users"
              " WHERE role = ? OR role = ? OR role = ?"
              " ORDER BY name ASC");
}

TEST_F(table_select_query_test, group_by_multiple_columns_with_having)
{
    auto query_string = m_query.select("region", "product", "SUM(quantity)")
                            .from("sales")
                            .group_by("region", "product")
                            .having("SUM(quantity) >= ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT region, product, SUM(quantity) FROM sales"
              " GROUP BY region, product"
              " HAVING SUM(quantity) >= ?");
}

TEST_F(table_select_query_test, offset_without_order_by)
{
    auto query_string = m_query.select("id", "message", "timestamp")
                            .from("logs")
                            .limit(100)
                            .offset(500)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT id, message, timestamp FROM logs LIMIT 100 OFFSET 500");
}

TEST_F(table_select_query_test, multiple_order_by_with_limit_offset)
{
    auto query_string = m_query.select("id", "name", "department", "salary")
                            .from("employees")
                            .order_by("department", sort_order::ascending)
                            .order_by("salary", sort_order::descending)
                            .order_by("name", sort_order::ascending)
                            .limit(25)
                            .offset(50)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT id, name, department, salary FROM employees"
              " ORDER BY department ASC, salary DESC, name ASC"
              " LIMIT 25 OFFSET 50");
}

TEST_F(table_select_query_test, full_query_all_clauses)
{
    auto query_string = m_query.distinct()
                            .select("t.account_id", "a.name", "SUM(t.amount)", "COUNT(*)")
                            .from("transactions", "t")
                            .inner_join("accounts", "a", "t.account_id = a.id")
                            .left_join("account_types", "at", "a.type_id = at.id")
                            .where("t.date >= ?")
                            .and_where("t.date <= ?")
                            .and_where("at.category = ?")
                            .group_by("t.account_id", "a.name")
                            .having("SUM(t.amount) > ?")
                            .having("COUNT(*) >= ?")
                            .order_by("SUM(t.amount)", sort_order::descending)
                            .order_by("a.name", sort_order::ascending)
                            .limit(100)
                            .offset(0)
                            .get_query_string();

    EXPECT_EQ(query_string,
              "SELECT DISTINCT t.account_id, a.name, SUM(t.amount), COUNT(*)"
              " FROM transactions t"
              " INNER JOIN accounts AS a ON t.account_id = a.id"
              " LEFT JOIN account_types AS at ON a.type_id = at.id"
              " WHERE t.date >= ? AND t.date <= ? AND at.category = ?"
              " GROUP BY t.account_id, a.name"
              " HAVING SUM(t.amount) > ?"
              " HAVING COUNT(*) >= ?"
              " ORDER BY SUM(t.amount) DESC, a.name ASC"
              " LIMIT 100 OFFSET 0");
}

TEST_F(table_select_query_test, join_without_alias)
{
    auto query_string = m_query.select("users.id", "users.name", "orders.total")
                            .from("users")
                            .inner_join("orders", "users.id = orders.user_id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT users.id, users.name, orders.total FROM users"
              " INNER JOIN orders ON users.id = orders.user_id");
}

TEST_F(table_select_query_test, right_join_full_query)
{
    auto query_string = m_query.select("d.name", "COUNT(e.id)")
                            .from("departments", "d")
                            .right_join("employees", "e", "d.id = e.department_id")
                            .group_by("d.name")
                            .order_by("COUNT(e.id)", sort_order::descending)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT d.name, COUNT(e.id) FROM departments d"
              " RIGHT JOIN employees AS e ON d.id = e.department_id"
              " GROUP BY d.name"
              " ORDER BY COUNT(e.id) DESC");
}

TEST_F(table_select_query_test, skip_optional_clauses_where_directly_to_order)
{
    auto query_string = m_query.select("id", "name", "price")
                            .from("products")
                            .order_by("price", sort_order::ascending)
                            .limit(10)
                            .get_query_string();
    EXPECT_EQ(query_string,
              "SELECT id, name, price FROM products ORDER BY price ASC LIMIT 10");
}

TEST_F(table_select_query_test, skip_optional_clauses_where_directly_to_limit)
{
    auto query_string =
        m_query.select("id", "name").from("events").limit(5).get_query_string();
    EXPECT_EQ(query_string, "SELECT id, name FROM events LIMIT 5");
}

TEST_F(table_select_query_test, having_without_group_by)
{
    // Note: This is semantically invalid SQL but syntactically allowed by builder
    auto query_string = m_query.select("category", "total")
                            .from("stats")
                            .having("total > ?")
                            .get_query_string();
    EXPECT_EQ(query_string, "SELECT category, total FROM stats HAVING total > ?");
}

TEST_F(table_select_query_test, kernel_dispatch_timeline_event_query)
{
    auto query_string =
        m_query.select("K.id", "K.start", "K.end", "E.category_id", "KS.display_name")
            .from("rocpd_kernel_dispatch", "K")
            .inner_join("rocpd_event", "E", "E.id = event_id")
            .inner_join("rocpd_info_kernel_symbol", "KS", "KS.id = kernel_id")
            .get_query_string();
    EXPECT_EQ(
        query_string,
        "SELECT K.id, K.start, K.end, E.category_id, KS.display_name FROM "
        "rocpd_kernel_dispatch K INNER JOIN rocpd_event AS E ON E.id = event_id INNER "
        "JOIN rocpd_info_kernel_symbol AS KS ON KS.id = kernel_id");
}

TEST_F(table_select_query_test, query_reusability)
{
    auto query_builder = m_query.select("id", "name").from("users");
    EXPECT_EQ(query_builder.get_query_string(), "SELECT id, name FROM users");

    auto query1 = query_builder.where("age > ?").get_query_string();
    EXPECT_EQ(query1, "SELECT id, name FROM users WHERE age > ?");

    auto query2 = query_builder.where("name LIKE '?'").get_query_string();
    EXPECT_EQ(query2, "SELECT id, name FROM users WHERE name LIKE '?'");
}

TEST_F(table_select_query_test, query_reusability_with_join)
{
    auto query_builder = m_query.select("id", "name")
                             .from("users")
                             .inner_join("orders", "users.id = orders.user_id");
    EXPECT_EQ(
        query_builder.get_query_string(),
        "SELECT id, name FROM users INNER JOIN orders ON users.id = orders.user_id");

    auto query1 = query_builder.where("age > ?").get_query_string();
    EXPECT_EQ(query1,
              "SELECT id, name FROM users INNER JOIN orders ON users.id = orders.user_id "
              "WHERE age > ?");

    auto query2 = query_builder.where("name LIKE '?'").get_query_string();
    EXPECT_EQ(query2,
              "SELECT id, name FROM users INNER JOIN orders ON users.id = orders.user_id "
              "WHERE name LIKE '?'");
}

}  // namespace
