// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "query_common.hpp"
#include "select/table_select_query.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

using namespace rocstorage::queries::select;
using rocstorage::queries::sort_order;

class table_select_query_test : public ::testing::Test
{
protected:
    table_select_query m_query;
};

TEST_F(table_select_query_test, simple_select_all)
{
    auto query_string = m_query.from("users").select_all().get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT *");
}

TEST_F(table_select_query_test, select_specific_columns)
{
    auto query_string =
        m_query.from("users").select("id", "name", "email").get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT id, name, email");
}

TEST_F(table_select_query_test, select_with_table_alias)
{
    auto query_string =
        m_query.from("users", "u").select("u.id", "u.name").get_query_string();
    EXPECT_EQ(query_string, " FROM users u SELECT u.id, u.name");
}

TEST_F(table_select_query_test, select_distinct)
{
    auto query_string =
        m_query.from("users").distinct().select("name").get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT DISTINCT name");
}

TEST_F(table_select_query_test, select_with_where)
{
    auto query_string =
        m_query.from("users").select("id", "name").where("active = ?").get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT id, name WHERE active = ?");
}

TEST_F(table_select_query_test, select_with_multiple_where_conditions)
{
    auto query_string = m_query.from("users")
                            .select("id", "name")
                            .where("active = ?")
                            .and_where("age > ?")
                            .or_where("role = ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM users SELECT id, name WHERE active = ? AND age > ? OR role = ?");
}

TEST_F(table_select_query_test, select_with_inner_join)
{
    auto query_string = m_query.from("users", "u")
                            .select("u.id", "o.total")
                            .inner_join("orders", "o", "u.id = o.user_id")
                            .get_query_string();
    EXPECT_EQ(
        query_string,
        " FROM users u SELECT u.id, o.total INNER JOIN orders AS o ON u.id = o.user_id");
}

TEST_F(table_select_query_test, select_with_left_join)
{
    auto query_string = m_query.from("users")
                            .select("users.id", "orders.total")
                            .left_join("orders", "users.id = orders.user_id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM users SELECT users.id, orders.total LEFT JOIN orders ON users.id = "
              "orders.user_id");
}

TEST_F(table_select_query_test, select_with_right_join_and_alias)
{
    auto query_string = m_query.from("users", "u")
                            .select("u.id", "o.total")
                            .right_join("orders", "o", "u.id = o.user_id")
                            .get_query_string();
    EXPECT_EQ(
        query_string,
        " FROM users u SELECT u.id, o.total RIGHT JOIN orders AS o ON u.id = o.user_id");
}

TEST_F(table_select_query_test, select_with_order_by)
{
    auto query_string = m_query.from("users")
                            .select("id", "name")
                            .order_by("name", sort_order::asc)
                            .get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT id, name ORDER BY name ASC");
}

TEST_F(table_select_query_test, select_with_order_by_desc)
{
    auto query_string = m_query.from("users")
                            .select("id", "created_at")
                            .order_by("created_at", sort_order::desc)
                            .get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT id, created_at ORDER BY created_at DESC");
}

TEST_F(table_select_query_test, select_with_multiple_order_by)
{
    auto query_string = m_query.from("users")
                            .select("id", "name", "age")
                            .order_by("name", sort_order::asc)
                            .order_by("age", sort_order::desc)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM users SELECT id, name, age ORDER BY name ASC, age DESC");
}

TEST_F(table_select_query_test, select_with_limit)
{
    auto query_string =
        m_query.from("users").select("id", "name").limit(10).get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT id, name LIMIT 10");
}

TEST_F(table_select_query_test, select_with_limit_and_offset)
{
    auto query_string = m_query.from("users")
                            .select("id", "name")
                            .limit(10)
                            .offset(20)
                            .get_query_string();
    EXPECT_EQ(query_string, " FROM users SELECT id, name LIMIT 10 OFFSET 20");
}

TEST_F(table_select_query_test, select_with_group_by)
{
    auto query_string = m_query.from("orders")
                            .select("customer_id", "COUNT(*)")
                            .group_by("customer_id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM orders SELECT customer_id, COUNT(*) GROUP BY customer_id");
}

TEST_F(table_select_query_test, select_with_group_by_and_having)
{
    auto query_string = m_query.from("orders")
                            .select("customer_id", "COUNT(*)")
                            .group_by("customer_id")
                            .having("COUNT(*) > ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM orders SELECT customer_id, COUNT(*) GROUP BY customer_id HAVING "
              "COUNT(*) > ?");
}

TEST_F(table_select_query_test, complex_query)
{
    auto query_string = m_query.from("users", "u")
                            .select("u.id", "u.name", "COUNT(o.id)")
                            .left_join("orders", "o", "u.id = o.user_id")
                            .where("u.active = ?")
                            .group_by("u.id", "u.name")
                            .having("COUNT(o.id) > ?")
                            .order_by("u.name", sort_order::asc)
                            .limit(100)
                            .get_query_string();

    EXPECT_EQ(query_string,
              " FROM users u SELECT u.id, u.name, COUNT(o.id)"
              " LEFT JOIN orders AS o ON u.id = o.user_id"
              " WHERE u.active = ?"
              " GROUP BY u.id, u.name"
              " HAVING COUNT(o.id) > ?"
              " ORDER BY u.name ASC"
              " LIMIT 100");
}

TEST_F(table_select_query_test, reuse_query_builder)
{
    auto query1 = m_query.from("users").select("id").get_query_string();

    auto query2 = m_query.from("orders")
                      .select("id", "total")
                      .where("total > ?")
                      .get_query_string();

    EXPECT_EQ(query1, " FROM users SELECT id");
    EXPECT_EQ(query2, " FROM orders SELECT id, total WHERE total > ?");
}

TEST_F(table_select_query_test, select_all_with_where_and_limit)
{
    auto query_string = m_query.from("products")
                            .select_all()
                            .where("price > ?")
                            .limit(50)
                            .get_query_string();
    EXPECT_EQ(query_string, " FROM products SELECT * WHERE price > ? LIMIT 50");
}

TEST_F(table_select_query_test, select_distinct_with_order_by_and_limit)
{
    auto query_string = m_query.from("categories")
                            .distinct()
                            .select("name")
                            .order_by("name", sort_order::asc)
                            .limit(100)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM categories SELECT DISTINCT name ORDER BY name ASC LIMIT 100");
}

TEST_F(table_select_query_test, multiple_joins)
{
    auto query_string = m_query.from("orders", "o")
                            .select("o.id", "c.name", "p.title")
                            .inner_join("customers", "c", "o.customer_id = c.id")
                            .left_join("products", "p", "o.product_id = p.id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM orders o SELECT o.id, c.name, p.title"
              " INNER JOIN customers AS c ON o.customer_id = c.id"
              " LEFT JOIN products AS p ON o.product_id = p.id");
}

TEST_F(table_select_query_test, multiple_joins_with_where)
{
    auto query_string = m_query.from("orders", "o")
                            .select("o.id", "c.name", "p.title", "o.total")
                            .inner_join("customers", "c", "o.customer_id = c.id")
                            .left_join("products", "p", "o.product_id = p.id")
                            .where("o.status = ?")
                            .and_where("o.total > ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM orders o SELECT o.id, c.name, p.title, o.total"
              " INNER JOIN customers AS c ON o.customer_id = c.id"
              " LEFT JOIN products AS p ON o.product_id = p.id"
              " WHERE o.status = ? AND o.total > ?");
}

TEST_F(table_select_query_test, join_with_group_by_having_order_limit)
{
    auto query_string = m_query.from("sales", "s")
                            .select("s.product_id", "p.name", "SUM(s.amount)")
                            .inner_join("products", "p", "s.product_id = p.id")
                            .group_by("s.product_id", "p.name")
                            .having("SUM(s.amount) > ?")
                            .order_by("SUM(s.amount)", sort_order::desc)
                            .limit(10)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM sales s SELECT s.product_id, p.name, SUM(s.amount)"
              " INNER JOIN products AS p ON s.product_id = p.id"
              " GROUP BY s.product_id, p.name"
              " HAVING SUM(s.amount) > ?"
              " ORDER BY SUM(s.amount) DESC"
              " LIMIT 10");
}

TEST_F(table_select_query_test, distinct_with_join_and_where)
{
    auto query_string = m_query.from("orders", "o")
                            .distinct()
                            .select("c.country")
                            .left_join("customers", "c", "o.customer_id = c.id")
                            .where("o.year = ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM orders o SELECT DISTINCT c.country"
              " LEFT JOIN customers AS c ON o.customer_id = c.id"
              " WHERE o.year = ?");
}

TEST_F(table_select_query_test, where_with_or_conditions_and_order_by)
{
    auto query_string = m_query.from("users")
                            .select("id", "name", "role")
                            .where("role = ?")
                            .or_where("role = ?")
                            .or_where("role = ?")
                            .order_by("name", sort_order::asc)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM users SELECT id, name, role"
              " WHERE role = ? OR role = ? OR role = ?"
              " ORDER BY name ASC");
}

TEST_F(table_select_query_test, group_by_multiple_columns_with_having)
{
    auto query_string = m_query.from("sales")
                            .select("region", "product", "SUM(quantity)")
                            .group_by("region", "product")
                            .having("SUM(quantity) >= ?")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM sales SELECT region, product, SUM(quantity)"
              " GROUP BY region, product"
              " HAVING SUM(quantity) >= ?");
}

TEST_F(table_select_query_test, offset_without_order_by)
{
    auto query_string = m_query.from("logs")
                            .select("id", "message", "timestamp")
                            .limit(100)
                            .offset(500)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM logs SELECT id, message, timestamp LIMIT 100 OFFSET 500");
}

TEST_F(table_select_query_test, multiple_order_by_with_limit_offset)
{
    auto query_string = m_query.from("employees")
                            .select("id", "name", "department", "salary")
                            .order_by("department", sort_order::asc)
                            .order_by("salary", sort_order::desc)
                            .order_by("name", sort_order::asc)
                            .limit(25)
                            .offset(50)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM employees SELECT id, name, department, salary"
              " ORDER BY department ASC, salary DESC, name ASC"
              " LIMIT 25 OFFSET 50");
}

TEST_F(table_select_query_test, full_query_all_clauses)
{
    auto query_string = m_query.from("transactions", "t")
                            .distinct()
                            .select("t.account_id", "a.name", "SUM(t.amount)", "COUNT(*)")
                            .inner_join("accounts", "a", "t.account_id = a.id")
                            .left_join("account_types", "at", "a.type_id = at.id")
                            .where("t.date >= ?")
                            .and_where("t.date <= ?")
                            .and_where("at.category = ?")
                            .group_by("t.account_id", "a.name")
                            .having("SUM(t.amount) > ?")
                            .having("COUNT(*) >= ?")
                            .order_by("SUM(t.amount)", sort_order::desc)
                            .order_by("a.name", sort_order::asc)
                            .limit(100)
                            .offset(0)
                            .get_query_string();

    EXPECT_EQ(query_string,
              " FROM transactions t SELECT DISTINCT t.account_id, a.name, SUM(t.amount), "
              "COUNT(*)"
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
    auto query_string = m_query.from("users")
                            .select("users.id", "users.name", "orders.total")
                            .inner_join("orders", "users.id = orders.user_id")
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM users SELECT users.id, users.name, orders.total"
              " INNER JOIN orders ON users.id = orders.user_id");
}

TEST_F(table_select_query_test, right_join_full_query)
{
    auto query_string = m_query.from("departments", "d")
                            .select("d.name", "COUNT(e.id)")
                            .right_join("employees", "e", "d.id = e.department_id")
                            .group_by("d.name")
                            .order_by("COUNT(e.id)", sort_order::desc)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM departments d SELECT d.name, COUNT(e.id)"
              " RIGHT JOIN employees AS e ON d.id = e.department_id"
              " GROUP BY d.name"
              " ORDER BY COUNT(e.id) DESC");
}

TEST_F(table_select_query_test, skip_optional_clauses_where_directly_to_order)
{
    auto query_string = m_query.from("products")
                            .select("id", "name", "price")
                            .order_by("price", sort_order::asc)
                            .limit(10)
                            .get_query_string();
    EXPECT_EQ(query_string,
              " FROM products SELECT id, name, price ORDER BY price ASC LIMIT 10");
}

TEST_F(table_select_query_test, skip_optional_clauses_where_directly_to_limit)
{
    auto query_string =
        m_query.from("events").select("id", "name").limit(5).get_query_string();
    EXPECT_EQ(query_string, " FROM events SELECT id, name LIMIT 5");
}

TEST_F(table_select_query_test, having_without_group_by)
{
    // Note: This is semantically invalid SQL but syntactically allowed by builder
    auto query_string = m_query.from("stats")
                            .select("category", "total")
                            .having("total > ?")
                            .get_query_string();
    EXPECT_EQ(query_string, " FROM stats SELECT category, total HAVING total > ?");
}

}  // namespace
