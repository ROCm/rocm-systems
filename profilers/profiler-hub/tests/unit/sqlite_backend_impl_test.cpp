// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/sqlite_backend_impl.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

TEST(sqlite_backend_impl_test, get_schema_query_returns_non_empty_sql_for_rocpd_tables)
{
    const std::string query =
        get_schema_query(ROCPD_SQL_SCHEMA_ROCPD_TABLES, "test-uuid");
    EXPECT_FALSE(query.empty());
}

}  // namespace
