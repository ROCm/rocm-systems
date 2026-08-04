// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/schema_catalog.hpp"
#include "backends/sqlite_backend_impl.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

TEST(sqlite_backend_impl_test, get_schema_query_returns_non_empty_sql_for_rocpd_tables)
{
    const std::string query =
        get_schema_query(::rocpd::data_storage::schema::VERSIONS_3_0_0_ROCPD_TABLES_SQL,
                         "test-uuid",
                         profiler_hub::version_t{ 3, 0, 0 });
    EXPECT_FALSE(query.empty());
}

TEST(sqlite_backend_impl_test, get_schema_query_for_metadata_embeds_requested_version)
{
    const std::string query =
        get_schema_query(::rocpd::data_storage::schema::VERSIONS_3_0_1_ROCPD_METADATA_SQL,
                         "test-uuid",
                         profiler_hub::version_t{ 3, 0, 1 });

    EXPECT_NE(query.find(R"(("schema_version", "3.0.1"))"), std::string::npos);
    EXPECT_NE(query.find(R"(("schema_version_major", "3"))"), std::string::npos);
    EXPECT_NE(query.find(R"(("schema_version_minor", "0"))"), std::string::npos);
    EXPECT_NE(query.find(R"(("schema_version_patch", "1"))"), std::string::npos);
}

TEST(sqlite_backend_impl_test, get_schema_query_returns_empty_for_null_schema_content)
{
    const std::string query =
        get_schema_query(nullptr, "test-uuid", profiler_hub::version_t{ 3, 0, 0 });
    EXPECT_TRUE(query.empty());
}

}  // namespace
