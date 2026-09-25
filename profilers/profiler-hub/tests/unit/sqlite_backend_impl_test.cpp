// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/sqlite_backend_impl.hpp"
#include "data_storage/schema_version.hpp"

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

TEST(sqlite_backend_impl_test, get_schema_query_returns_non_empty_sql_for_rocpd_metadata)
{
    const std::string query =
        get_schema_query(ROCPD_SQL_SCHEMA_ROCPD_METADATA, "test-uuid");
    EXPECT_FALSE(query.empty());
}

TEST(sqlite_backend_impl_test, get_schema_query_substitutes_metadata_version_placeholders)
{
    const std::string query =
        get_schema_query(ROCPD_SQL_SCHEMA_ROCPD_METADATA, "test-uuid");

    constexpr auto schema_version = profiler_hub::data_storage::rocpd_schema_version;
    const auto     has = [&](const std::string& tag, const std::string& value) {
        return query.find("(\"" + tag + "\", \"" + value + "\")") != std::string::npos;
    };
    EXPECT_TRUE(has("schema_version",
                    std::to_string(schema_version.major) + "." +
                        std::to_string(schema_version.minor) + "." +
                        std::to_string(schema_version.patch)));
    EXPECT_TRUE(has("schema_version_major", std::to_string(schema_version.major)));
    EXPECT_TRUE(has("schema_version_minor", std::to_string(schema_version.minor)));
    EXPECT_TRUE(has("schema_version_patch", std::to_string(schema_version.patch)));
    EXPECT_EQ(query.find("{{schema_version}}"), std::string::npos);
    EXPECT_EQ(query.find("{{schema_version_major}}"), std::string::npos);
    EXPECT_EQ(query.find("{{schema_version_minor}}"), std::string::npos);
    EXPECT_EQ(query.find("{{schema_version_patch}}"), std::string::npos);
}

TEST(sqlite_backend_impl_test, get_schema_query_substitutes_metadata_uuid_and_guid)
{
    const std::string query =
        get_schema_query(ROCPD_SQL_SCHEMA_ROCPD_METADATA, "test-uuid");

    EXPECT_NE(query.find("rocpd_metadata_test-uuid"), std::string::npos);
    // {{uuid}} is substituted as "_" + uuid everywhere, including the metadata value row.
    EXPECT_NE(query.find("(\"uuid\", \"_test-uuid\")"), std::string::npos);
    EXPECT_NE(query.find("(\"guid\", \"test-uuid\")"), std::string::npos);
    EXPECT_EQ(query.find("{{uuid}}"), std::string::npos);
    EXPECT_EQ(query.find("{{guid}}"), std::string::npos);
}

}  // namespace
