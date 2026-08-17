// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/schema_manifest.hpp"
#include "backends/sqlite_backend_impl.hpp"

#include <profiler-hub/version.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{

using profiler_hub::data_storage::load_schema_manifest;
using profiler_hub::data_storage::resolve_schema_version;
using profiler_hub::data_storage::version_file_map_t;

TEST(sqlite_backend_impl_test, get_schema_query_returns_non_empty_sql_for_rocpd_tables)
{
    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    load_schema_manifest(schema_directory(), version_file_map, latest_version);

    const auto resolved = resolve_schema_version(
        version_file_map, latest_version, profiler_hub::version_t{});
    const auto& kind_paths = version_file_map.at(resolved.to_string());

    const std::string query = get_schema_query(
        ROCPD_SQL_SCHEMA_ROCPD_TABLES, kind_paths, resolved, "test-uuid");
    EXPECT_FALSE(query.empty());
}

TEST(schema_manifest_test, load_schema_manifest_returns_non_empty_manifest)
{
    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    load_schema_manifest(schema_directory(), version_file_map, latest_version);

    EXPECT_FALSE(version_file_map.empty());

    for(const auto& [version, kind_paths] : version_file_map)
    {
        EXPECT_FALSE(kind_paths.at("rocpd_tables").empty());
        EXPECT_FALSE(kind_paths.at("rocpd_views").empty());
        EXPECT_FALSE(kind_paths.at("rocpd_indexes").empty());
        EXPECT_FALSE(kind_paths.at("rocpd_data_views").empty());
        EXPECT_FALSE(kind_paths.at("rocpd_summary_views").empty());
        EXPECT_FALSE(kind_paths.at("rocpd_metadata").empty());
    }
}

TEST(schema_manifest_test, resolve_schema_version_latest_sentinel_picks_highest_version)
{
    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    load_schema_manifest(schema_directory(), version_file_map, latest_version);

    const auto resolved = resolve_schema_version(
        version_file_map, latest_version, profiler_hub::version_t{});
    EXPECT_TRUE(resolved == latest_version);
}

TEST(schema_manifest_test, resolve_schema_version_exact_match_returns_matching_entry)
{
    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    load_schema_manifest(schema_directory(), version_file_map, latest_version);

    // latest_version is guaranteed to be a valid key in the manifest, so
    // requesting it explicitly (rather than via the {0,0,0} sentinel)
    // should resolve to itself.
    const auto resolved =
        resolve_schema_version(version_file_map, latest_version, latest_version);
    EXPECT_TRUE(resolved == latest_version);
}

TEST(schema_manifest_test, resolve_schema_version_unknown_version_throws_runtime_error)
{
    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    load_schema_manifest(schema_directory(), version_file_map, latest_version);

    EXPECT_THROW((void) resolve_schema_version(version_file_map,
                                               latest_version,
                                               profiler_hub::version_t{ 99, 0, 0 }),
                 std::runtime_error);
}

TEST(schema_manifest_test,
     load_schema_manifest_partial_version_entry_throws_runtime_error)
{
    const auto temp_dir = std::filesystem::temp_directory_path() /
                          "profiler_hub_schema_manifest_partial_version_test";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    // rocpd_tables is present, but every other required kind is missing.
    {
        std::ofstream manifest{ temp_dir / "versions.yml" };
        manifest << "rocprofiler-sdk-rocpd:\n"
                    "  rocpd_schemas:\n"
                    "    - version: \"9.9.9\"\n"
                    "      rocpd_tables: versions/9.9.9/rocpd_tables.sql\n";
    }

    version_file_map_t      version_file_map;
    profiler_hub::version_t latest_version;
    EXPECT_THROW(
        load_schema_manifest(temp_dir.string(), version_file_map, latest_version),
        std::runtime_error);

    std::filesystem::remove_all(temp_dir);
}

}  // namespace
