#pragma once

#include "profiler-hub/version.hpp"

// Include all the generated schema header files.
#include "schema/data_views.hpp"
#include "schema/marker_views.hpp"
#include "schema/rocpd_indexes.hpp"
#include "schema/rocpd_metadata.hpp"
#include "schema/rocpd_tables.hpp"
#include "schema/rocpd_views.hpp"
#include "schema/summary_views.hpp"
// version 3.0.0
#include "schema/versions/3.0.0/data_views.hpp"
#include "schema/versions/3.0.0/rocpd_indexes.hpp"
#include "schema/versions/3.0.0/rocpd_metadata.hpp"
#include "schema/versions/3.0.0/rocpd_tables.hpp"
#include "schema/versions/3.0.0/rocpd_views.hpp"
#include "schema/versions/3.0.0/summary_views.hpp"
// version 3.0.1
#include "schema/versions/3.0.1/data_views.hpp"
#include "schema/versions/3.0.1/rocpd_indexes.hpp"
#include "schema/versions/3.0.1/rocpd_metadata.hpp"
#include "schema/versions/3.0.1/rocpd_tables.hpp"
#include "schema/versions/3.0.1/rocpd_views.hpp"
#include "schema/versions/3.0.1/summary_views.hpp"
// version 3.0.2
#include "schema/versions/3.0.2/data_views.hpp"
#include "schema/versions/3.0.2/rocpd_indexes.hpp"
#include "schema/versions/3.0.2/rocpd_metadata.hpp"
#include "schema/versions/3.0.2/rocpd_tables.hpp"
#include "schema/versions/3.0.2/rocpd_views.hpp"
#include "schema/versions/3.0.2/summary_views.hpp"

#include <array>
#include <vector>

namespace profiler_hub::data_storage
{

/**
 * Basenames of the SQL files that make up a rocpd schema version, in the
 * order they must be executed: tables/indexes/views before the metadata insert
 * that depends on the rocpd_metadata table already existing. schema_version_entry_t::sql
 * below is positioned to match this same order.
 */
inline constexpr std::array<const char*, 6> SCHEMA_KIND_BASENAMES = {
    "rocpd_tables", "rocpd_indexes", "rocpd_views",
    "data_views",   "summary_views", "rocpd_metadata",
};

/**
 * One compiled-in rocpd schema version: its version number, plus a direct pointer
 * to each kind's embedded SQL text (indexed to match SCHEMA_KIND_BASENAMES, e.g.
 * sql[0] is always "rocpd_tables"). No symbol-name strings or lookups involved --
 * these are the actual compiled-in constants from the #includes above.
 */
struct schema_version_entry_t
{
    profiler_hub::version_t                               version;
    std::array<const char*, SCHEMA_KIND_BASENAMES.size()> sql;
};

/**
 * Every rocpd schema version compiled into this binary, fully resolved at compile
 * time -- no path/symbol-name derivation or lookups needed at runtime.
 *
 * Adding a new schema version? Add its #includes above, then add one entry here
 * (in SCHEMA_KIND_BASENAMES order: tables, indexes, views, data_views, summary_views,
 * metadata).
 */
inline const std::vector<schema_version_entry_t>&
known_schema_versions()
{
    using namespace ::rocpd::data_storage::schema;

    static const std::vector<schema_version_entry_t> versions = {
        { { 3, 0, 0 },
          { VERSIONS_3_0_0_ROCPD_TABLES_SQL,
            VERSIONS_3_0_0_ROCPD_INDEXES_SQL,
            VERSIONS_3_0_0_ROCPD_VIEWS_SQL,
            VERSIONS_3_0_0_DATA_VIEWS_SQL,
            VERSIONS_3_0_0_SUMMARY_VIEWS_SQL,
            VERSIONS_3_0_0_ROCPD_METADATA_SQL } },
        { { 3, 0, 1 },
          { VERSIONS_3_0_1_ROCPD_TABLES_SQL,
            VERSIONS_3_0_1_ROCPD_INDEXES_SQL,
            VERSIONS_3_0_1_ROCPD_VIEWS_SQL,
            VERSIONS_3_0_1_DATA_VIEWS_SQL,
            VERSIONS_3_0_1_SUMMARY_VIEWS_SQL,
            VERSIONS_3_0_1_ROCPD_METADATA_SQL } },
        { { 3, 0, 2 },
          { VERSIONS_3_0_2_ROCPD_TABLES_SQL,
            VERSIONS_3_0_2_ROCPD_INDEXES_SQL,
            VERSIONS_3_0_2_ROCPD_VIEWS_SQL,
            VERSIONS_3_0_2_DATA_VIEWS_SQL,
            VERSIONS_3_0_2_SUMMARY_VIEWS_SQL,
            VERSIONS_3_0_2_ROCPD_METADATA_SQL } },
        { { 3, 0, 3 },
          // unversioned/top-level schema files are the latest revision always(3.0.3)
          { ROCPD_TABLES_SQL,
            ROCPD_INDEXES_SQL,
            ROCPD_VIEWS_SQL,
            DATA_VIEWS_SQL,
            SUMMARY_VIEWS_SQL,
            ROCPD_METADATA_SQL } },
    };
    return versions;
}

}  // namespace profiler_hub::data_storage
