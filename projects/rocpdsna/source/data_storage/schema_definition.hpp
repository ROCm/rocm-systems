// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file schema_definition.hpp
 * @brief Comprehensive table and column definitions for rocpd schema versions
 *
 * This file provides compile-time constants for all table names, column names,
 * and their version-specific presence. This enables:
 * - Version-agnostic query generation
 * - Compile-time validation of column access
 * - Easy addition of new schema versions
 *
 * SCHEMA VERSIONS:
 * - v3: Legacy schema with call_stack/line_info JSONB, user_name column
 * - v4+: Current schema with separate tables, track_id, timestamp tables
 */

#include <string_view>
#include <cstdint>

namespace rocpdsna::data_storage::schema_def
{

// =============================================================================
// Version Range Constants
// =============================================================================

constexpr uint8_t V3_ONLY = 3;      ///< Only in v3
constexpr uint8_t V4_PLUS = 4;      ///< v4 and later
constexpr uint8_t ALL_VERSIONS = 0; ///< All versions

// =============================================================================
// Table Names
// =============================================================================

namespace tables
{
    // Info tables (all versions)
    constexpr std::string_view node_info        = "rocpd_info_node";
    constexpr std::string_view process_info     = "rocpd_info_process";
    constexpr std::string_view thread_info      = "rocpd_info_thread";
    constexpr std::string_view agent_info       = "rocpd_info_agent";
    constexpr std::string_view queue_info       = "rocpd_info_queue";
    constexpr std::string_view stream_info      = "rocpd_info_stream";
    constexpr std::string_view code_object_info = "rocpd_info_code_object";
    constexpr std::string_view kernel_symbol    = "rocpd_info_kernel_symbol";
    constexpr std::string_view pmc_info         = "rocpd_info_pmc";
    constexpr std::string_view string_info      = "rocpd_string";

    // Data tables (all versions)
    constexpr std::string_view event            = "rocpd_event";
    constexpr std::string_view arg              = "rocpd_arg";
    constexpr std::string_view region           = "rocpd_region";
    constexpr std::string_view sample           = "rocpd_sample";
    constexpr std::string_view pmc_event        = "rocpd_pmc_event";
    constexpr std::string_view kernel_dispatch  = "rocpd_kernel_dispatch";
    constexpr std::string_view memory_copy      = "rocpd_memory_copy";
    constexpr std::string_view memory_alloc     = "rocpd_memory_allocate";

    // v4+ new tables
    constexpr std::string_view metadata         = "rocpd_metadata";         // v4+
    constexpr std::string_view timestamp        = "rocpd_timestamp";        // v4+
    constexpr std::string_view track            = "rocpd_track";            // v4+
    constexpr std::string_view category_info    = "rocpd_info_category";    // v4+
    constexpr std::string_view call_stack       = "rocpd_call_stack";       // v4+
    constexpr std::string_view line_info        = "rocpd_line_info";        // v4+
    constexpr std::string_view address_range    = "rocpd_info_address_range"; // v4+
    constexpr std::string_view source_code      = "rocpd_info_source_code"; // v4+
    constexpr std::string_view pc_info          = "rocpd_info_pc";          // v4+
}

// =============================================================================
// Column Definitions - rocpd_event
// =============================================================================

namespace event_columns
{
    // Common columns (all versions)
    constexpr std::string_view id             = "id";
    constexpr std::string_view stack_id       = "stack_id";
    constexpr std::string_view parent_id      = "parent_stack_id";
    constexpr std::string_view correlation_id = "correlation_id";
    constexpr std::string_view extdata        = "extdata";

    // v3 only columns
    constexpr std::string_view call_stack     = "call_stack";   // v3: JSONB
    constexpr std::string_view line_info      = "line_info";    // v3: JSONB

    // v4+ columns
    constexpr std::string_view category_id    = "category_id";  // v4+: FK to category table
}

// =============================================================================
// Column Definitions - rocpd_info_agent
// =============================================================================

namespace agent_columns
{
    // Common columns (all versions)
    constexpr std::string_view id              = "id";
    constexpr std::string_view nid             = "nid";
    constexpr std::string_view pid             = "pid";
    constexpr std::string_view agent_type      = "agent_type";
    constexpr std::string_view absolute_index  = "absolute_index";
    constexpr std::string_view logical_index   = "logical_index";
    constexpr std::string_view type_index      = "type_index";
    constexpr std::string_view uuid            = "uuid";
    constexpr std::string_view name            = "name";
    constexpr std::string_view model_name      = "model_name";
    constexpr std::string_view vendor_name     = "vendor_name";
    constexpr std::string_view product_name    = "product_name";
    constexpr std::string_view extdata         = "extdata";

    // v3 only
    constexpr std::string_view user_name       = "user_name";    // v3 only

    // v4+ only
    constexpr std::string_view generic_name    = "generic_name"; // v4+
}

// =============================================================================
// Column Definitions - Data Tables (region, kernel_dispatch, etc.)
// =============================================================================

namespace data_columns
{
    // v3: Direct timestamp columns
    constexpr std::string_view start           = "start";        // v3: BIGINT
    constexpr std::string_view end             = "end";          // v3: BIGINT
    constexpr std::string_view nid             = "nid";          // v3: direct FK
    constexpr std::string_view pid             = "pid";          // v3: direct FK
    constexpr std::string_view tid             = "tid";          // v3: direct FK

    // v4+: FK to timestamp and track tables
    constexpr std::string_view start_id        = "start_id";     // v4+: FK to timestamp
    constexpr std::string_view end_id          = "end_id";       // v4+: FK to timestamp
    constexpr std::string_view track_id        = "track_id";     // v4+: FK to track
    constexpr std::string_view event_id        = "event_id";     // v4+: FK to event
}

// =============================================================================
// Column Definitions - v4+ New Tables
// =============================================================================

namespace timestamp_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view value           = "value";
    constexpr std::string_view phase           = "phase";
    constexpr std::string_view track_id        = "track_id";
}

namespace track_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view guid            = "guid";
    constexpr std::string_view nid             = "nid";
    constexpr std::string_view ppid            = "ppid";
    constexpr std::string_view pid             = "pid";
    constexpr std::string_view tid             = "tid";
    constexpr std::string_view agent_id        = "agent_id";
    constexpr std::string_view queue_id        = "queue_id";
    constexpr std::string_view stream_id       = "stream_id";
    constexpr std::string_view name_id         = "name_id";
    constexpr std::string_view extdata         = "extdata";
}

namespace category_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view name            = "name";
    constexpr std::string_view extdata         = "extdata";
}

namespace address_range_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view nid             = "nid";
    constexpr std::string_view pid             = "pid";
    constexpr std::string_view address_base    = "address_base";
    constexpr std::string_view address_low     = "address_low";
    constexpr std::string_view address_high    = "address_high";
    constexpr std::string_view extdata         = "extdata";
}

namespace source_code_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view nid             = "nid";
    constexpr std::string_view pid             = "pid";
    constexpr std::string_view address_id      = "address_id";
    constexpr std::string_view file            = "file";
    constexpr std::string_view line_number     = "line_number";
    constexpr std::string_view lines           = "lines";
    constexpr std::string_view instructions    = "instructions";
    constexpr std::string_view extdata         = "extdata";
}

namespace pc_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view nid             = "nid";
    constexpr std::string_view pid             = "pid";
    constexpr std::string_view function        = "function";
    constexpr std::string_view address_id      = "address_id";
    constexpr std::string_view file            = "file";
    constexpr std::string_view line            = "line";
    constexpr std::string_view extdata         = "extdata";
}

namespace call_stack_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view event_id        = "event_id";
    constexpr std::string_view pc_id           = "pc_id";
    constexpr std::string_view depth           = "depth";
    constexpr std::string_view extdata         = "extdata";
}

namespace line_info_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view event_id        = "event_id";
    constexpr std::string_view source_code_id  = "source_code_id";
    constexpr std::string_view pc_id           = "pc_id";
    constexpr std::string_view extdata         = "extdata";
}

namespace metadata_columns
{
    constexpr std::string_view id              = "id";
    constexpr std::string_view tag             = "tag";
    constexpr std::string_view value           = "value";
}

}  // namespace rocpdsna::data_storage::schema_def
