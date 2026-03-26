// MIT License
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc.

#pragma once

/**
 * @file schema_traits.hpp
 * @brief Schema-agnostic traits system for rocpdsna
 *
 * DESIGN GOAL: Adding a new schema version (v5, v6, etc.) should only require
 * adding ONE template specialization in this file. No other files need changes.
 *
 * HOW IT WORKS:
 * 1. schema_traits<N> defines all differences for version N
 * 2. Generic templates use traits via if constexpr
 * 3. No version-specific methods (no insert_v3_style, insert_v4_style)
 *
 * TO ADD SCHEMA v5:
 * 1. Add to schema_version enum: v5 = 5
 * 2. Add schema_traits<5> specialization below
 * 3. Done! All generic code automatically works.
 */

#include <cstdint>
#include <string>
#include <tuple>

namespace rocpdsna::data_storage
{

// ============================================================================
// Schema Version Enum - ADD NEW VERSIONS HERE
// ============================================================================

enum class schema_version : uint8_t
{
    v3 = 3,  ///< Legacy: user_name, call_stack/line_info in event
    v4 = 4,  ///< Current: track_id, timestamp tables
    // v5 = 5,  // Future: just add here and specialize schema_traits<5>
};

// ============================================================================
// Schema Traits Primary Template (must be specialized)
// ============================================================================

template<uint8_t Version>
struct schema_traits
{
    static_assert(Version != Version,
                  "schema_traits must be specialized for each version");
};

// ============================================================================
// Schema v3 Traits
// ============================================================================

template<>
struct schema_traits<3>
{
    static constexpr uint8_t     version_number = 3;
    static constexpr const char* version_string = "v3";
    static constexpr auto        version_triplet = std::make_tuple(3, 0, 0);

    struct features
    {
        // Column presence in v3
        static constexpr bool has_call_stack_column   = true;   ///< rocpd_event has call_stack JSONB
        static constexpr bool has_line_info_column    = true;   ///< rocpd_event has line_info JSONB
        static constexpr bool has_user_name_column    = true;   ///< agent_info has user_name
        static constexpr bool has_generic_name_column = false;  ///< No generic_name in v3
        
        // Timestamp/track handling
        static constexpr bool uses_direct_timestamps  = true;   ///< Direct start/end columns
        static constexpr bool uses_timestamp_tables   = false;  ///< No separate timestamp table
        static constexpr bool uses_track_id           = false;  ///< Uses nid/pid/tid directly
        static constexpr bool uses_version_triplet_api = false; ///< Uses version string
        
        // Tables that don't exist in v3
        static constexpr bool has_metadata_table      = false;  ///< No rocpd_metadata
        static constexpr bool has_timestamp_table     = false;  ///< No rocpd_timestamp
        static constexpr bool has_track_table         = false;  ///< No rocpd_track
        static constexpr bool has_category_table      = false;  ///< No rocpd_info_category
        static constexpr bool has_call_stack_table    = false;  ///< No rocpd_call_stack
        static constexpr bool has_line_info_table     = false;  ///< No rocpd_line_info
        static constexpr bool has_address_range_table = false;  ///< No rocpd_info_address_range
        static constexpr bool has_source_code_table   = false;  ///< No rocpd_info_source_code
        static constexpr bool has_pc_table            = false;  ///< No rocpd_info_pc
    };

    struct sql
    {
        static std::string timestamp_select(const std::string& alias)
        { return alias + ".start, " + alias + ".end"; }

        static std::string timestamp_join(const std::string&, const std::string&)
        { return ""; }

        static std::string track_select(const std::string& alias)
        { return alias + ".nid, " + alias + ".pid, " + alias + ".tid"; }

        static std::string track_join(const std::string&, const std::string&)
        { return ""; }

        static std::string call_stack_select(const std::string& alias)
        { return alias + ".call_stack"; }

        static std::string line_info_select(const std::string& alias)
        { return alias + ".line_info"; }

        static std::string data_insert_columns()
        { return "start, end, nid, pid, tid"; }

        static constexpr int data_insert_placeholder_count = 5;
    };

    struct binding
    {
        template<typename Stmt, typename TS>
        static void bind_timestamps(Stmt& s, int& i, const TS& start, const TS& end)
        { s.bind(i++, start.value); s.bind(i++, end.value); }

        template<typename Stmt, typename TR>
        static void bind_track(Stmt& s, int& i, const TR& t)
        { s.bind(i++, t.nid); s.bind(i++, t.pid); s.bind(i++, t.tid); }
    };
};

// ============================================================================
// Schema v4 Traits
// ============================================================================

template<>
struct schema_traits<4>
{
    static constexpr uint8_t     version_number = 4;
    static constexpr const char* version_string = "v4";
    static constexpr auto        version_triplet = std::make_tuple(0, 0, 0);

    struct features
    {
        // Column differences from v3
        static constexpr bool has_call_stack_column   = false;  ///< No call_stack in rocpd_event
        static constexpr bool has_line_info_column    = false;  ///< No line_info in rocpd_event
        static constexpr bool has_user_name_column    = false;  ///< No user_name in agent_info
        static constexpr bool has_generic_name_column = true;   ///< agent_info has generic_name
        
        // Timestamp/track handling
        static constexpr bool uses_direct_timestamps  = false;  ///< Uses FK to timestamp table
        static constexpr bool uses_timestamp_tables   = true;   ///< rocpd_timestamp table exists
        static constexpr bool uses_track_id           = true;   ///< Tables use track_id FK
        static constexpr bool uses_version_triplet_api = true;  ///< Supports (major,minor,patch)
        
        // New tables in v4
        static constexpr bool has_metadata_table      = true;   ///< rocpd_metadata table
        static constexpr bool has_timestamp_table     = true;   ///< rocpd_timestamp table
        static constexpr bool has_track_table         = true;   ///< rocpd_track table
        static constexpr bool has_category_table      = true;   ///< rocpd_info_category table
        static constexpr bool has_call_stack_table    = true;   ///< rocpd_call_stack table
        static constexpr bool has_line_info_table     = true;   ///< rocpd_line_info table
        static constexpr bool has_address_range_table = true;   ///< rocpd_info_address_range table
        static constexpr bool has_source_code_table   = true;   ///< rocpd_info_source_code table
        static constexpr bool has_pc_table            = true;   ///< rocpd_info_pc table
    };

    struct sql
    {
        static std::string timestamp_select(const std::string&)
        { return "ts_start.value AS start, ts_end.value AS end"; }

        static std::string timestamp_join(const std::string& uuid, const std::string& alias)
        {
            return " JOIN rocpd_timestamp_" + uuid + " ts_start ON " + alias + ".start_id = ts_start.id"
                   " JOIN rocpd_timestamp_" + uuid + " ts_end ON " + alias + ".end_id = ts_end.id";
        }

        static std::string track_select(const std::string&)
        { return "tr.nid, COALESCE(p.pid, 0) AS pid, COALESCE(th.tid, 0) AS tid"; }

        static std::string track_join(const std::string& uuid, const std::string& alias)
        {
            return " JOIN rocpd_track_" + uuid + " tr ON " + alias + ".track_id = tr.id"
                   " LEFT JOIN rocpd_info_process_" + uuid + " p ON tr.pid = p.id"
                   " LEFT JOIN rocpd_info_thread_" + uuid + " th ON tr.tid = th.id";
        }

        static std::string call_stack_select(const std::string&)
        { return "'' AS call_stack"; }

        static std::string line_info_select(const std::string&)
        { return "'' AS line_info"; }

        static std::string data_insert_columns()
        { return "start_id, end_id, track_id"; }

        static constexpr int data_insert_placeholder_count = 3;
    };

    struct binding
    {
        template<typename Stmt, typename TS>
        static void bind_timestamps(Stmt& s, int& i, const TS& start, const TS& end)
        { s.bind(i++, start.id); s.bind(i++, end.id); }

        template<typename Stmt, typename TR>
        static void bind_track(Stmt& s, int& i, const TR& t)
        { s.bind(i++, t.track_id); }
    };
};

// ============================================================================
// Active Schema Selection
// ============================================================================

#ifdef ROCPDSNA_USE_SCHEMA_V4
inline constexpr uint8_t active_schema_version = 4;
#else
inline constexpr uint8_t active_schema_version = 3;
#endif

using active_traits = schema_traits<active_schema_version>;

}  // namespace rocpdsna::data_storage
