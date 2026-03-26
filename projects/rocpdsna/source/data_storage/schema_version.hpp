// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocpdsna::data_storage
{

// =============================================================================
// Schema Version Tags (compile-time selection)
// =============================================================================

/// @brief Tag type for schema v3
struct schema_v3_tag
{};

/// @brief Tag type for schema v4 (latest schema )
struct schema_v4_tag
{};

// =============================================================================
// Schema Version Enum (runtime selection)
// =============================================================================

/// @brief Runtime schema version identifier
enum class schema_version_t : uint8_t
{
    unknown = 0,
    v3      = 3,  // Schema 3.0.0: has user_name, call_stack, line_info, direct timestamps
    v4      = 4   // Schema 4.0.0+: has track_id, timestamp tables, no call_stack/line_info
};

// =============================================================================
// Schema Version Info Structure
// =============================================================================

/// @brief Contains schema version information and feature flags
struct schema_version_info_t
{
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;

    // ==========================================================================
    // Feature flags for schema v3 vs v4 differences
    // ==========================================================================

    // rocpd_event table differences
    bool has_call_stack_column = true;   // rocpd_event has call_stack JSONB column
    bool has_line_info_column  = true;   // rocpd_event has line_info JSONB column

    // rocpd_info_agent table differences
    bool has_user_name_column    = true;   // agent_info has user_name column
    bool has_generic_name_column = false;  // agent_info has generic_name column (v4)

    // rocpd_region/kernel_dispatch/memory_copy/memory_alloc differences
    bool uses_direct_timestamps = true;    // Tables have start/end BIGINT columns
    bool uses_timestamp_tables  = false;   // Tables reference rocpd_timestamp table
    bool uses_track_id          = false;   // Tables use track_id instead of nid/pid/tid

    // New tables in v4+
    bool has_metadata_table      = false;  // rocpd_metadata table exists
    bool has_timestamp_table     = false;  // rocpd_timestamp table exists
    bool has_track_table         = false;  // rocpd_track table exists
    bool has_category_table      = false;  // rocpd_info_category table exists
    bool has_call_stack_table    = false;  // rocpd_call_stack table exists
    bool has_line_info_table     = false;  // rocpd_line_info table exists
    bool has_address_range_table = false;  // rocpd_info_address_range table exists
    bool has_source_code_table   = false;  // rocpd_info_source_code table exists
    bool has_pc_table            = false;  // rocpd_info_pc table exists

    /// @brief Get the schema version enum from feature flags
    [[nodiscard]] schema_version_t get_schema_version() const noexcept
    {
        // Schema v4+ uses timestamp tables and track_id
        if (uses_timestamp_tables && uses_track_id && !has_call_stack_column)
        {
            return schema_version_t::v4;
        }
        // Default to v3 for backwards compatibility
        return schema_version_t::v3;
    }

    /// @brief Create info for schema v3.0.0
    static schema_version_info_t v3()
    {
        schema_version_info_t info;
        info.major = 3;
        info.minor = 0;
        info.patch = 0;
        // v3 defaults are already set
        return info;
    }

    /// @brief Create info for latest schema (v4+)
    static schema_version_info_t v4()
    {
        schema_version_info_t info;
        info.major = 4;
        info.minor = 0;
        info.patch = 0;

        // rocpd_event differences
        info.has_call_stack_column = false;
        info.has_line_info_column  = false;

        // rocpd_info_agent differences
        info.has_user_name_column    = false;
        info.has_generic_name_column = true;

        // Timestamp/track differences
        info.uses_direct_timestamps = false;
        info.uses_timestamp_tables  = true;
        info.uses_track_id          = true;

        // New tables in v4+
        info.has_metadata_table      = true;
        info.has_timestamp_table     = true;
        info.has_track_table         = true;
        info.has_category_table      = true;
        info.has_call_stack_table    = true;
        info.has_line_info_table     = true;
        info.has_address_range_table = true;
        info.has_source_code_table   = true;
        info.has_pc_table            = true;

        return info;
    }
};

// =============================================================================
// Schema Version Detection
// =============================================================================

/// @brief Get schema info based on requested version triplet
/// @param major Major version (0 = latest)
/// @param minor Minor version
/// @param patch Patch version
/// @return Schema version info with feature flags
inline schema_version_info_t get_schema_info(uint32_t major, uint32_t minor, uint32_t patch)
{
    // {0, 0, 0} means latest schema
    if (major == 0 && minor == 0 && patch == 0)
    {
        return schema_version_info_t::v4();
    }

    // Explicit v3.x.x
    if (major == 3)
    {
        return schema_version_info_t::v3();
    }

    // v4+ uses latest schema
    if (major >= 4)
    {
        return schema_version_info_t::v4();
    }

    // Default to v3 for unknown versions
    return schema_version_info_t::v3();
}

/// @brief Check if the schema version supports user_name column
inline bool schema_has_user_name_column(schema_version_t version) noexcept
{
    return version == schema_version_t::v3;
}

/// @brief Check if the schema uses direct timestamps (start/end columns)
inline bool schema_uses_direct_timestamps(schema_version_t version) noexcept
{
    return version == schema_version_t::v3;
}

/// @brief Check if the schema uses track_id
inline bool schema_uses_track_id(schema_version_t version) noexcept
{
    return version == schema_version_t::v4;
}

/// @brief Check if the schema has call_stack column in rocpd_event
inline bool schema_has_call_stack_column(schema_version_t version) noexcept
{
    return version == schema_version_t::v3;
}

/// @brief Convert schema version to string
inline std::string to_string(schema_version_t version)
{
    switch (version)
    {
        case schema_version_t::v3: return "v3";
        case schema_version_t::v4: return "v4";
        default: return "unknown";
    }
}

}  // namespace rocpdsna::data_storage
