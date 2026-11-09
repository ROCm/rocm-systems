// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/version.hpp"

#include <string_view>

namespace rocpd::data_storage::schema
{

struct embedded_schema_t
{
    const char* version;
    unsigned    major;
    unsigned    minor;
    unsigned    patch;
    const char* tables;
    const char* views;
    const char* data_views;
    const char* summary_views;
    const char* metadata;
    const char* indexes;
};

}  // namespace rocpd::data_storage::schema

// Generated from versions.yml. embedded_schema_t must already be declared,
// because this header defines k_embedded_schemas as an array of that type.
#include "schema/embedded_schemas.hpp"

namespace rocpd::data_storage::schema
{

/**
 * @brief Look up an embedded RocPD schema by its "MAJOR.MINOR.PATCH" string.
 * The catalog is `k_embedded_schemas`, generated from versions.yml.
 * @param version Schema version string, for example "3.0.5"
 * @return The matching catalog row, or nullptr when the version is not embedded.
 */
inline constexpr const embedded_schema_t*
find_embedded_schema(std::string_view version)
{
    for(const auto& schema : k_embedded_schemas)
    {
        if(version == schema.version)
        {
            return &schema;
        }
    }
    return nullptr;
}

/**
 * @brief The newest schema version listed in versions.yml.
 * Walks `k_embedded_schemas` and keeps the greatest major.minor.patch. This is the
 * highest embedded schema, not the newest schema the writer creates.
 * @return That version, or version_t{} when no schema is embedded.
 */
[[nodiscard]] inline profiler_hub::version_t
latest_schema_version()
{
    profiler_hub::version_t newest{};
    for(const auto& schema : k_embedded_schemas)
    {
        const profiler_hub::version_t embedded{ schema.major,
                                                schema.minor,
                                                schema.patch };
        if(newest < embedded)
        {
            newest = embedded;
        }
    }
    return newest;
}

}  // namespace rocpd::data_storage::schema
