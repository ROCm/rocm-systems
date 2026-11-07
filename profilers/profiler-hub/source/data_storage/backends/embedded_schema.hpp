// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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

}  // namespace rocpd::data_storage::schema
