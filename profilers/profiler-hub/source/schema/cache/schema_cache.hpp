// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/version.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace profiler_hub::schema
{

struct column_model
{
    std::string              name;
    bool                     not_null    = false;
    bool                     has_default = false;
    bool                     primary_key = false;
    std::vector<std::string> allowed_values;

    // True when the caller has to set this column.
    // No input. Returns false for a primary key or a column that already has a DEFAULT.
    [[nodiscard]] bool required_from_user() const noexcept
    {
        return not_null && !has_default && !primary_key;
    }
};

struct table_model
{
    std::string                                   name;
    std::unordered_map<std::string, column_model> columns;

    // Looks up a column on this table.
    // Takes the column name. Returns the column, or null if the table does not have it.
    [[nodiscard]] const column_model* find(std::string_view column) const;
};

struct schema_model
{
    std::unordered_map<std::string, table_model> tables;

    // Looks up a table in this schema version.
    // Takes the table name without {{uuid}}. Returns the table, or null if it is absent.
    [[nodiscard]] const table_model* find_table(std::string_view table) const;
};

// One parsed copy of each schema version. The SQL lives in the library.
class schema_cache
{
public:
    // Parsed schema for one version. The embedded SQL is parsed on the first call.
    // Takes a version such as 3.0.5. Returns that model. Throws if the library has no
    // such version.
    [[nodiscard]] const schema_model& get(profiler_hub::version_t version);

    // How many versions have been parsed.
    // No input. Returns the count. A second get() of the same version does not add to it.
    [[nodiscard]] std::size_t parse_count() const;

private:
    mutable std::mutex                            m_mutex;
    std::unordered_map<std::string, schema_model> m_models;
    std::size_t                                   m_parse_count = 0;
};

// Cache used by validation. One version is parsed once for the process.
// No input. Returns that cache.
schema_cache&
shared_cache();

}  // namespace profiler_hub::schema
