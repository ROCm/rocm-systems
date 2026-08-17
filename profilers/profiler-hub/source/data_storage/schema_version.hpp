// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace profiler_hub::data_storage
{

struct schema_v3_tag
{};

/**
 * Identifies a rocpd SQL schema version (as published in
 * share/profiler-hub/schema/versions.yml).
 *
 * The default value {0, 0, 0} is a sentinel meaning "latest available
 * schema" -- see resolve_schema_version() in schema_manifest.hpp.
 */
struct schema_version_t
{
    unsigned major = 0;
    unsigned minor = 0;
    unsigned patch = 0;

    [[nodiscard]] bool is_latest() const noexcept
    {
        return major == 0 && minor == 0 && patch == 0;
    }

    [[nodiscard]] std::string to_string() const
    {
        return std::to_string(major) + "." + std::to_string(minor) + "." +
               std::to_string(patch);
    }

    friend bool operator==(const schema_version_t& lhs,
                           const schema_version_t& rhs) noexcept
    {
        return lhs.major == rhs.major && lhs.minor == rhs.minor && lhs.patch == rhs.patch;
    }

    friend bool operator!=(const schema_version_t& lhs,
                           const schema_version_t& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    friend bool operator<(const schema_version_t& lhs,
                          const schema_version_t& rhs) noexcept
    {
        if(lhs.major != rhs.major) return lhs.major < rhs.major;
        if(lhs.minor != rhs.minor) return lhs.minor < rhs.minor;
        return lhs.patch < rhs.patch;
    }
};

}  // namespace profiler_hub::data_storage
