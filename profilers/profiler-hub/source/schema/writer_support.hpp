// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/version.hpp"

#include <string>

namespace profiler_hub::schema
{

// The schema_v3 writer inserts the 3.0.0 / 3.0.1 column set.
// 3.0.2 added graph columns the writer does not write.
inline bool
writer_supports_schema(profiler_hub::version_t version) noexcept
{
    return version == profiler_hub::version_t{ 3, 0, 0 } ||
           version == profiler_hub::version_t{ 3, 0, 1 };
}

inline std::string
unsupported_writer_reason(profiler_hub::version_t version)
{
    std::string reason =
        "No writer is supported\nfor schema version " + version.to_string();
    if(version.is_latest())
    {
        reason += " (latest).\nThe latest schema has no writer.";
    }
    else
    {
        reason += ".";
    }
    reason += "\nSupported writer versions:\n3.0.0\n3.0.1";
    return reason;
}

}  // namespace profiler_hub::schema
