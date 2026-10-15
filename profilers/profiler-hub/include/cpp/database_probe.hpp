// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/version.hpp"

#include <string>

namespace profiler_hub
{

/** @brief Physical database container recognized by a format detector. */
enum class database_kind_t
{
    unknown,
    sqlite,
};

/** @brief Trace schema family stored inside a recognized database. */
enum class trace_schema_t
{
    unknown,
    rocpd,
};

/**
 * @brief Outcome of probing a path before a reader is constructed.
 *        Expected incompatibilities are statuses. They are not exceptions.
 */
enum class probe_status_t
{
    supported,
    file_not_found,
    unreadable,
    unknown_format,
    metadata_missing,
    metadata_malformed,
    ambiguous_trace,
    unsupported_version,
};

/** @brief What a path contains, and whether a reader can open it. */
struct database_probe_result_t
{
    probe_status_t  status{ probe_status_t::unknown_format };
    database_kind_t database_kind{ database_kind_t::unknown };
    trace_schema_t  trace_schema{ trace_schema_t::unknown };
    version_t       schema_version{ 0, 0, 0 };
    std::string     uuid;
    std::string     message;
};

/**
 * @brief Detects the database format and trace schema of @p path.
 *        Opens a recognized database read-only and does not create or modify it.
 *        Another format is added by registering a detector; another schema by
 *        registering the versions its reader accepts.
 */
[[nodiscard]] database_probe_result_t
probe_database(const std::string& path);

}  // namespace profiler_hub
