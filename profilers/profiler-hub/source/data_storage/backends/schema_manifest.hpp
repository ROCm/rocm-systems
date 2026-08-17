// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "data_storage/schema_version.hpp"

#include <string>
#include <unordered_map>

namespace profiler_hub::data_storage
{

/**
 * schema kind name (e.g. "rocpd_tables") -> relative SQL file path for one
 * schema version. Mirrors rocpd::sql::kind_filename_map_t in
 * rocprofiler-sdk-rocpd's sql.cpp.
 */
using kind_filename_map_t = std::unordered_map<std::string, std::string>;

/**
 * schema version string (e.g. "3.0.1") -> kind_filename_map_t for that
 * version. Mirrors rocpd::sql::version_file_map_t in
 * rocprofiler-sdk-rocpd's sql.cpp.
 */
using version_file_map_t = std::unordered_map<std::string, kind_filename_map_t>;

/**
 * Reads versions.yml and fills the version -> schema file map.
 * latest_version is set to the highest manifest version found.
 * Throws if the manifest is missing, invalid, or empty.
 */
void
load_schema_manifest(const std::string&  schema_dir,
                     version_file_map_t& version_file_map,
                     schema_version_t&   latest_version);

/**
 * Resolves a requested schema version against a loaded manifest.
 * The "latest" sentinel resolves to latest_version; explicit versions must match.
 */
[[nodiscard]] schema_version_t
resolve_schema_version(const version_file_map_t& version_file_map,
                       const schema_version_t&   latest_version,
                       const schema_version_t&   requested);

}  // namespace profiler_hub::data_storage
