// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "formats/trace_formats.hpp"
#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct ph_future;

namespace profiler_hub
{

/**
 * Copies every source into one database at @p output_path. Sources that share
 * a path are read once. A per-node table uses the window of the source whose
 * uuid matches its suffix; other tables use the first window of that file.
 */
[[nodiscard]] ph_result_t trim_trace_files(const std::vector<trim_source_t>& inputs,
                                           const std::string&                output_path,
                                           reader_types::trace_schema_t      schema,
                                           version_t                         version,
                                           ph_future*                        future);

/**
 * Copies one database into @p output_path. Every table in the file uses
 * @p start_ts and @p end_ts.
 */
[[nodiscard]] ph_result_t trim_trace_file(const std::string&             source_path,
                                          const std::string&             output_path,
                                          reader_types::trace_schema_t   schema,
                                          version_t                      version,
                                          uint64_t                       start_ts,
                                          uint64_t                       end_ts,
                                          ph_future*                     future);

}  // namespace profiler_hub
