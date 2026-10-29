// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstdint>
#include <string>

struct ph_future;

namespace profiler_hub
{

/**
 * Copies one database into @p output_path, keeping rows selected by the trim
 * schema for @p schema and @p version. @p start_ts and @p end_ts are absolute.
 */
[[nodiscard]] ph_result_t trim_trace_file(const std::string&             source_path,
                                          const std::string&             output_path,
                                          reader_types::trace_schema_t   schema,
                                          version_t                      version,
                                          uint64_t                       start_ts,
                                          uint64_t                       end_ts,
                                          ph_future*                     future);

}  // namespace profiler_hub
