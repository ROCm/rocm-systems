// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include <string>

namespace profiler_hub
{

/** @brief True when the file starts with the SQLite header, as every rocpd trace does. */
[[nodiscard]] bool
rocpd_matches(const std::string& path);

/** @brief Reports the rocpd schema, version and GUIDs of a SQLite file. */
[[nodiscard]] reader_types::file_probe_result_t
rocpd_probe(const std::string& path);

}  // namespace profiler_hub
