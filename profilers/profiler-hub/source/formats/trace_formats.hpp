// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"

#include <memory>

namespace profiler_hub
{
class storage_t;
struct reader_catalog_t;

/**
 * @brief Registry of readable trace formats.
 *
 * One entry per schema family and major version. The probe accepts a file only
 * when an entry exists, connections build their reader from it, and the
 * trimmer takes its row filter from it. A new format adds a
 * PROFILER_HUB_ENABLE_<FORMAT> CMake option, a detector in file_probe.cpp,
 * and an entry in trace_formats.cpp. The option defaults to ON. Turning it
 * off leaves that format's sources out of the build, and the probe reports
 * the file as unsupported.
 */
[[nodiscard]] bool is_supported_format(const reader_types::trace_format_t& format);

/** @brief Reader backend for @p format. Throws std::invalid_argument when no
 *         entry reads it. */
[[nodiscard]] std::unique_ptr<reader_t::impl> make_reader_impl(
    const reader_types::trace_format_t&      format,
    std::unique_ptr<profiler_hub::storage_t> storage,
    std::shared_ptr<reader_catalog_t>        catalog);

}  // namespace profiler_hub
