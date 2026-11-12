// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"

#include <memory>
#include <span>
#include <string>

namespace profiler_hub
{
class format_source;
class storage_t;
class trim_schema_t;
struct reader_catalog_t;
struct source_request_t;

/**
 * @brief One readable trace format in the registry.
 *
 * An entry recognises its own files from their first bytes, probes them for
 * version and GUIDs, and says which versions this build reads. The probe walks
 * the entries in order and the first one that recognises a file wins. Once a
 * file is accepted, its entry opens the file's source, the rocpd source's
 * connections build their reader from the entry, and the trimmer takes its row
 * filter from it. A new format adds an entry in
 * trace_formats.cpp, and a PROFILER_HUB_ENABLE_<FORMAT> CMake option when its
 * sources can be left out. The rocpd entry is always built: with both rocpd
 * version options off it supports no version, and the probe reports a rocpd
 * file as unsupported.
 */
struct format_entry_t
{
    const char*                  name;  ///< Cache folder label and log text.
    reader_types::trace_schema_t schema;
    bool (*supports)(version_t version);
    bool (*matches)(const std::string& path);
    reader_types::file_probe_result_t (*probe)(const std::string& path);
    std::unique_ptr<format_source> (*open)(const source_request_t& request);
    std::unique_ptr<reader_t::impl> (*make_reader)(
        version_t                                version,
        std::unique_ptr<profiler_hub::storage_t> storage,
        std::shared_ptr<reader_catalog_t>        catalog);
    std::unique_ptr<trim_schema_t> (*make_trim)(version_t version);
};

/** @brief The entries in registry order. */
[[nodiscard]] std::span<const format_entry_t>
format_entries();

/** @brief True when an entry of @p format's schema reads its version. */
[[nodiscard]] bool is_supported_format(const reader_types::trace_format_t& format);

/** @brief Opens @p request's file with the entry that reads its format. Throws
 *         std::invalid_argument when no entry does. */
[[nodiscard]] std::unique_ptr<format_source>
open_format_source(const source_request_t& request);

/** @brief Reader backend for @p format. Throws std::invalid_argument when no
 *         entry reads it. */
[[nodiscard]] std::unique_ptr<reader_t::impl> make_reader_impl(
    const reader_types::trace_format_t&      format,
    std::unique_ptr<profiler_hub::storage_t> storage,
    std::shared_ptr<reader_catalog_t>        catalog);

}  // namespace profiler_hub
