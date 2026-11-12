// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct ph_future;

namespace profiler_hub
{
class format_source;
class storage_t;
class trim_schema_t;
struct reader_catalog_t;
struct source_request_t;

/** One file and GUID, with the absolute time window that GUID keeps. */
struct trim_source_t
{
    std::string path;
    std::string uuid;
    uint64_t    start_ts{};
    uint64_t    end_ts{};
};

/**
 * @brief One readable trace format in the registry.
 *
 * An entry recognises its own files from their first bytes, probes them for
 * version and GUIDs, and says which versions this build reads. The probe walks
 * the entries in order and the first one that recognises a file wins. Once a
 * file is accepted, its entry opens the file's source and the rocpd source's
 * connections build their reader from the entry. Trimming goes through the
 * entry's trim, which is null for a format that cannot trim; the rocpd trim
 * takes its row filter from make_trim. A new format adds an entry in
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
    ph_result_t (*trim)(const std::vector<trim_source_t>& inputs,
                        const std::string&                output_path,
                        version_t                         version,
                        ph_future*                        future);
};

/** @brief The entries in registry order. */
[[nodiscard]] std::span<const format_entry_t>
format_entries();

/** @brief True when an entry of @p format's schema reads its version. */
[[nodiscard]] bool is_supported_format(const reader_types::trace_format_t& format);

/** @brief Trims @p inputs into one database with the entry that reads
 *         @p schema at @p version. Returns PH_RESULT_UNSUPPORTED_FILE when no
 *         entry does or its trim is null. */
[[nodiscard]] ph_result_t
trim_in_format(reader_types::trace_schema_t      schema,
               version_t                         version,
               const std::vector<trim_source_t>& inputs,
               const std::string&                output_path,
               ph_future*                        future);

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
