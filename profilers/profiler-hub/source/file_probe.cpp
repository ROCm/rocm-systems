// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/cpp/reader.hpp"

#include "common/debug.hpp"
#include "formats/trace_formats.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

namespace profiler_hub
{

using reader_types::file_probe_result_t;
using reader_types::trace_schema_t;

namespace
{

file_probe_result_t
result_of(ph_result_t    status,
          std::string    message,
          trace_schema_t schema  = trace_schema_t::unknown,
          version_t      version = {},
          std::string    uuid    = {})
{
    file_probe_result_t result;
    result.status         = status;
    result.trace_schema   = schema;
    result.schema_version = version;
    result.uuid           = std::move(uuid);
    result.message        = std::move(message);
    return result;
}

}  // namespace

file_probe_result_t
probe_file(const std::string& path)
{
    std::error_code error;
    if(path.empty() || !std::filesystem::exists(path, error) || error)
    {
        return result_of(PH_RESULT_INVALID_FILE, "trace file does not exist");
    }
    if(!std::filesystem::is_regular_file(path, error) || error)
    {
        return result_of(PH_RESULT_UNSUPPORTED_FILE, "trace path is not a file");
    }

    std::ifstream input{ path, std::ios::binary };
    if(!input)
    {
        return result_of(PH_RESULT_INVALID_FILE, "trace file could not be read");
    }

    for(const format_entry_t& entry : format_entries())
    {
        if(entry.matches(path))
        {
            LOG_DEBUG("probe_file: '{}' matched format '{}'", path, entry.name);
            return entry.probe(path);
        }
    }
    return result_of(PH_RESULT_UNSUPPORTED_FILE, "trace file is not a supported format");
}

}  // namespace profiler_hub
