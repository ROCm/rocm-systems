// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trace_formats.hpp"

#include "common/debug.hpp"
#include "formats/rocpd/rocpd_probe.hpp"
#include "formats/rocpd/rocpd_source.hpp"
#if PROFILER_HUB_ENABLE_ROCPD_V2
#include "formats/rocpd_v2/rocpd_v2_reader.hpp"
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V3
#include "formats/rocpd_v3/rocpd_v3_reader.hpp"
#endif
#include "reader_impl.hpp"
#include "trace_trim.hpp"
#include "trim_schema.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace profiler_hub
{
namespace
{
using reader_types::trace_format_t;
using reader_types::trace_schema_t;

bool
rocpd_supports(version_t version)
{
#if PROFILER_HUB_ENABLE_ROCPD_V3
    if(version.major == 3) return true;
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V2
    if(version.major == 2) return true;
#endif
    (void) version;
    return false;
}

std::unique_ptr<format_source>
open_rocpd_source(const source_request_t& request)
{
    return std::make_unique<rocpd_source>(request);
}

std::unique_ptr<reader_t::impl>
make_rocpd_reader(version_t                                version,
                  std::unique_ptr<profiler_hub::storage_t> storage,
                  std::shared_ptr<reader_catalog_t>        catalog)
{
#if PROFILER_HUB_ENABLE_ROCPD_V3
    if(version.major == 3)
    {
        return make_rocpd_v3_reader(std::move(storage), std::move(catalog));
    }
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V2
    if(version.major == 2)
    {
        return make_rocpd_v2_reader(std::move(storage), std::move(catalog));
    }
#endif
    (void) version;
    (void) storage;
    (void) catalog;
    return nullptr;
}

std::unique_ptr<trim_schema_t>
make_rocpd_trim(version_t version)
{
#if PROFILER_HUB_ENABLE_ROCPD_V3
    if(version.major == 3) return make_rocpd_v3_trim_schema();
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V2
    if(version.major == 2) return make_rocpd_v2_trim_schema();
#endif
    (void) version;
    return nullptr;
}

ph_result_t
trim_rocpd(const std::vector<trim_source_t>& inputs,
           const std::string&                output_path,
           version_t                         version,
           ph_future*                        future)
{
    return trim_trace_files(inputs, output_path, trace_schema_t::rocpd, version, future);
}

const format_entry_t*
find_format(trace_schema_t schema, version_t version)
{
    for(const format_entry_t& entry : format_entries())
    {
        if(entry.schema == schema && entry.supports(version)) return &entry;
    }
    return nullptr;
}
}  // namespace

std::span<const format_entry_t>
format_entries()
{
    static const format_entry_t entries[]{ { "rocpd",
                                             trace_schema_t::rocpd,
                                             rocpd_supports,
                                             rocpd_matches,
                                             rocpd_probe,
                                             open_rocpd_source,
                                             make_rocpd_reader,
                                             make_rocpd_trim,
                                             trim_rocpd } };
    return entries;
}

bool
is_supported_format(const trace_format_t& format)
{
    return find_format(format.schema, format.version) != nullptr;
}

ph_result_t
trim_in_format(trace_schema_t                    schema,
               version_t                         version,
               const std::vector<trim_source_t>& inputs,
               const std::string&                output_path,
               ph_future*                        future)
{
    const format_entry_t* entry = find_format(schema, version);
    if(entry == nullptr || entry->trim == nullptr) return PH_RESULT_UNSUPPORTED_FILE;
    return entry->trim(inputs, output_path, version, future);
}

std::unique_ptr<format_source>
open_format_source(const source_request_t& request)
{
    const format_entry_t* entry =
        find_format(request.format.schema, request.format.version);
    if(entry == nullptr)
    {
        throw std::invalid_argument{ "no format reads trace schema version " +
                                     std::to_string(request.format.version.major) };
    }
    LOG_INFO("[Profiler-Hub] Opening {} {}.{}.{} trace {}",
             entry->name,
             request.format.version.major,
             request.format.version.minor,
             request.format.version.patch,
             request.path);
    return entry->open(request);
}

std::unique_ptr<reader_t::impl>
make_reader_impl(const trace_format_t&                    format,
                 std::unique_ptr<profiler_hub::storage_t> storage,
                 std::shared_ptr<reader_catalog_t>        catalog)
{
    const format_entry_t* entry = find_format(format.schema, format.version);
    if(entry == nullptr)
    {
        throw std::invalid_argument{ "no reader for trace schema version " +
                                     std::to_string(format.version.major) };
    }
    return entry->make_reader(format.version, std::move(storage), std::move(catalog));
}

std::unique_ptr<trim_schema_t>
make_trim_schema(reader_types::trace_schema_t schema, version_t version)
{
    const format_entry_t* entry = find_format(schema, version);
    if(entry == nullptr) return nullptr;
    return entry->make_trim(version);
}

}  // namespace profiler_hub
