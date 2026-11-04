// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trace_formats.hpp"

#if PROFILER_HUB_ENABLE_ROCPD_V2
#include "formats/rocpd_v2/rocpd_v2_reader.hpp"
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V3
#include "formats/rocpd_v3/rocpd_v3_reader.hpp"
#endif
#include "reader_impl.hpp"
#include "trim_schema.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace profiler_hub
{
namespace
{
using reader_types::trace_format_t;
using reader_types::trace_schema_t;

struct format_entry_t
{
    trace_schema_t schema;
    uint32_t       major;
    std::unique_ptr<reader_t::impl> (*make_reader)(std::unique_ptr<storage_t> storage,
                                                   std::shared_ptr<reader_catalog_t> catalog);
    std::unique_ptr<trim_schema_t> (*make_trim)();
};

const format_entry_t*
find_format(trace_schema_t schema, version_t version)
{
#if PROFILER_HUB_ENABLE_ROCPD_V3
    static const format_entry_t rocpd_v3{ trace_schema_t::rocpd, 3, make_rocpd_v3_reader,
                                          make_rocpd_v3_trim_schema };
    if(schema == rocpd_v3.schema && version.major == rocpd_v3.major) return &rocpd_v3;
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V2
    static const format_entry_t rocpd_v2{ trace_schema_t::rocpd, 2, make_rocpd_v2_reader,
                                          make_rocpd_v2_trim_schema };
    if(schema == rocpd_v2.schema && version.major == rocpd_v2.major) return &rocpd_v2;
#endif
    (void) schema;
    (void) version;
    return nullptr;
}
}  // namespace

bool
is_supported_format(const trace_format_t& format)
{
    return find_format(format.schema, format.version) != nullptr;
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
    return entry->make_reader(std::move(storage), std::move(catalog));
}

std::unique_ptr<trim_schema_t>
make_trim_schema(reader_types::trace_schema_t schema, version_t version)
{
    const format_entry_t* entry = find_format(schema, version);
    if(entry == nullptr) return nullptr;
    return entry->make_trim();
}

}  // namespace profiler_hub
