// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/version.hpp"

#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

namespace profiler_hub::data_storage
{

/** Parses an unsigned integer that occupies the whole text. */
inline std::optional<std::uint32_t>
parse_u32(std::string_view text)
{
    if(text.empty()) return std::nullopt;

    std::uint32_t value  = 0;
    const char*   begin  = text.data();
    const char*   end    = begin + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if(ec != std::errc{} || ptr != end) return std::nullopt;
    return value;
}

/**
 * Parses an older rocpd `schema_version` value.
 * "3" is major 3 with minor and patch 0. "3.0.1" fills all three fields.
 * "3.0" is major 3, minor 0, patch 0.
 */
inline std::optional<profiler_hub::version_t>
parse_schema_version_text(std::string_view text)
{
    profiler_hub::version_t version{ 0, 0, 0 };
    std::uint32_t* const    fields[] = { &version.major, &version.minor, &version.patch };
    std::size_t             count    = 0;
    std::size_t             start    = 0;

    while(start <= text.size() && count < 3)
    {
        const std::size_t      dot  = text.find('.', start);
        const std::string_view part = text.substr(
            start, dot == std::string_view::npos ? text.size() - start : dot - start);
        const auto parsed = parse_u32(part);
        if(!parsed.has_value()) return std::nullopt;
        *fields[count] = parsed.value();
        ++count;
        if(dot == std::string_view::npos)
        {
            start = text.size();
            break;
        }
        start = dot + 1;
        if(start == text.size()) return std::nullopt;
    }
    if(count == 0 || start < text.size()) return std::nullopt;
    return version;
}

/** Which schema-version tags were present, before they are interpreted. */
struct schema_version_fields_t
{
    std::optional<std::string_view> version;
    std::optional<std::string_view> major;
    std::optional<std::string_view> minor;
    std::optional<std::string_view> patch;
};

/** Why a set of schema-version tags did or did not produce a version. */
enum class schema_version_status_t
{
    resolved,
    component_not_integer,
    text_malformed,
    incomplete,
};

/** The split schema-version tag that failed to parse as an integer. */
enum class schema_version_component_t
{
    major,
    minor,
    patch,
};

/** A schema version resolved from metadata tags. */
struct schema_version_resolution_t
{
    schema_version_status_t    status{ schema_version_status_t::incomplete };
    schema_version_component_t component{ schema_version_component_t::major };
    profiler_hub::version_t    version{ 0, 0, 0 };
};

/** SQL tag name for a split schema-version component. */
inline std::string_view
schema_version_component_name(schema_version_component_t component)
{
    switch(component)
    {
        case schema_version_component_t::major: return "schema_version_major";
        case schema_version_component_t::minor: return "schema_version_minor";
        case schema_version_component_t::patch: return "schema_version_patch";
    }
    return {};
}

/**
 * Resolves rocpd schema-version tags.
 * All three split tags win when each is an integer. Otherwise a schema_version
 * value of "3", "3.0", or "3.0.1" is used. A partial split set is incomplete,
 * and a non-integer split tag is rejected even when schema_version is present.
 */
inline schema_version_resolution_t
resolve_schema_version(const schema_version_fields_t& fields)
{
    schema_version_resolution_t result;

    const std::optional<std::string_view> components[]    = { fields.major,
                                                              fields.minor,
                                                              fields.patch };
    const schema_version_component_t      component_ids[] = {
        schema_version_component_t::major,
        schema_version_component_t::minor,
        schema_version_component_t::patch,
    };
    std::optional<std::uint32_t> parsed_components[3];

    for(std::size_t index = 0; index < 3; ++index)
    {
        if(!components[index].has_value()) continue;

        const auto parsed = parse_u32(*components[index]);
        if(!parsed.has_value())
        {
            result.status    = schema_version_status_t::component_not_integer;
            result.component = component_ids[index];
            return result;
        }
        parsed_components[index] = parsed;
    }

    if(parsed_components[0].has_value() && parsed_components[1].has_value() &&
       parsed_components[2].has_value())
    {
        result.status  = schema_version_status_t::resolved;
        result.version = { parsed_components[0].value(),
                           parsed_components[1].value(),
                           parsed_components[2].value() };
        return result;
    }

    if(fields.version.has_value())
    {
        const auto parsed = parse_schema_version_text(*fields.version);
        if(!parsed.has_value())
        {
            result.status = schema_version_status_t::text_malformed;
            return result;
        }
        result.status  = schema_version_status_t::resolved;
        result.version = parsed.value();
        return result;
    }

    result.status = schema_version_status_t::incomplete;
    return result;
}

}  // namespace profiler_hub::data_storage
