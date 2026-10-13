// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "storage_impl.hpp"
#include "debug.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"

#include "data_storage/backends/sqlite_backend.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace profiler_hub
{

namespace
{
struct metadata_row_t
{
    std::string tag;
    std::string value;
};

constexpr std::array<std::string_view, 3> version_tags{ "schema_version_major",
                                                        "schema_version_minor",
                                                        "schema_version_patch" };

[[nodiscard]] std::uint32_t
parse_version_component(std::string_view text)
{
    std::uint32_t value{};
    const auto*   end      = text.data() + text.size();
    const auto [ptr, errc] = std::from_chars(text.data(), end, value);
    if(errc != std::errc{} || ptr != end)
    {
        throw std::runtime_error("invalid schema version component '" +
                                 std::string{ text } + "'");
    }
    return value;
}

[[nodiscard]] profiler_hub::version_t
assemble_version(const std::vector<metadata_row_t>& rows)
{
    std::array<std::optional<std::uint32_t>, version_tags.size()> components;
    for(const auto& row : rows)
    {
        const auto tag = std::ranges::find(version_tags, std::string_view{ row.tag });
        if(tag != version_tags.end())
        {
            components[static_cast<size_t>(tag - version_tags.begin())] =
                parse_version_component(row.value);
        }
    }

    if(!std::ranges::all_of(components, [](const auto& c) { return c.has_value(); }))
    {
        throw std::runtime_error("incomplete schema version in metadata");
    }
    return { .major = *components[0], .minor = *components[1], .patch = *components[2] };
}

[[nodiscard]] std::vector<metadata_row_t>
read_metadata_rows(data_storage::sqlite_backend& backend)
{
    const auto query = fmt::format("SELECT tag, value FROM \"rocpd_metadata_{}\" "
                                   "WHERE tag IN ('{}', '{}', '{}')",
                                   backend.get_uuid(),
                                   version_tags[0],
                                   version_tags[1],
                                   version_tags[2]);

    const auto executor = backend.create_read_statement_executor<metadata_row_t>(
        query, &metadata_row_t::tag, &metadata_row_t::value);
    return executor().to_vector();
}
}  // namespace

struct storage_t::impl::database_factory_t
{
    static std::shared_ptr<data_storage::sqlite_backend> create_database(
        const std::string&    database_path,
        const std::string&    uuid,
        const storage_type_t& storage_type)
    {
        switch(storage_type)
        {
            case storage_type_t::read:
                return data_storage::sqlite_backend::create(
                    database_path,
                    uuid,
                    data_storage::sqlite_backend::storage_mode_t::on_disk);
            case storage_type_t::write:
                return data_storage::sqlite_backend::create(
                    database_path,
                    uuid,
                    data_storage::sqlite_backend::storage_mode_t::in_memory);
            default:
                throw std::invalid_argument(
                    "Invalid storage type: " +
                    std::to_string(static_cast<int>(storage_type)));
        }
    }
};

storage_t::impl::impl(std::string database_path, std::string uuid)
: m_database_path(std::move(database_path))
, m_uuid(std::move(uuid))
{}

std::string
storage_t::impl::get_database_path() const
{
    return m_database_path;
}

std::string
storage_t::impl::get_uuid() const
{
    return m_uuid;
}

profiler_hub::version_t
storage_t::impl::get_storage_version() const
{
    std::error_code ec;
    const bool      present = std::filesystem::exists(m_database_path, ec);
    if(ec)
    {
        LOG_ERROR("Cannot check '{}': {}", m_database_path, ec.message());
    }
    if(!present)
    {
        return {};
    }

    try
    {
        // create_database() would create the file and pin the writer's backend mode.
        const auto backend = database_factory_t::create_database(
            m_database_path, m_uuid, storage_type_t::read);
        return assemble_version(read_metadata_rows(*backend));
    } catch(const std::exception& err)
    {
        LOG_ERROR(
            "Failed to read schema version from '{}': {}", m_database_path, err.what());
        return {};
    }
}

std::shared_ptr<data_storage::sqlite_backend>
storage_t::impl::create_database(const storage_type_t& storage_type)
{
    if(!m_database)
    {
        m_database =
            database_factory_t::create_database(m_database_path, m_uuid, storage_type);
        m_storage_type = storage_type;
    }
    return m_database;
}

}  // namespace profiler_hub
