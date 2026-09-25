// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "storage_impl.hpp"
#include "debug.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"

#include "data_storage/backends/sqlite_backend.hpp"

#include <fmt/format.h>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace profiler_hub
{

namespace
{
struct metadata_row_t
{
    std::string tag;
    std::string value;
};

std::uint32_t
parse_version_component(const std::string& text)
{
    std::uint32_t value{};
    const auto*   end      = text.data() + text.size();
    const auto [ptr, errc] = std::from_chars(text.data(), end, value);
    if(errc != std::errc{} || ptr != end)
    {
        throw std::runtime_error("invalid schema version component '" + text + "'");
    }
    return value;
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
    const std::scoped_lock lock{ m_version_mutex };
    if(m_version.has_value())
    {
        return m_version.value();
    }

    if(!m_database && !std::filesystem::exists(m_database_path))
    {
        return {};
    }

    profiler_hub::version_t version{};
    try
    {
        // create_database() would create the file and pin the writer's backend mode.
        const auto backend = m_database
                                 ? m_database
                                 : database_factory_t::create_database(
                                       m_database_path, m_uuid, storage_type_t::read);

        const auto query = fmt::format(
            "SELECT tag, value FROM \"rocpd_metadata_{}\" WHERE tag IN "
            "('schema_version_major', 'schema_version_minor', 'schema_version_patch')",
            backend->get_uuid());

        const auto executor = backend->create_read_statement_executor<metadata_row_t>(
            query, &metadata_row_t::tag, &metadata_row_t::value);

        std::optional<std::uint32_t> major;
        std::optional<std::uint32_t> minor;
        std::optional<std::uint32_t> patch;
        for(const auto& row : executor().to_vector())
        {
            if(row.tag == "schema_version_major")
            {
                major = parse_version_component(row.value);
            }
            else if(row.tag == "schema_version_minor")
            {
                minor = parse_version_component(row.value);
            }
            else if(row.tag == "schema_version_patch")
            {
                patch = parse_version_component(row.value);
            }
        }

        if(!major || !minor || !patch)
        {
            throw std::runtime_error("incomplete schema version in metadata");
        }

        version = { .major = *major, .minor = *minor, .patch = *patch };
    } catch(const std::exception& err)
    {
        LOG_ERROR(
            "Failed to read schema version from '{}': {}", m_database_path, err.what());
        version = {};
    }

    m_version = version;
    return version;
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
