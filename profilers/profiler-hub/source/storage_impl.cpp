// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "storage_impl.hpp"
#include "debug.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "profiler-hub/cpp/version.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "data_storage/schema_version_text.hpp"

#include <fmt/format.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
    if(m_version.has_value())
    {
        return m_version.value();
    }

    auto backend = create_database(storage_type_t::read);

    const auto table = fmt::format("rocpd_metadata_{}", backend->get_uuid());
    const auto query = fmt::format("SELECT tag, value FROM {} WHERE tag IN "
                                   "('schema_version', 'schema_version_major', "
                                   "'schema_version_minor', 'schema_version_patch')",
                                   table);

    profiler_hub::version_t version{ .major = 0, .minor = 0, .patch = 0 };
    try
    {
        auto executor = backend->create_read_statement_executor<metadata_row_t>(
            query, &metadata_row_t::tag, &metadata_row_t::value);

        bool                         saw_component = false;
        std::optional<std::string>   version_text;
        std::optional<std::uint32_t> major;
        std::optional<std::uint32_t> minor;
        std::optional<std::uint32_t> patch;
        for(const auto& row : executor().to_vector())
        {
            if(row.tag == "schema_version")
            {
                version_text = row.value;
            }
            else if(row.tag == "schema_version_major")
            {
                major         = static_cast<std::uint32_t>(std::stoul(row.value));
                saw_component = true;
            }
            else if(row.tag == "schema_version_minor")
            {
                minor         = static_cast<std::uint32_t>(std::stoul(row.value));
                saw_component = true;
            }
            else if(row.tag == "schema_version_patch")
            {
                patch         = static_cast<std::uint32_t>(std::stoul(row.value));
                saw_component = true;
            }
        }
        if(major.has_value() && minor.has_value() && patch.has_value())
        {
            version = { major.value(), minor.value(), patch.value() };
        }
        else if(version_text.has_value())
        {
            const auto parsed =
                data_storage::parse_schema_version_text(version_text.value());
            if(parsed.has_value()) version = parsed.value();
        }
        else if(saw_component)
        {
            if(major.has_value()) version.major = major.value();
            if(minor.has_value()) version.minor = minor.value();
            if(patch.has_value()) version.patch = patch.value();
        }
    } catch(const std::exception& err)
    {
        LOG_ERROR("Failed to read schema version from '{}': {}", table, err.what());
    }

    m_version = version;
    return version;
}

std::shared_ptr<data_storage::sqlite_backend>
storage_t::impl::create_database(const storage_type_t& storage_type) const
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
