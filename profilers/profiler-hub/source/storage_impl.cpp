// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "storage_impl.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"

#include "data_storage/backends/sqlite_backend.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace profiler_hub
{

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
    return m_version;
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
