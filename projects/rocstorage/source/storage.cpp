// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/reader.hpp>
#include <rocstorage/storage.hpp>
#include <rocstorage/writer.hpp>

#include "data_storage/database.hpp"

namespace rocm
{

struct storage::impl
{
    explicit impl(std::string                 database_path,
                  const std::string&          uuid,
                  rocstorage::database_type_t database_type)
    : m_database(std::make_shared<rocstorage::data_storage::database>(database_path,
                                                                      uuid,
                                                                      database_type))
    , m_uuid(uuid)
    , m_writer(new rocstorage::writer(m_database, m_uuid))
    , m_reader(new rocstorage::reader(m_database, m_uuid))
    {
        if(!m_database)
        {
            throw std::invalid_argument("Unable to create database!");
        }
    }

    std::shared_ptr<rocstorage::data_storage::database> m_database;
    std::string                                         m_uuid;
    std::shared_ptr<rocstorage::writer>                 m_writer;
    std::shared_ptr<rocstorage::reader>                 m_reader;
    const rocstorage::version_t m_version{ ROCSTORAGE_VERSION_MAJOR,
                                           ROCSTORAGE_VERSION_MINOR,
                                           ROCSTORAGE_VERSION_PATCH };
};

storage::storage(std::string                 database_path,
                 std::string                 uuid,
                 rocstorage::database_type_t database_type)
: m_impl(std::make_unique<impl>(std::move(database_path), std::move(uuid), database_type))
{}

storage::~storage() { m_impl.reset(); }

std::shared_ptr<rocstorage::writer>
storage::get_writer() const
{
    return m_impl->m_writer;
}

std::shared_ptr<rocstorage::reader>
storage::get_reader() const
{
    return m_impl->m_reader;
}

rocstorage::version_t
storage::get_storage_version() const
{
    return m_impl->m_version;
}

}  // namespace rocm
