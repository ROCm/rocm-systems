// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "storage_impl.hpp"

#include <stdexcept>
#include <string>

namespace rocstorage
{

storage_t::impl::impl(const std::string&         database_path,
                      const std::string&         uuid,
                      rocstorage::storage_type_t desired_storage_type)
: m_database(std::make_shared<rocstorage::data_storage::database>(database_path,
                                                                  uuid,
                                                                  desired_storage_type))
, m_uuid(uuid)
{
    if(!m_database)
    {
        throw std::invalid_argument("Unable to create storage!");
    }
}

std::shared_ptr<rocstorage::data_storage::database>
storage_t::impl::get_database() const
{
    return m_database;
}

std::string
storage_t::impl::get_uuid() const
{
    return m_uuid;
}

rocstorage::version_t
storage_t::impl::get_storage_version() const
{
    return m_version;
}

}  // namespace rocstorage
