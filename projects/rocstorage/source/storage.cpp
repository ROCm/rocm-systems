// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/storage.hpp>

#include "storage_impl.hpp"

namespace rocstorage
{

storage_t::storage_t(const std::string&         database_path,
                     const std::string&         uuid,
                     rocstorage::storage_type_t desired_storage_type)
: m_impl(std::make_unique<impl>(database_path, uuid, desired_storage_type))
{}

storage_t::~storage_t() { m_impl.reset(); }

rocstorage::version_t
storage_t::get_storage_version() const
{
    return m_impl->get_storage_version();
}

}  // namespace rocstorage
