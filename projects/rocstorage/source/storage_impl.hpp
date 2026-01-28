// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/storage.hpp>

#include "data_storage/database.hpp"

#include <memory>
#include <string>

namespace rocstorage
{

struct storage_t::impl
{
    explicit impl(const std::string&         database_path,
                  const std::string&         uuid,
                  rocstorage::storage_type_t desired_storage_type);

    [[nodiscard]] std::shared_ptr<rocstorage::data_storage::database> get_database()
        const;

    [[nodiscard]] std::string get_uuid() const;

    [[nodiscard]] rocstorage::version_t get_storage_version() const;

private:
    std::shared_ptr<rocstorage::data_storage::database> m_database;
    std::string                                         m_uuid;

    rocstorage::version_t m_version{ ROCSTORAGE_VERSION_MAJOR,
                                     ROCSTORAGE_VERSION_MINOR,
                                     ROCSTORAGE_VERSION_PATCH };
};

}  // namespace rocstorage
