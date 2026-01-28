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
    enum class storage_type_t
    {
        none  = 0,
        read  = 1,
        write = 2
    };

    explicit impl(const std::string& database_path, const std::string& uuid);

    [[nodiscard]] std::string get_database_path() const;
    [[nodiscard]] std::string get_uuid() const;

    [[nodiscard]] rocstorage::version_t get_storage_version() const;

    std::shared_ptr<data_storage::database> create_database(
        const storage_type_t& storage_type);

private:
    rocstorage::version_t m_version{ ROCSTORAGE_VERSION_MAJOR,
                                     ROCSTORAGE_VERSION_MINOR,
                                     ROCSTORAGE_VERSION_PATCH };

    storage_type_t                          m_storage_type{ storage_type_t::none };
    std::shared_ptr<data_storage::database> m_database{ nullptr };

    std::string m_database_path;
    std::string m_uuid;

    struct database_factory_t;
};

}  // namespace rocstorage
