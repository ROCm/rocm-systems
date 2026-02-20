// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocpdsna/storage.hpp"

#include "rocpdsna/storage_types.hpp"

#include "data_storage/backends/sqlite_backend.hpp"

#include <memory>
#include <string>

namespace rocpdsna
{

struct storage_t::impl
{
    enum class storage_type_t
    {
        none  = 0,
        read  = 1,
        write = 2
    };

    explicit impl(std::string database_path, std::string uuid);

    [[nodiscard]] std::string get_database_path() const;
    [[nodiscard]] std::string get_uuid() const;

    [[nodiscard]] rocpdsna::version_t get_storage_version() const;

    std::shared_ptr<data_storage::sqlite_backend> create_database(
        const storage_type_t& storage_type);

private:
    rocpdsna::version_t m_version{ ROCPDSNA_VERSION_MAJOR,
                                   ROCPDSNA_VERSION_MINOR,
                                   ROCPDSNA_VERSION_PATCH };

    storage_type_t                                m_storage_type{ storage_type_t::none };
    std::shared_ptr<data_storage::sqlite_backend> m_database{ nullptr };

    std::string m_database_path;
    std::string m_uuid;

    struct database_factory_t;
};

}  // namespace rocpdsna
