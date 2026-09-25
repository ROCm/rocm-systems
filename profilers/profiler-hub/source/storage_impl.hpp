// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/storage.hpp"

#include "profiler-hub/version.hpp"

#include "data_storage/backends/sqlite_backend.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace profiler_hub
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

    // Reads the schema version on first call and caches it, including a failed read
    // on an existing file. Returns {} uncached when the trace file does not exist yet;
    // never creates the file.
    [[nodiscard]] profiler_hub::version_t get_storage_version() const;

    std::shared_ptr<data_storage::sqlite_backend> create_database(
        const storage_type_t& storage_type);

private:
    mutable std::mutex                             m_version_mutex;
    mutable std::optional<profiler_hub::version_t> m_version;
    storage_type_t                                 m_storage_type{ storage_type_t::none };
    std::shared_ptr<data_storage::sqlite_backend>  m_database{ nullptr };

    std::string m_database_path;
    std::string m_uuid;

    struct database_factory_t;
};

}  // namespace profiler_hub
