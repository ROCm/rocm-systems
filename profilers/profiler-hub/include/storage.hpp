// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <profiler-hub/version.hpp>

#include <memory>
#include <string>

namespace profiler_hub
{
struct writer_t;
struct reader_t;

class storage_t
{
public:
    /**
     * @brief Construct a storage handle backed by a database on disk.
     * Writes schema 3.0.1. That version is ignored when only reading an existing
     * database. This overload is a separate exported symbol. A defaulted third argument
     * would not keep the two-argument symbol in the shared library.
     * @param database_path Path to the on-disk rocpd database
     * @param uuid Unique identifier embedded into this storage's table names
     */
    explicit storage_t(const std::string& database_path, const std::string& uuid);

    /**
     * @brief Construct a storage handle that writes `schema_version`.
     * Ignored when only reading an existing database.
     * @param database_path Path to the on-disk rocpd database
     * @param uuid Unique identifier embedded into this storage's table names
     * @param schema_version rocpd schema version to initialize when writing
     */
    explicit storage_t(const std::string&      database_path,
                       const std::string&      uuid,
                       profiler_hub::version_t schema_version);
    ~storage_t();

    storage_t(const storage_t&)            = delete;
    storage_t(storage_t&&)                 = delete;
    storage_t& operator=(const storage_t&) = delete;
    storage_t& operator=(storage_t&&)      = delete;

    /**
     * @brief Rocpd schema version this storage was constructed with
     * @return The schema_version passed to the constructor (default 3.0.1)
     */
    [[nodiscard]] profiler_hub::version_t get_storage_version() const;

private:
    friend struct writer_t;
    friend struct reader_t;

    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace profiler_hub
