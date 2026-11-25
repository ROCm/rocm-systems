// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <profiler-hub/cpp/version.hpp>

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
     * database. A writer opened on a file that already exists throws
     * std::runtime_error and leaves the file unchanged. This overload is a separate
     * exported symbol. A defaulted third argument
     * would not keep the two-argument symbol in the shared library.
     * @param database_path Path to the on-disk rocpd database
     * @param uuid Unique identifier embedded into this storage's table names
     */
    explicit storage_t(const std::string& database_path, const std::string& uuid);

    /**
     * @brief Construct a storage handle that writes `schema_version`.
     * Ignored when only reading an existing database. A writer opened on a file that
     * already exists throws std::runtime_error and does not apply `schema_version`.
     * version_t{} selects the newest schema the writer supports. Any other version
     * must be one the writer can populate. Currently those are 3.0.0 and 3.0.1.
     * version_t{} selects 3.0.1.
     * @param database_path Path to the on-disk rocpd database
     * @param uuid Unique identifier embedded into this storage's table names
     * @param schema_version rocpd schema version to initialize when writing
     * @throws std::runtime_error when the writer cannot create `schema_version`,
     *         or when a writer is opened on a database file that already exists
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
     * @brief Profiler-hub library version this storage was built with.
     * @return The version from the VERSION file, for example 0.2.0.
     *         This is not the RocPD schema version.
     */
    [[nodiscard]] profiler_hub::version_t get_storage_version() const;

    /**
     * @brief Schema version the writer creates for this storage
     * @return The resolved writable version. version_t{} is returned as 3.0.1.
     *         An explicit 3.0.0 or 3.0.1 is returned unchanged.
     */
    [[nodiscard]] profiler_hub::version_t get_writer_schema_version() const;

    /**
     * @brief RocPD schema version stamped into the opened trace's rocpd_metadata table.
     * @return The schema version of the trace, read once and cached. An empty database
     *         path has no trace, so the library version is returned. An unreadable
     *         table or missing rows yield 0.0.0 and log an error.
     */
    [[nodiscard]] profiler_hub::version_t get_trace_schema_version() const;

private:
    friend struct writer_t;
    friend struct reader_t;

    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace profiler_hub
