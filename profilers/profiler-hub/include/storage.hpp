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
    explicit storage_t(const std::string& database_path, const std::string& uuid);
    ~storage_t();

    storage_t(const storage_t&)            = delete;
    storage_t(storage_t&&)                 = delete;
    storage_t& operator=(const storage_t&) = delete;
    storage_t& operator=(storage_t&&)      = delete;

    /**
     * Returns the RocPD schema version stamped in the trace's metadata.
     * @return {0,0,0} if the trace does not exist yet or its metadata cannot be read.
     * @note Does not create the file; the result is cached once read.
     */
    [[nodiscard]] profiler_hub::version_t get_storage_version() const;

private:
    friend struct writer_t;
    friend struct reader_t;

    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace profiler_hub
