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

    [[nodiscard]] profiler_hub::version_t get_storage_version() const;

private:
    friend struct writer_t;
    friend struct reader_t;

    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace profiler_hub
