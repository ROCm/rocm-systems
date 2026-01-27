// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/reader.hpp>
#include <rocstorage/storage_types.hpp>
#include <rocstorage/writer.hpp>

#include <memory>
#include <string>

namespace rocm
{

class storage
{
public:
    explicit storage(const std::string&          database_path,
                     const std::string&          uuid,
                     rocstorage::database_type_t database_type =
                         rocstorage::database_type_t::in_memory);
    virtual ~storage();

    storage(const storage&)            = delete;
    storage(storage&&)                 = delete;
    storage& operator=(const storage&) = delete;
    storage& operator=(storage&&)      = delete;

    [[nodiscard]] std::shared_ptr<rocstorage::writer> get_writer() const;
    [[nodiscard]] std::shared_ptr<rocstorage::reader> get_reader() const;

    [[nodiscard]] rocstorage::version_t get_storage_version() const;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace rocm
