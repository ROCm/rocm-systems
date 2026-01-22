// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocstorage/reader.hpp>
#include <rocstorage/writer.hpp>

#include <memory>
#include <string>

namespace rocm
{

class storage
{
public:
    explicit storage(std::string database_path, std::string uuid);
    virtual ~storage();

    storage(const storage&)            = delete;
    storage(storage&&)                 = delete;
    storage& operator=(const storage&) = delete;
    storage& operator=(storage&&)      = delete;

    std::shared_ptr<rocstorage::writer> get_writer() const;
    std::shared_ptr<rocstorage::reader> get_reader() const;

    // TODO: Add get_version()
private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace rocm
