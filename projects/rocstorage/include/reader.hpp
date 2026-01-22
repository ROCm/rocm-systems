// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

namespace rocm
{
class storage;
}

namespace rocstorage
{
namespace data_storage
{
class database;
}

struct reader
{
    friend class rocm::storage;

private:
    explicit reader(std::shared_ptr<data_storage::database> database, std::string uuid);

public:
    virtual ~reader();

    reader()                          = delete;
    reader(const reader&)             = delete;
    reader& operator=(const reader&)  = delete;
    reader(const reader&&)            = delete;
    reader& operator=(const reader&&) = delete;

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace rocstorage
