// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/reader.hpp>

#include "data_storage/database.hpp"

#include <stdexcept>

namespace rocstorage
{

struct reader::impl
{
    explicit impl(std::shared_ptr<data_storage::database> database, std::string uuid)
    : m_database(std::move(database))
    , m_uuid(std::move(uuid))
    {
        if(!m_database)
        {
            throw std::invalid_argument("Provided pointer to a non-existing database!");
        }
    }

    std::shared_ptr<data_storage::database> m_database;
    std::string                             m_uuid;
};

reader::reader(std::shared_ptr<data_storage::database> database, std::string uuid)
: m_impl(std::make_unique<impl>(std::move(database), std::move(uuid)))
{}

reader::~reader() = default;

}  // namespace rocstorage
