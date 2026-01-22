// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "insert_query_builders.hpp"

namespace rocstorage
{
namespace data_storage
{
namespace queries
{

struct table_insert_query
{
    table_insert_query()
    : _query_columns_builder{ _ss }
    {}

    inline query_builders::query_columns_builder& set_table_name(
        const std::string& tableName)
    {
        _ss.str("");
        _ss << "INSERT INTO " << tableName << " ";
        return _query_columns_builder;
    }

private:
    std::stringstream                     _ss;
    query_builders::query_columns_builder _query_columns_builder;
};

}  // namespace queries
}  // namespace data_storage
}  // namespace rocstorage
