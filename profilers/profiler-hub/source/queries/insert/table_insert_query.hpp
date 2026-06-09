// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "insert_query_builders.hpp"

#include <sstream>
#include <string>

namespace profiler_hub::queries::insert
{

struct table_insert_query
{
    table_insert_query()
    : m_query_columns_builder{ m_ss }
    {}

    query_columns_builder& set_table_name(const std::string& table_name)
    {
        m_ss.str("");
        m_ss << "INSERT INTO " << table_name << " ";
        return m_query_columns_builder;
    }

private:
    std::stringstream     m_ss;
    query_columns_builder m_query_columns_builder;
};

}  // namespace profiler_hub::queries::insert
