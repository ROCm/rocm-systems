// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "database.hpp"

#include "rocstorage/data_types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "queries/select/table_select_query.hpp"

namespace rocstorage::data_storage::schema_v3
{

struct string_result
{
    size_t      id;
    const char* value;
};

struct read_statements
{
    explicit read_statements(std::shared_ptr<database> db, std::string uuid)
    : m_database{ std::move(db) }
    , m_uuid{ std::move(uuid) }
    {
        initialize_string_statement();
        initialize_node_info_statement();
    }
    read_statements()                                  = delete;
    read_statements(const read_statements&)            = delete;
    read_statements(read_statements&&)                 = delete;
    read_statements& operator=(const read_statements&) = delete;
    read_statements& operator=(read_statements&&)      = delete;
    virtual ~read_statements()                         = default;

    using string_statement_func_t = std::function<statement_result<string_result>()>;

    using node_info_statement_func_t =
        std::function<statement_result<data_types::node_info_t>()>;

    [[nodiscard]] string_statement_func_t string_statement() const
    {
        return m_string_statement;
    }

    [[nodiscard]] node_info_statement_func_t node_info_statement() const
    {
        return m_node_info_statement;
    }

private:
    void initialize_string_statement()
    {
        const auto uuid = m_database->get_uuid();

        queries::select::table_select_query query_builder = {};
        const auto                          query = query_builder.select("id", "string")
                               .from(fmt::format("rocpd_string_{}", uuid))
                               .get_query_string();

        m_string_statement = m_database->create_read_statement_executor<string_result>(
            query, &string_result::id, &string_result::value);
    }

    void initialize_node_info_statement()
    {
        auto query = queries::select::table_select_query{}
                         .select("id",
                                 "hash",
                                 "machine_id",
                                 "system_name",
                                 "hostname",
                                 "release",
                                 "version",
                                 "hardware_name",
                                 "domain_name")
                         .from(fmt::format("rocpd_info_node_{}", m_uuid))
                         .get_query_string();

        m_node_info_statement =
            m_database->create_read_statement_executor<data_types::node_info_t>(
                query,
                &data_types::node_info_t::node_id,
                &data_types::node_info_t::hash,
                &data_types::node_info_t::machine_id,
                &data_types::node_info_t::system_name,
                &data_types::node_info_t::hostname,
                &data_types::node_info_t::release,
                &data_types::node_info_t::version,
                &data_types::node_info_t::hardware_name,
                &data_types::node_info_t::domain_name);
    }

    std::shared_ptr<database> m_database;
    std::string               m_uuid;

    string_statement_func_t    m_string_statement;
    node_info_statement_func_t m_node_info_statement;
};
}  // namespace rocstorage::data_storage::schema_v3
