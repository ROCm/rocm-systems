// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocstorage/data_types.hpp"
#include "rocstorage/reader.hpp"
#include "rocstorage/storage.hpp"

#include "data_storage/database.hpp"
#include "data_storage/read_statements.hpp"

#include <memory>

namespace rocstorage
{

struct reader_t::impl
{
    explicit impl(std::unique_ptr<rocstorage::storage_t> storage);

    [[nodiscard]] data_types::node_info_list_t          get_all_nodes();
    [[nodiscard]] data_types::process_info_list_t       get_process_list();
    [[nodiscard]] data_types::thread_info_list_t        get_thread_list();
    [[nodiscard]] data_types::agent_info_t              get_agent_list();
    [[nodiscard]] data_types::track_info_list_t         get_track_list();
    [[nodiscard]] data_types::kernel_symbol_info_list_t get_kernel_symbol_list();
    [[nodiscard]] data_types::code_object_info_list_t   get_code_object_list();
    [[nodiscard]] data_types::stream_info_list_t        get_stream_list();
    [[nodiscard]] data_types::queue_info_list_t         get_queue_list();
    [[nodiscard]] data_types::pmc_info_list_t           get_pmc_info_list();

private:
    // void all_get_uuids() const;

    std::unique_ptr<rocstorage::storage_t>                    m_storage;
    std::shared_ptr<data_storage::database>                   m_database;
    std::shared_ptr<data_storage::schema_v3::read_statements> m_read_statements;

    data_types::node_info_list_t          m_node_info_list;
    data_types::process_info_list_t       m_process_info_list;
    data_types::thread_info_list_t        m_thread_info_list;
    data_types::agent_info_list_t         m_agent_info_list;
    data_types::track_info_list_t         m_track_info_list;
    data_types::kernel_symbol_info_list_t m_kernel_symbol_info_list;
    data_types::code_object_info_list_t   m_code_object_info_list;
    data_types::stream_info_list_t        m_stream_info_list;
    data_types::queue_info_list_t         m_queue_info_list;
    data_types::pmc_info_list_t           m_pmc_info_list;
};

}  // namespace rocstorage
