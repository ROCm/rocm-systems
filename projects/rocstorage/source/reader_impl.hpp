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

    [[nodiscard]] data_types::node_info_list_t          get_node_list() const;
    [[nodiscard]] data_types::process_info_list_t       get_process_list() const;
    [[nodiscard]] data_types::thread_info_list_t        get_thread_list() const;
    [[nodiscard]] data_types::agent_info_t              get_agent_list() const;
    [[nodiscard]] data_types::track_info_list_t         get_track_list() const;
    [[nodiscard]] data_types::kernel_symbol_info_list_t get_kernel_symbol_list() const;
    [[nodiscard]] data_types::code_object_info_list_t   get_code_object_list() const;
    [[nodiscard]] data_types::stream_info_list_t        get_stream_list() const;
    [[nodiscard]] data_types::queue_info_list_t         get_queue_list() const;
    [[nodiscard]] data_types::pmc_info_list_t           get_pmc_info_list() const;

private:
    // void all_get_uuids() const;

    std::unique_ptr<rocstorage::storage_t>                    m_storage;
    std::shared_ptr<data_storage::database>                   m_database;
    std::shared_ptr<data_storage::schema_v3::read_statements> m_read_statements;
};

}  // namespace rocstorage
