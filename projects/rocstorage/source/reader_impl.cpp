// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "reader_impl.hpp"
#include "rocstorage/reader.hpp"
#include "rocstorage/storage.hpp"
#include "storage_impl.hpp"

#include "queries/select/table_select_query.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace rocstorage
{

reader_t::impl::impl(std::unique_ptr<rocstorage::storage_t> storage)
: m_storage(std::move(storage))
, m_database(m_storage->m_impl->create_database(storage_t::impl::storage_type_t::read))
{
    if(!m_storage)
    {
        throw std::invalid_argument("Provided pointer to a non-existing storage!");
    }

    const auto get_uuids_query =
        "SELECT DISTINCT replace(name, rtrim(name, replace(name, '_', '')), '') AS guid "
        "FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_%';";

    m_database->execute_query(get_uuids_query);
}

data_types::node_info_list_t
reader_t::impl::get_node_list() const
{
    auto query = queries::select::table_select_query{}
                     .from("rocpd_info_node")
                     .select_all()
                     .get_query_string();
    m_database->execute_query(query);

    return {};
}

data_types::process_info_list_t
reader_t::impl::get_process_list() const
{
    return {};
}
data_types::thread_info_list_t
reader_t::impl::get_thread_list() const
{
    return {};
}
data_types::agent_info_t
reader_t::impl::get_agent_list() const
{
    return {};
}
data_types::track_info_list_t
reader_t::impl::get_track_list() const
{
    return {};
}
data_types::kernel_symbol_info_list_t
reader_t::impl::get_kernel_symbol_list() const
{
    return {};
}
data_types::code_object_info_list_t
reader_t::impl::get_code_object_list() const
{
    return {};
}
data_types::stream_info_list_t
reader_t::impl::get_stream_list() const
{
    return {};
}
data_types::queue_info_list_t
reader_t::impl::get_queue_list() const
{
    return {};
}
data_types::pmc_info_list_t
reader_t::impl::get_pmc_info_list() const
{
    return {};
}

}  // namespace rocstorage
