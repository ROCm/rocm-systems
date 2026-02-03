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
, m_read_statements(
      std::make_shared<data_storage::schema_v3::read_statements>(m_database,
                                                                 m_database->get_uuid()))
{
    if(!m_storage)
    {
        throw std::invalid_argument("Provided pointer to a non-existing storage!");
    }
}

data_types::node_info_list_t
reader_t::impl::get_all_nodes()
{
    if(m_node_info_list.empty())
    {
        auto statement      = m_read_statements->node_info_statement();
        auto node_info_list = statement().to_vector();

        m_node_info_list.reserve(node_info_list.size());
        for(const auto& node_info : node_info_list)
        {
            m_node_info_list.push_back(
                std::make_shared<data_types::node_info_t>(node_info));
        }
    }

    return m_node_info_list;
}

data_types::process_info_list_t
reader_t::impl::get_process_list()
{
    return {};
}
data_types::thread_info_list_t
reader_t::impl::get_thread_list()
{
    return {};
}
data_types::agent_info_t
reader_t::impl::get_agent_list()
{
    return {};
}
data_types::track_info_list_t
reader_t::impl::get_track_list()
{
    return {};
}
data_types::kernel_symbol_info_list_t
reader_t::impl::get_kernel_symbol_list()
{
    return {};
}
data_types::code_object_info_list_t
reader_t::impl::get_code_object_list()
{
    return {};
}
data_types::stream_info_list_t
reader_t::impl::get_stream_list()
{
    return {};
}
data_types::queue_info_list_t
reader_t::impl::get_queue_list()
{
    return {};
}
data_types::pmc_info_list_t
reader_t::impl::get_pmc_info_list()
{
    return {};
}

// TODO: Write reader functions for tracks and other data events when defined

}  // namespace rocstorage
