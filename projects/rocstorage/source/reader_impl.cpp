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

    initialize_all_info_lists();
}

void
reader_t::impl::initialize_string_list()
{
    const auto& statement   = m_read_statements->string_statement();
    const auto  string_list = statement().to_vector();

    m_string_info_utility.reserve(string_list.size());
    for(const auto& string : string_list)
    {
        m_string_info_utility.emplace(string.id, string.value);
    }
}

void
reader_t::impl::initialize_all_info_lists()
{
    initialize_string_list();
    m_node_info_list          = get_all_nodes();
    m_process_info_list       = get_process_list();
    m_thread_info_list        = get_thread_list();
    m_agent_info_list         = get_agent_list();
    m_kernel_symbol_info_list = get_kernel_symbol_list();
    m_code_object_info_list   = get_code_object_list();
    m_stream_info_list        = get_stream_list();
    m_queue_info_list         = get_queue_list();
    m_pmc_info_list           = get_pmc_info_list();
    m_track_info_list         = get_track_list();
}

data_types::node_info_list_t
reader_t::impl::get_all_nodes()
{
    if(m_node_info_list.empty())
    {
        const auto& statement      = m_read_statements->node_info_statement();
        const auto  node_info_list = statement().to_vector();

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
    if(m_process_info_list.empty())
    {
        const auto& statement         = m_read_statements->process_info_statement();
        const auto  process_info_list = statement().to_vector();

        m_process_info_list.reserve(process_info_list.size());
        for(const auto& process_info : process_info_list)
        {
            auto process_info_ptr     = std::make_shared<data_types::process_info_t>();
            process_info_ptr->ppid    = process_info.ppid.value_or(0);
            process_info_ptr->pid     = process_info.pid;
            process_info_ptr->init    = process_info.init.value_or(0);
            process_info_ptr->fini    = process_info.fini.value_or(0);
            process_info_ptr->start   = process_info.start.value_or(0);
            process_info_ptr->end     = process_info.end.value_or(0);
            process_info_ptr->command = process_info.command.value_or(nullptr);
            process_info_ptr->environment = process_info.environment;
            process_info_ptr->extdata     = process_info.extdata;
            process_info_ptr->node_id     = process_info.nid;

            m_process_info_list.push_back(process_info_ptr);
            m_process_info_utility.emplace(process_info.id, process_info_ptr);
        }
    }

    return m_process_info_list;
}

data_types::thread_info_list_t
reader_t::impl::get_thread_list()
{
    if(m_thread_info_list.empty())
    {
        const auto& statement        = m_read_statements->thread_info_statement();
        const auto  thread_info_list = statement().to_vector();

        m_thread_info_list.reserve(thread_info_list.size());
        for(const auto& thread_info : thread_info_list)
        {
            auto thread_info_ptr = std::make_shared<data_types::thread_info_t>();
            thread_info_ptr->parent_process_id = thread_info.ppid.value_or(0);
            thread_info_ptr->thread_id         = thread_info.tid;
            thread_info_ptr->name              = thread_info.name.value_or(nullptr);
            thread_info_ptr->start             = thread_info.start.value_or(0);
            thread_info_ptr->end               = thread_info.end.value_or(0);
            thread_info_ptr->extdata           = thread_info.extdata;
            thread_info_ptr->node_id           = thread_info.nid;
            thread_info_ptr->process_id        = thread_info.pid;

            m_thread_info_list.push_back(thread_info_ptr);
            m_thread_info_utility.emplace(thread_info.id, thread_info_ptr);
        }
    }

    return m_thread_info_list;
}
data_types::agent_info_list_t
reader_t::impl::get_agent_list()
{
    if(m_agent_info_list.empty())
    {
        const auto& statement       = m_read_statements->agent_info_statement();
        const auto  agent_info_list = statement().to_vector();

        m_agent_info_list.reserve(agent_info_list.size());
        for(const auto& agent_info : agent_info_list)
        {
            if(!agent_info.type.has_value() || !agent_info.type_index.has_value())
            {
                LOG_ERROR("Corrupted database detected. Agent type or type index is not "
                          "available for agent info with id: {}",
                          agent_info.id);
                continue;
            }

            auto agent_info_ptr = std::make_shared<data_types::agent_info_t>();
            agent_info_ptr->unique_id =
                data_types::agent_unique_id_t{ agent_info.type.value(),
                                               agent_info.type_index.value() };
            agent_info_ptr->absolute_index = agent_info.absolute_index.value_or(0);
            agent_info_ptr->logical_index  = agent_info.logical_index.value_or(0);
            agent_info_ptr->uuid           = agent_info.uuid.value_or(0);
            agent_info_ptr->name           = agent_info.name.value_or(nullptr);
            agent_info_ptr->model_name     = agent_info.model_name.value_or(nullptr);
            agent_info_ptr->vendor_name    = agent_info.vendor_name.value_or(nullptr);
            agent_info_ptr->product_name   = agent_info.product_name.value_or(nullptr);
            agent_info_ptr->user_name      = agent_info.user_name.value_or(nullptr);
            agent_info_ptr->extdata        = agent_info.extdata;
            agent_info_ptr->node_id        = agent_info.nid;
            agent_info_ptr->process_id     = agent_info.pid;

            m_agent_info_list.push_back(agent_info_ptr);
            m_agent_info_utility.emplace(agent_info.id, agent_info_ptr);
        }
    }

    return m_agent_info_list;
}

data_types::track_info_list_t
reader_t::impl::get_track_list()
{
    if(m_track_info_list.empty())
    {
        const auto& statement       = m_read_statements->track_info_statement();
        const auto  track_info_list = statement().to_vector();

        m_track_info_list.reserve(track_info_list.size());
        for(const auto& track_info : track_info_list)
        {
            const char* track_name = nullptr;
            if(track_info.name_id.has_value())
            {
                const auto track_name_ptr =
                    m_string_info_utility.find(track_info.name_id.value());
                if(track_name_ptr == m_string_info_utility.end())
                {
                    LOG_ERROR(
                        "Corrupted database detected. Track name is not available for "
                        "track info with id: {}",
                        track_info.id);
                }
                else
                {
                    track_name = track_name_ptr->second.c_str();
                }
            }

            auto track_info_ptr = std::make_shared<data_types::track_info_t>();
            track_info_ptr->name =
                track_name ? std::make_optional(track_name) : std::nullopt;
            track_info_ptr->extdata    = track_info.extdata;
            track_info_ptr->node_id    = track_info.nid;
            track_info_ptr->process_id = track_info.pid;
            track_info_ptr->thread_id  = track_info.tid;

            m_track_info_list.push_back(track_info_ptr);
            m_track_info_utility.emplace(track_info.id, track_info_ptr);
        }
    }

    return m_track_info_list;
}

data_types::kernel_symbol_info_list_t
reader_t::impl::get_kernel_symbol_list()
{
    if(m_kernel_symbol_info_list.empty())
    {
        const auto& statement = m_read_statements->kernel_symbol_info_statement();
        const auto  kernel_symbol_info_list = statement().to_vector();

        m_kernel_symbol_info_list.reserve(kernel_symbol_info_list.size());
        for(const auto& kernel_symbol_info : kernel_symbol_info_list)
        {
            auto kernel_symbol_info_ptr =
                std::make_shared<data_types::kernel_symbol_info_t>();
            kernel_symbol_info_ptr->id = kernel_symbol_info.id;
            kernel_symbol_info_ptr->name =
                kernel_symbol_info.kernel_name.value_or(nullptr);
            kernel_symbol_info_ptr->display_name =
                kernel_symbol_info.display_name.value_or(nullptr);
            kernel_symbol_info_ptr->kernel_object =
                kernel_symbol_info.kernel_object.value_or(0);
            kernel_symbol_info_ptr->kernarg_segment_size =
                kernel_symbol_info.kernarg_segment_size.value_or(0);
            kernel_symbol_info_ptr->kernarg_segment_alignment =
                kernel_symbol_info.kernarg_segment_alignment.value_or(0);
            kernel_symbol_info_ptr->group_segment_size =
                kernel_symbol_info.group_segment_size.value_or(0);
            kernel_symbol_info_ptr->private_segment_size =
                kernel_symbol_info.private_segment_size.value_or(0);
            kernel_symbol_info_ptr->sgpr_count =
                kernel_symbol_info.sgpr_count.value_or(0);
            kernel_symbol_info_ptr->arch_vgpr_count =
                kernel_symbol_info.arch_vgpr_count.value_or(0);
            kernel_symbol_info_ptr->accum_vgpr_count =
                kernel_symbol_info.accum_vgpr_count.value_or(0);
            kernel_symbol_info_ptr->extdata     = kernel_symbol_info.extdata;
            kernel_symbol_info_ptr->node_id     = kernel_symbol_info.nid;
            kernel_symbol_info_ptr->process_id  = kernel_symbol_info.pid;
            kernel_symbol_info_ptr->code_obj_id = kernel_symbol_info.code_object_id;

            m_kernel_symbol_info_list.push_back(kernel_symbol_info_ptr);
            m_kernel_symbol_info_utility.emplace(kernel_symbol_info.id,
                                                 kernel_symbol_info_ptr);
        }
    }

    return m_kernel_symbol_info_list;
}

data_types::code_object_info_list_t
reader_t::impl::get_code_object_list()
{
    if(m_code_object_info_list.empty())
    {
        const auto& statement = m_read_statements->code_object_info_statement();
        const auto  code_object_info_list = statement().to_vector();

        m_code_object_info_list.reserve(code_object_info_list.size());
        for(const auto& code_object_info : code_object_info_list)
        {
            auto code_object_info_ptr =
                std::make_shared<data_types::code_object_info_t>();
            code_object_info_ptr->id         = code_object_info.id;
            code_object_info_ptr->uri        = code_object_info.uri.value_or(nullptr);
            code_object_info_ptr->load_base  = code_object_info.load_base.value_or(0);
            code_object_info_ptr->load_size  = code_object_info.load_size.value_or(0);
            code_object_info_ptr->load_delta = code_object_info.load_delta.value_or(0);
            code_object_info_ptr->storage_type =
                code_object_info.storage_type.value_or(nullptr);
            code_object_info_ptr->extdata    = code_object_info.extdata;
            code_object_info_ptr->node_id    = code_object_info.nid;
            code_object_info_ptr->process_id = code_object_info.pid;

            // Resolve agent_id FK to agent_unique_id_t if present
            if(code_object_info.agent_id.has_value())
            {
                auto agent_it =
                    m_agent_info_utility.find(code_object_info.agent_id.value());
                if(agent_it != m_agent_info_utility.end() && agent_it->second)
                {
                    code_object_info_ptr->agent_id = agent_it->second->unique_id;
                }
            }

            m_code_object_info_list.push_back(code_object_info_ptr);
            m_code_object_info_utility.emplace(code_object_info.id, code_object_info_ptr);
        }
    }

    return m_code_object_info_list;
}

data_types::stream_info_list_t
reader_t::impl::get_stream_list()
{
    if(m_stream_info_list.empty())
    {
        const auto& statement        = m_read_statements->stream_info_statement();
        const auto  stream_info_list = statement().to_vector();

        m_stream_info_list.reserve(stream_info_list.size());
        for(const auto& stream_info : stream_info_list)
        {
            auto stream_info_ptr        = std::make_shared<data_types::stream_info_t>();
            stream_info_ptr->stream_id  = stream_info.id;
            stream_info_ptr->name       = stream_info.name.value_or(nullptr);
            stream_info_ptr->extdata    = stream_info.extdata;
            stream_info_ptr->node_id    = stream_info.nid;
            stream_info_ptr->process_id = stream_info.pid;

            m_stream_info_list.push_back(stream_info_ptr);
            m_stream_info_utility.emplace(stream_info.id, stream_info_ptr);
        }
    }

    return m_stream_info_list;
}

data_types::queue_info_list_t
reader_t::impl::get_queue_list()
{
    if(m_queue_info_list.empty())
    {
        const auto& statement       = m_read_statements->queue_info_statement();
        const auto  queue_info_list = statement().to_vector();

        m_queue_info_list.reserve(queue_info_list.size());
        for(const auto& queue_info : queue_info_list)
        {
            auto queue_info_ptr        = std::make_shared<data_types::queue_info_t>();
            queue_info_ptr->queue_id   = queue_info.id;
            queue_info_ptr->name       = queue_info.name.value_or(nullptr);
            queue_info_ptr->extdata    = queue_info.extdata;
            queue_info_ptr->node_id    = queue_info.nid;
            queue_info_ptr->process_id = queue_info.pid;

            m_queue_info_list.push_back(queue_info_ptr);
            m_queue_info_utility.emplace(queue_info.id, queue_info_ptr);
        }
    }

    return m_queue_info_list;
}

data_types::pmc_info_list_t
reader_t::impl::get_pmc_info_list()
{
    if(m_pmc_info_list.empty())
    {
        const auto& statement     = m_read_statements->pmc_info_statement();
        const auto  pmc_info_list = statement().to_vector();

        m_pmc_info_list.reserve(pmc_info_list.size());
        for(const auto& pmc_info : pmc_info_list)
        {
            // Resolve agent_id FK to agent_unique_id_t if present
            std::optional<data_types::agent_unique_id_t> agent_unique_id;
            if(pmc_info.agent_id.has_value())
            {
                auto agent_it = m_agent_info_utility.find(pmc_info.agent_id.value());
                if(agent_it != m_agent_info_utility.end() && agent_it->second)
                {
                    agent_unique_id = agent_it->second->unique_id;
                }
            }

            auto pmc_info_ptr = std::make_shared<data_types::pmc_info_t>();
            pmc_info_ptr->unique_id =
                data_types::pmc_info_unique_id_t{ pmc_info.name, agent_unique_id };
            pmc_info_ptr->target_arch      = pmc_info.target_arch.value_or(nullptr);
            pmc_info_ptr->event_code       = pmc_info.event_code.value_or(0);
            pmc_info_ptr->instance_id      = pmc_info.instance_id.value_or(0);
            pmc_info_ptr->symbol           = pmc_info.symbol;
            pmc_info_ptr->description      = pmc_info.description.value_or(nullptr);
            pmc_info_ptr->long_description = pmc_info.long_description.value_or(nullptr);
            pmc_info_ptr->component        = pmc_info.component.value_or(nullptr);
            pmc_info_ptr->units            = pmc_info.units.value_or(nullptr);
            pmc_info_ptr->value_type       = pmc_info.value_type.value_or(nullptr);
            pmc_info_ptr->block            = pmc_info.block.value_or(nullptr);
            pmc_info_ptr->expression       = pmc_info.expression.value_or(nullptr);
            pmc_info_ptr->is_constant      = pmc_info.is_constant.value_or(0);
            pmc_info_ptr->is_derived       = pmc_info.is_derived.value_or(0);
            pmc_info_ptr->extdata          = pmc_info.extdata;
            pmc_info_ptr->node_id          = pmc_info.nid;
            pmc_info_ptr->process_id       = pmc_info.pid;

            m_pmc_info_list.push_back(pmc_info_ptr);
            m_pmc_info_utility.emplace(pmc_info.id, pmc_info_ptr);
        }
    }

    return m_pmc_info_list;
}

// TODO: Write reader functions for tracks and other data events when defined

}  // namespace rocstorage
