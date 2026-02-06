// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "reader_impl.hpp"
#include "rocstorage/reader.hpp"
#include "rocstorage/storage.hpp"
#include "storage_impl.hpp"

#include "queries/select/table_select_query.hpp"

#include <algorithm>
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
    m_process_info_list       = get_all_processes();
    m_thread_info_list        = get_all_threads();
    m_agent_info_list         = get_all_agents();
    m_code_object_info_list   = get_all_code_objects();
    m_kernel_symbol_info_list = get_all_kernel_symbols();
    m_stream_info_list        = get_all_streams();
    m_queue_info_list         = get_all_queues();
    m_pmc_info_list           = get_all_pmc_infos();
    m_track_info_list         = get_all_tracks();
}

reader_types::node_info_list_t
reader_t::impl::get_all_nodes()
{
    if(m_node_info_list.empty())
    {
        const auto& statement      = m_read_statements->node_info_statement();
        const auto  node_info_list = statement().to_vector();

        m_node_info_list.reserve(node_info_list.size());
        for(const auto& node_info : node_info_list)
        {
            auto node_info_ptr           = std::make_shared<reader_types::node_info_t>();
            node_info_ptr->node_id       = node_info.node_id;
            node_info_ptr->hash          = node_info.hash;
            node_info_ptr->machine_id    = node_info.machine_id;
            node_info_ptr->system_name   = node_info.system_name;
            node_info_ptr->hostname      = node_info.hostname;
            node_info_ptr->release       = node_info.release;
            node_info_ptr->version       = node_info.version;
            node_info_ptr->hardware_name = node_info.hardware_name;
            node_info_ptr->domain_name   = node_info.domain_name;

            m_node_info_list.push_back(node_info_ptr);
            m_node_info_utility.emplace(node_info.node_id, node_info_ptr);
        }
    }

    return m_node_info_list;
}

reader_types::process_info_list_t
reader_t::impl::get_all_processes()
{
    if(m_process_info_list.empty())
    {
        const auto& statement         = m_read_statements->process_info_statement();
        const auto  process_info_list = statement().to_vector();

        m_process_info_list.reserve(process_info_list.size());
        for(const auto& process_info : process_info_list)
        {
            auto process_info_ptr     = std::make_shared<reader_types::process_info_t>();
            process_info_ptr->ppid    = process_info.ppid.value_or(0);
            process_info_ptr->pid     = process_info.pid;
            process_info_ptr->init    = process_info.init.value_or(0);
            process_info_ptr->fini    = process_info.fini.value_or(0);
            process_info_ptr->start   = process_info.start.value_or(0);
            process_info_ptr->end     = process_info.end.value_or(0);
            process_info_ptr->command = process_info.command.value_or(nullptr);
            process_info_ptr->environment = process_info.environment;
            process_info_ptr->extdata     = process_info.extdata;

            auto node_it = m_node_info_utility.find(process_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                process_info_ptr->node_info = node_it->second;
            }

            m_process_info_list.push_back(process_info_ptr);
            m_process_info_utility.emplace(process_info.id, process_info_ptr);
        }
    }

    return m_process_info_list;
}

reader_types::thread_info_list_t
reader_t::impl::get_all_threads()
{
    if(m_thread_info_list.empty())
    {
        const auto& statement        = m_read_statements->thread_info_statement();
        const auto  thread_info_list = statement().to_vector();

        m_thread_info_list.reserve(thread_info_list.size());
        for(const auto& thread_info : thread_info_list)
        {
            auto thread_info_ptr = std::make_shared<reader_types::thread_info_t>();
            thread_info_ptr->parent_process_id = thread_info.ppid.value_or(0);
            thread_info_ptr->thread_id         = thread_info.tid;
            thread_info_ptr->name              = thread_info.name.value_or(nullptr);
            thread_info_ptr->start             = thread_info.start.value_or(0);
            thread_info_ptr->end               = thread_info.end.value_or(0);
            thread_info_ptr->extdata           = thread_info.extdata;

            auto node_it = m_node_info_utility.find(thread_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                thread_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(thread_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                thread_info_ptr->process_info = process_it->second;
            }

            m_thread_info_list.push_back(thread_info_ptr);
            m_thread_info_utility.emplace(thread_info.id, thread_info_ptr);
        }
    }

    return m_thread_info_list;
}
reader_types::agent_info_list_t
reader_t::impl::get_all_agents()
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

            auto agent_info_ptr        = std::make_shared<reader_types::agent_info_t>();
            agent_info_ptr->agent_type = agent_info.type.value_or(nullptr);
            agent_info_ptr->type_index = agent_info.type_index.value_or(0);
            agent_info_ptr->absolute_index = agent_info.absolute_index.value_or(0);
            agent_info_ptr->logical_index  = agent_info.logical_index.value_or(0);
            agent_info_ptr->uuid           = agent_info.uuid.value_or(0);
            agent_info_ptr->name           = agent_info.name.value_or(nullptr);
            agent_info_ptr->model_name     = agent_info.model_name.value_or(nullptr);
            agent_info_ptr->vendor_name    = agent_info.vendor_name.value_or(nullptr);
            agent_info_ptr->product_name   = agent_info.product_name.value_or(nullptr);
            agent_info_ptr->user_name      = agent_info.user_name.value_or(nullptr);
            agent_info_ptr->extdata        = agent_info.extdata;

            auto node_it = m_node_info_utility.find(agent_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                agent_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(agent_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                agent_info_ptr->process_info = process_it->second;
            }

            m_agent_info_list.push_back(agent_info_ptr);
            m_agent_info_utility.emplace(agent_info.id, agent_info_ptr);
        }
    }

    return m_agent_info_list;
}

reader_types::track_info_list_t
reader_t::impl::get_all_tracks()
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

            auto track_info_ptr     = std::make_shared<reader_types::track_info_t>();
            track_info_ptr->name    = track_name != nullptr ? track_name : "";
            track_info_ptr->extdata = track_info.extdata;

            auto node_it = m_node_info_utility.find(track_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                track_info_ptr->node_info = node_it->second;
            }

            if(track_info.pid.has_value())
            {
                auto process_it = m_process_info_utility.find(track_info.pid.value());
                if(process_it != m_process_info_utility.end() && process_it->second)
                {
                    track_info_ptr->process_info = process_it->second;
                }
            }

            if(track_info.tid.has_value())
            {
                auto thread_it = m_thread_info_utility.find(track_info.tid.value());
                if(thread_it != m_thread_info_utility.end() && thread_it->second)
                {
                    track_info_ptr->thread_info = thread_it->second;
                }
            }

            m_track_info_list.push_back(track_info_ptr);
            m_track_info_utility.emplace(track_info.id, track_info_ptr);
            m_track_ptr_to_db_id.emplace(track_info_ptr, track_info.id);

            topology_key_t topo{ track_info.nid,
                                 track_info.pid.value_or(0),
                                 track_info.tid.value_or(0) };
            m_track_ptr_to_topology.emplace(track_info_ptr, topo);
            m_topology_to_track_ptr.emplace(topo, track_info_ptr);
        }
    }

    return m_track_info_list;
}

reader_types::kernel_symbol_info_list_t
reader_t::impl::get_all_kernel_symbols()
{
    if(m_kernel_symbol_info_list.empty())
    {
        const auto& statement = m_read_statements->kernel_symbol_info_statement();
        const auto  kernel_symbol_info_list = statement().to_vector();

        m_kernel_symbol_info_list.reserve(kernel_symbol_info_list.size());
        for(const auto& kernel_symbol_info : kernel_symbol_info_list)
        {
            auto kernel_symbol_info_ptr =
                std::make_shared<reader_types::kernel_symbol_info_t>();
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
            kernel_symbol_info_ptr->extdata = kernel_symbol_info.extdata;

            auto node_it = m_node_info_utility.find(kernel_symbol_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                kernel_symbol_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(kernel_symbol_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                kernel_symbol_info_ptr->process_info = process_it->second;
            }

            auto code_object_it =
                m_code_object_info_utility.find(kernel_symbol_info.code_object_id);
            if(code_object_it != m_code_object_info_utility.end() &&
               code_object_it->second)
            {
                kernel_symbol_info_ptr->code_object_info = code_object_it->second;
            }

            m_kernel_symbol_info_list.push_back(kernel_symbol_info_ptr);
            m_kernel_symbol_info_utility.emplace(kernel_symbol_info.id,
                                                 kernel_symbol_info_ptr);
        }
    }

    return m_kernel_symbol_info_list;
}

reader_types::code_object_info_list_t
reader_t::impl::get_all_code_objects()
{
    if(m_code_object_info_list.empty())
    {
        const auto& statement = m_read_statements->code_object_info_statement();
        const auto  code_object_info_list = statement().to_vector();

        m_code_object_info_list.reserve(code_object_info_list.size());
        for(const auto& code_object_info : code_object_info_list)
        {
            auto code_object_info_ptr =
                std::make_shared<reader_types::code_object_info_t>();
            code_object_info_ptr->id         = code_object_info.id;
            code_object_info_ptr->uri        = code_object_info.uri.value_or(nullptr);
            code_object_info_ptr->load_base  = code_object_info.load_base.value_or(0);
            code_object_info_ptr->load_size  = code_object_info.load_size.value_or(0);
            code_object_info_ptr->load_delta = code_object_info.load_delta.value_or(0);
            code_object_info_ptr->storage_type =
                code_object_info.storage_type.value_or(nullptr);
            code_object_info_ptr->extdata = code_object_info.extdata;

            auto node_it = m_node_info_utility.find(code_object_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                code_object_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(code_object_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                code_object_info_ptr->process_info = process_it->second;
            }

            if(code_object_info.agent_id.has_value())
            {
                auto agent_it =
                    m_agent_info_utility.find(code_object_info.agent_id.value());
                if(agent_it != m_agent_info_utility.end() && agent_it->second)
                {
                    code_object_info_ptr->agent_info = agent_it->second;
                }
            }

            m_code_object_info_list.push_back(code_object_info_ptr);
            m_code_object_info_utility.emplace(code_object_info.id, code_object_info_ptr);
        }
    }

    return m_code_object_info_list;
}

reader_types::stream_info_list_t
reader_t::impl::get_all_streams()
{
    if(m_stream_info_list.empty())
    {
        const auto& statement        = m_read_statements->stream_info_statement();
        const auto  stream_info_list = statement().to_vector();

        m_stream_info_list.reserve(stream_info_list.size());
        for(const auto& stream_info : stream_info_list)
        {
            auto stream_info_ptr       = std::make_shared<reader_types::stream_info_t>();
            stream_info_ptr->stream_id = stream_info.id;
            stream_info_ptr->name      = stream_info.name.value_or(nullptr);
            stream_info_ptr->extdata   = stream_info.extdata;

            auto node_it = m_node_info_utility.find(stream_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                stream_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(stream_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                stream_info_ptr->process_info = process_it->second;
            }

            m_stream_info_list.push_back(stream_info_ptr);
            m_stream_info_utility.emplace(stream_info.id, stream_info_ptr);
        }
    }

    return m_stream_info_list;
}

reader_types::queue_info_list_t
reader_t::impl::get_all_queues()
{
    if(m_queue_info_list.empty())
    {
        const auto& statement       = m_read_statements->queue_info_statement();
        const auto  queue_info_list = statement().to_vector();

        m_queue_info_list.reserve(queue_info_list.size());
        for(const auto& queue_info : queue_info_list)
        {
            auto queue_info_ptr      = std::make_shared<reader_types::queue_info_t>();
            queue_info_ptr->queue_id = queue_info.id;
            queue_info_ptr->name     = queue_info.name.value_or(nullptr);
            queue_info_ptr->extdata  = queue_info.extdata;

            auto node_it = m_node_info_utility.find(queue_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                queue_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(queue_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                queue_info_ptr->process_info = process_it->second;
            }

            m_queue_info_list.push_back(queue_info_ptr);
            m_queue_info_utility.emplace(queue_info.id, queue_info_ptr);
        }
    }

    return m_queue_info_list;
}

reader_types::pmc_info_list_t
reader_t::impl::get_all_pmc_infos()
{
    if(m_pmc_info_list.empty())
    {
        const auto& statement     = m_read_statements->pmc_info_statement();
        const auto  pmc_info_list = statement().to_vector();

        m_pmc_info_list.reserve(pmc_info_list.size());
        for(const auto& pmc_info : pmc_info_list)
        {
            auto pmc_info_ptr  = std::make_shared<reader_types::pmc_info_t>();
            pmc_info_ptr->name = pmc_info.name;

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

            auto node_it = m_node_info_utility.find(pmc_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                pmc_info_ptr->node_info = node_it->second;
            }

            auto process_it = m_process_info_utility.find(pmc_info.pid);
            if(process_it != m_process_info_utility.end() && process_it->second)
            {
                pmc_info_ptr->process_info = process_it->second;
            }

            if(pmc_info.agent_id.has_value())
            {
                auto agent_it = m_agent_info_utility.find(pmc_info.agent_id.value());
                if(agent_it != m_agent_info_utility.end() && agent_it->second)
                {
                    pmc_info_ptr->agent_info = agent_it->second;
                }
            }

            m_pmc_info_list.push_back(pmc_info_ptr);
            m_pmc_info_utility.emplace(pmc_info.id, pmc_info_ptr);
        }
    }

    return m_pmc_info_list;
}

reader_types::timeline_event_list_t
reader_t::impl::build_timeline_events(
    const std::vector<data_storage::schema_v3::timeline_event_result>& results,
    reader_types::event_type_t                                         type)
{
    reader_types::timeline_event_list_t events;
    events.reserve(results.size());

    for(const auto& result : results)
    {
        reader_types::timeline_event_t event;
        event.unique_identifier = { result.id, type };
        event.start_timestamp   = result.start_timestamp;
        event.end_timestamp     = result.end_timestamp;

        if(result.display_name_id.has_value())
        {
            auto it = m_string_info_utility.find(result.display_name_id.value());
            if(it != m_string_info_utility.end())
            {
                event.display_name = it->second;
            }
        }

        if(result.category_id.has_value())
        {
            auto it = m_string_info_utility.find(result.category_id.value());
            if(it != m_string_info_utility.end())
            {
                event.category = it->second;
            }
        }

        // Track resolution: try sample-based track_id first, fall back to topology
        if(result.track_id.has_value())
        {
            auto it = m_track_info_utility.find(result.track_id.value());
            if(it != m_track_info_utility.end())
            {
                event.track = it->second;
            }
        }

        if(!event.track)
        {
            topology_key_t topo{ result.nid,
                                 result.pid.value_or(0),
                                 result.tid.value_or(0) };
            auto           it = m_topology_to_track_ptr.find(topo);
            if(it != m_topology_to_track_ptr.end())
            {
                event.track = it->second;
            }
        }

        events.push_back(std::move(event));
    }

    return events;
}

void
reader_t::impl::apply_pagination(reader_types::timeline_event_list_t& events,
                                 const reader_types::pagination_t&    pagination)
{
    if(pagination.offset.has_value())
    {
        auto off = pagination.offset.value();
        if(off >= events.size())
        {
            events.clear();
            return;
        }
        events.erase(events.begin(), events.begin() + static_cast<ptrdiff_t>(off));
    }

    if(pagination.limit.has_value())
    {
        auto lim = pagination.limit.value();
        if(lim < events.size())
        {
            events.resize(lim);
        }
    }
}

reader_types::timeline_event_list_t
reader_t::impl::get_events(const reader_types::event_filter_t& filter)
{
    reader_types::timeline_event_list_t all_events;

    bool query_all    = filter.types.empty();
    auto should_query = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    auto query_event_type =
        [&](const data_storage::schema_v3::read_statements::timeline_event_statement_set&
                                       stmts,
            reader_types::event_type_t type) {
            std::vector<data_storage::schema_v3::timeline_event_result> results;
            if(has_time)
            {
                results = stmts
                              .time_filtered(filter.time_window.end.value(),
                                             filter.time_window.start.value())
                              .to_vector();
            }
            else
            {
                results = stmts.base().to_vector();
            }

            auto events = build_timeline_events(results, type);
            all_events.insert(all_events.end(),
                              std::make_move_iterator(events.begin()),
                              std::make_move_iterator(events.end()));
        };

    if(should_query(reader_types::event_type_t::region))
    {
        query_event_type(m_read_statements->region_statements(),
                         reader_types::event_type_t::region);
    }

    if(should_query(reader_types::event_type_t::kernel_dispatch))
    {
        query_event_type(m_read_statements->kernel_dispatch_statements(),
                         reader_types::event_type_t::kernel_dispatch);
    }

    if(should_query(reader_types::event_type_t::memory_allocate))
    {
        query_event_type(m_read_statements->memory_allocate_statements(),
                         reader_types::event_type_t::memory_allocate);
    }

    if(should_query(reader_types::event_type_t::memory_copy))
    {
        query_event_type(m_read_statements->memory_copy_statements(),
                         reader_types::event_type_t::memory_copy);
    }

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

reader_types::timeline_event_list_t
reader_t::impl::get_events_for_track(reader_types::track_info_ptr_t      track,
                                     const reader_types::event_filter_t& filter)
{
    if(!track) return {};

    auto topo_it = m_track_ptr_to_topology.find(track);
    if(topo_it == m_track_ptr_to_topology.end()) return {};

    auto db_id_it = m_track_ptr_to_db_id.find(track);
    if(db_id_it == m_track_ptr_to_db_id.end()) return {};

    const auto& topo  = topo_it->second;
    auto        db_id = db_id_it->second;

    reader_types::timeline_event_list_t all_events;

    bool query_all    = filter.types.empty();
    auto should_query = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    auto query_event_type =
        [&](const data_storage::schema_v3::read_statements::timeline_event_statement_set&
                                       stmts,
            reader_types::event_type_t type) {
            std::vector<data_storage::schema_v3::timeline_event_result> results;
            if(has_time)
            {
                results = stmts
                              .track_and_time_filtered(topo.nid,
                                                       topo.pid,
                                                       topo.tid,
                                                       db_id,
                                                       filter.time_window.end.value(),
                                                       filter.time_window.start.value())
                              .to_vector();
            }
            else
            {
                results =
                    stmts.track_filtered(topo.nid, topo.pid, topo.tid, db_id).to_vector();
            }

            auto events = build_timeline_events(results, type);
            all_events.insert(all_events.end(),
                              std::make_move_iterator(events.begin()),
                              std::make_move_iterator(events.end()));
        };

    if(should_query(reader_types::event_type_t::region))
    {
        query_event_type(m_read_statements->region_statements(),
                         reader_types::event_type_t::region);
    }

    if(should_query(reader_types::event_type_t::kernel_dispatch))
    {
        query_event_type(m_read_statements->kernel_dispatch_statements(),
                         reader_types::event_type_t::kernel_dispatch);
    }

    if(should_query(reader_types::event_type_t::memory_allocate))
    {
        query_event_type(m_read_statements->memory_allocate_statements(),
                         reader_types::event_type_t::memory_allocate);
    }

    if(should_query(reader_types::event_type_t::memory_copy))
    {
        query_event_type(m_read_statements->memory_copy_statements(),
                         reader_types::event_type_t::memory_copy);
    }

    apply_pagination(all_events, filter.pagination);
    return all_events;
}

size_t
reader_t::impl::get_event_count(const reader_types::event_filter_t& filter)
{
    return get_events(filter).size();
}

}  // namespace rocstorage
