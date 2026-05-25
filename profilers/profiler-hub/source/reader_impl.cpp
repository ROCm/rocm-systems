// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_impl.hpp"
#include "json_serializers.hpp"
#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "storage_impl.hpp"

#include "queries/select/table_select_query.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace profiler_hub
{

reader_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage)
: m_storage(storage ? std::move(storage)
                    : throw std::invalid_argument(
                          "Provided pointer to a non-existing storage!"))
, m_backend(m_storage->m_impl->create_database(storage_t::impl::storage_type_t::read))
, m_read_statements(
      std::make_shared<data_storage::schema_v3::read_statements>(m_backend,
                                                                 m_backend->get_uuid()))
{
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
            process_info_ptr->ppid    = process_info.ppid;
            process_info_ptr->pid     = process_info.pid;
            process_info_ptr->init    = process_info.init;
            process_info_ptr->fini    = process_info.fini;
            process_info_ptr->start   = process_info.start;
            process_info_ptr->end     = process_info.end;
            process_info_ptr->command = process_info.command.value_or("");
            process_info_ptr->environment = process_info.environment;
            process_info_ptr->extdata     = process_info.extdata;

            const auto node_it = m_node_info_utility.find(process_info.nid);
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
            thread_info_ptr->parent_process_id = thread_info.ppid;
            thread_info_ptr->thread_id         = thread_info.tid;
            thread_info_ptr->name              = thread_info.name.value_or("");
            thread_info_ptr->start             = thread_info.start;
            thread_info_ptr->end               = thread_info.end;
            thread_info_ptr->extdata           = thread_info.extdata;

            const auto node_it = m_node_info_utility.find(thread_info.nid);
            if(node_it != m_node_info_utility.end() && node_it->second)
            {
                thread_info_ptr->node_info = node_it->second;
            }

            const auto process_it = m_process_info_utility.find(thread_info.pid);
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
            agent_info_ptr->agent_type = agent_info.type.value();
            agent_info_ptr->type_index = agent_info.type_index.value();
            agent_info_ptr->absolute_index = agent_info.absolute_index;
            agent_info_ptr->logical_index  = agent_info.logical_index;
            agent_info_ptr->uuid           = agent_info.uuid;
            agent_info_ptr->name           = agent_info.name.value_or("");
            agent_info_ptr->model_name     = agent_info.model_name.value_or("");
            agent_info_ptr->vendor_name    = agent_info.vendor_name.value_or("");
            agent_info_ptr->product_name   = agent_info.product_name.value_or("");
            agent_info_ptr->user_name      = agent_info.user_name.value_or("");
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
            kernel_symbol_info_ptr->id   = kernel_symbol_info.id;
            kernel_symbol_info_ptr->name = kernel_symbol_info.kernel_name.value_or("");
            kernel_symbol_info_ptr->display_name =
                kernel_symbol_info.display_name.value_or("");
            kernel_symbol_info_ptr->kernel_object = kernel_symbol_info.kernel_object;
            kernel_symbol_info_ptr->kernarg_segment_size =
                kernel_symbol_info.kernarg_segment_size;
            kernel_symbol_info_ptr->kernarg_segment_alignment =
                kernel_symbol_info.kernarg_segment_alignment;
            kernel_symbol_info_ptr->group_segment_size =
                kernel_symbol_info.group_segment_size;
            kernel_symbol_info_ptr->private_segment_size =
                kernel_symbol_info.private_segment_size;
            kernel_symbol_info_ptr->sgpr_count      = kernel_symbol_info.sgpr_count;
            kernel_symbol_info_ptr->arch_vgpr_count = kernel_symbol_info.arch_vgpr_count;
            kernel_symbol_info_ptr->accum_vgpr_count =
                kernel_symbol_info.accum_vgpr_count;
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
            code_object_info_ptr->uri        = code_object_info.uri.value_or("");
            code_object_info_ptr->load_base  = code_object_info.load_base;
            code_object_info_ptr->load_size  = code_object_info.load_size;
            code_object_info_ptr->load_delta = code_object_info.load_delta;
            code_object_info_ptr->storage_type =
                code_object_info.storage_type.value_or("");
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
            stream_info_ptr->name      = stream_info.name.value_or("");
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
            queue_info_ptr->name     = queue_info.name.value_or("");
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

            pmc_info_ptr->target_arch      = pmc_info.target_arch.value_or("");
            pmc_info_ptr->event_code       = pmc_info.event_code;
            pmc_info_ptr->instance_id      = pmc_info.instance_id;
            pmc_info_ptr->symbol           = pmc_info.symbol;
            pmc_info_ptr->description      = pmc_info.description.value_or("");
            pmc_info_ptr->long_description = pmc_info.long_description.value_or("");
            pmc_info_ptr->component        = pmc_info.component.value_or("");
            pmc_info_ptr->units            = pmc_info.units.value_or("");
            pmc_info_ptr->value_type       = pmc_info.value_type.value_or("");
            pmc_info_ptr->block            = pmc_info.block.value_or("");
            pmc_info_ptr->expression       = pmc_info.expression.value_or("");
            pmc_info_ptr->is_constant      = pmc_info.is_constant;
            pmc_info_ptr->is_derived       = pmc_info.is_derived;
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
    const bool query_all    = filter.types.empty();
    auto       should_count = [&](reader_types::event_type_t t) {
        return query_all || std::find(filter.types.begin(), filter.types.end(), t) !=
                                filter.types.end();
    };

    const bool has_time =
        filter.time_window.start.has_value() && filter.time_window.end.has_value();

    auto run_count = [&](const auto& base_stmt, const auto& time_stmt) -> size_t {
        auto results = has_time ? time_stmt(filter.time_window.end.value(),
                                            filter.time_window.start.value())
                                      .to_vector()
                                : base_stmt().to_vector();
        return results.empty() ? 0 : results.front().count;
    };

    size_t total = 0;
    if(should_count(reader_types::event_type_t::region))
    {
        total += run_count(m_read_statements->region_count(),
                           m_read_statements->region_count_time_filtered());
    }
    if(should_count(reader_types::event_type_t::kernel_dispatch))
    {
        total += run_count(m_read_statements->kernel_dispatch_count(),
                           m_read_statements->kernel_dispatch_count_time_filtered());
    }
    if(should_count(reader_types::event_type_t::memory_copy))
    {
        total += run_count(m_read_statements->memory_copy_count(),
                           m_read_statements->memory_copy_count_time_filtered());
    }
    if(should_count(reader_types::event_type_t::memory_allocate))
    {
        total += run_count(m_read_statements->memory_alloc_count(),
                           m_read_statements->memory_alloc_count_time_filtered());
    }
    return total;
}

// ============================================================================
// Event metadata resolution helpers
// ============================================================================

std::optional<data_storage::schema_v3::event_id_result>
reader_t::impl::resolve_event_metadata(const reader_types::timeline_event_t& event)
{
    auto db_id = event.unique_identifier.id;

    std::vector<data_storage::schema_v3::event_id_result> results;
    switch(event.unique_identifier.type)
    {
        case reader_types::event_type_t::region:
            results = m_read_statements->region_event_id()(db_id).to_vector();
            break;
        case reader_types::event_type_t::kernel_dispatch:
            results = m_read_statements->kernel_dispatch_event_id()(db_id).to_vector();
            break;
        case reader_types::event_type_t::memory_copy:
            results = m_read_statements->memory_copy_event_id()(db_id).to_vector();
            break;
        case reader_types::event_type_t::memory_allocate:
            results = m_read_statements->memory_alloc_event_id()(db_id).to_vector();
            break;
        default: return std::nullopt;
    }

    if(results.empty()) return std::nullopt;
    return results.front();
}

reader_types::event_data_ptr_t
reader_t::impl::build_event_data(
    const data_storage::schema_v3::event_id_result& event_meta)
{
    auto event_data             = std::make_shared<reader_types::event_data_t>();
    event_data->stack_id        = event_meta.stack_id.value_or(0);
    event_data->parent_stack_id = event_meta.parent_stack_id.value_or(0);
    event_data->correlation_id  = event_meta.correlation_id.value_or(0);
    event_data->extdata         = event_meta.event_extdata;

    if(event_meta.category_id.has_value())
    {
        auto it = m_string_info_utility.find(event_meta.category_id.value());
        if(it != m_string_info_utility.end())
        {
            event_data->event_category = it->second;
        }
    }

    event_data->call_stack =
        json_serializers::deserialize_call_stack(event_meta.call_stack);
    event_data->line_info_list =
        json_serializers::deserialize_source_context(event_meta.line_info);

    return event_data;
}

// ============================================================================
// Event detail methods
// ============================================================================

std::optional<reader_types::region_data_t>
reader_t::impl::get_region_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::region)
    {
        return std::nullopt;
    }

    auto results =
        m_read_statements->region_detail()(event.unique_identifier.id).to_vector();
    if(results.empty())
    {
        return std::nullopt;
    }

    const auto& r = results.front();

    reader_types::region_data_t data;
    data.start_timestamp = r.start;
    data.end_timestamp   = r.end;
    data.extdata         = r.extdata;

    if(r.name_id.has_value())
    {
        auto it = m_string_info_utility.find(r.name_id.value());
        if(it != m_string_info_utility.end())
        {
            data.name = it->second;
        }
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value())
        {
            data.event = build_event_data(event_meta.value());
        }
    }

    return data;
}

std::optional<reader_types::kernel_dispatch_data_t>
reader_t::impl::get_kernel_dispatch_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::kernel_dispatch)
        return std::nullopt;

    auto results = m_read_statements->kernel_dispatch_detail()(event.unique_identifier.id)
                       .to_vector();
    if(results.empty()) return std::nullopt;

    const auto& r = results.front();

    reader_types::kernel_dispatch_data_t data;
    data.dispatch_id          = r.dispatch_id;
    data.start_timestamp      = r.start;
    data.end_timestamp        = r.end;
    data.private_segment_size = r.private_segment_size.value_or(0);
    data.group_segment_size   = r.group_segment_size.value_or(0);
    data.workgroup_size_x     = r.workgroup_size_x;
    data.workgroup_size_y     = r.workgroup_size_y;
    data.workgroup_size_z     = r.workgroup_size_z;
    data.grid_size_x          = r.grid_size_x;
    data.grid_size_y          = r.grid_size_y;
    data.grid_size_z          = r.grid_size_z;
    data.extdata              = r.extdata;

    if(r.region_name_id.has_value())
    {
        auto it = m_string_info_utility.find(r.region_name_id.value());
        if(it != m_string_info_utility.end()) data.name = it->second;
    }

    if(r.kernel_id.has_value())
    {
        auto it = m_kernel_symbol_info_utility.find(r.kernel_id.value());
        if(it != m_kernel_symbol_info_utility.end())
        {
            data.kernel_symbol_info = it->second;
            if(it->second && it->second->code_object_info)
                data.code_object_info = it->second->code_object_info;
        }
    }

    auto node_it = m_node_info_utility.find(r.nid);
    if(node_it != m_node_info_utility.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_process_info_utility.find(r.pid.value());
        if(it != m_process_info_utility.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_thread_info_utility.find(r.tid.value());
        if(it != m_thread_info_utility.end()) data.thread_info = it->second;
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value()) data.event = build_event_data(event_meta.value());
    }

    return data;
}

std::optional<reader_types::memory_copy_data_t>
reader_t::impl::get_memory_copy_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::memory_copy)
        return std::nullopt;

    auto results =
        m_read_statements->memory_copy_detail()(event.unique_identifier.id).to_vector();
    if(results.empty()) return std::nullopt;

    const auto& r = results.front();

    reader_types::memory_copy_data_t data;
    data.start_timestamp = r.start;
    data.end_timestamp   = r.end;
    data.dst_address     = r.dst_address;
    data.src_address     = r.src_address;
    data.size            = r.size;
    data.extdata         = r.extdata;

    if(r.name_id.has_value())
    {
        auto it = m_string_info_utility.find(r.name_id.value());
        if(it != m_string_info_utility.end()) data.name = it->second;
    }

    if(r.region_name_id.has_value())
    {
        auto it = m_string_info_utility.find(r.region_name_id.value());
        if(it != m_string_info_utility.end()) data.region_name = it->second;
    }

    if(r.dst_agent_id.has_value())
    {
        auto it = m_agent_info_utility.find(r.dst_agent_id.value());
        if(it != m_agent_info_utility.end()) data.dst_agent_id = it->second;
    }

    if(r.src_agent_id.has_value())
    {
        auto it = m_agent_info_utility.find(r.src_agent_id.value());
        if(it != m_agent_info_utility.end()) data.src_agent_id = it->second;
    }

    auto node_it = m_node_info_utility.find(r.nid);
    if(node_it != m_node_info_utility.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_process_info_utility.find(r.pid.value());
        if(it != m_process_info_utility.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_thread_info_utility.find(r.tid.value());
        if(it != m_thread_info_utility.end()) data.thread_info = it->second;
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value()) data.event = build_event_data(event_meta.value());
    }

    return data;
}

std::optional<reader_types::memory_alloc_data_t>
reader_t::impl::get_memory_alloc_details(const reader_types::timeline_event_t& event)
{
    if(event.unique_identifier.type != reader_types::event_type_t::memory_allocate)
        return std::nullopt;

    auto results =
        m_read_statements->memory_alloc_detail()(event.unique_identifier.id).to_vector();
    if(results.empty()) return std::nullopt;

    const auto& r = results.front();

    reader_types::memory_alloc_data_t data;
    data.type            = r.type.value_or("");
    data.level           = r.level.value_or("");
    data.start_timestamp = r.start;
    data.end_timestamp   = r.end;
    data.address         = r.address;
    data.size            = r.size;
    data.extdata         = r.extdata;

    auto node_it = m_node_info_utility.find(r.nid);
    if(node_it != m_node_info_utility.end()) data.node_info = node_it->second;

    if(r.pid.has_value())
    {
        auto it = m_process_info_utility.find(r.pid.value());
        if(it != m_process_info_utility.end()) data.process_info = it->second;
    }

    if(r.tid.has_value())
    {
        auto it = m_thread_info_utility.find(r.tid.value());
        if(it != m_thread_info_utility.end()) data.thread_info = it->second;
    }

    if(r.event_id.has_value())
    {
        auto event_meta = resolve_event_metadata(event);
        if(event_meta.has_value()) data.event = build_event_data(event_meta.value());
    }

    return data;
}

// ============================================================================
// Event property methods
// ============================================================================

reader_types::call_stack_t
reader_t::impl::get_call_stack(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value()) return {};

    return json_serializers::deserialize_call_stack(event_meta->call_stack);
}

reader_types::source_context_list_t
reader_t::impl::get_source_context(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value()) return {};

    return json_serializers::deserialize_source_context(event_meta->line_info);
}

reader_types::arg_data_list_t
reader_t::impl::get_arguments(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value() || !event_meta->event_id.has_value()) return {};

    auto results =
        m_read_statements->arg_detail()(event_meta->event_id.value()).to_vector();

    reader_types::arg_data_list_t args;
    args.reserve(results.size());
    for(const auto& r : results)
    {
        auto arg      = std::make_shared<reader_types::arg_data_t>();
        arg->position = r.position;
        arg->type     = r.type;
        arg->name     = r.name;
        arg->value    = r.value;
        arg->extdata  = r.extdata;
        args.push_back(std::move(arg));
    }
    return args;
}

reader_types::timeline_event_list_t
reader_t::impl::get_correlated_events(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value() || !event_meta->stack_id.has_value()) return {};

    auto stack_id          = event_meta->stack_id.value();
    auto excluded_event_id = event_meta->event_id.value_or(0);

    reader_types::timeline_event_list_t all_events;

    const auto& stmts = m_read_statements->correlated_event_statements();

    auto query_type = [&](const auto& stmt, reader_types::event_type_t type) {
        auto results = stmt(stack_id, excluded_event_id).to_vector();
        auto events  = build_timeline_events(results, type);
        all_events.insert(all_events.end(),
                          std::make_move_iterator(events.begin()),
                          std::make_move_iterator(events.end()));
    };

    query_type(stmts.region, reader_types::event_type_t::region);
    query_type(stmts.kernel_dispatch, reader_types::event_type_t::kernel_dispatch);
    query_type(stmts.memory_copy, reader_types::event_type_t::memory_copy);
    query_type(stmts.memory_allocate, reader_types::event_type_t::memory_allocate);

    return all_events;
}

// ============================================================================
// Database metadata methods
// ============================================================================

reader_types::time_window_t
reader_t::impl::get_data_time_range()
{
    size_t global_min = std::numeric_limits<size_t>::max();
    size_t global_max = 0;

    auto process_range = [&](const auto& stmt) {
        auto results = stmt().to_vector();
        if(!results.empty())
        {
            if(results.front().min_start.has_value())
            {
                global_min = std::min(global_min, results.front().min_start.value());
            }
            if(results.front().max_end.has_value())
            {
                global_max = std::max(global_max, results.front().max_end.value());
            }
        }
    };

    process_range(m_read_statements->region_time_range());
    process_range(m_read_statements->kernel_dispatch_time_range());
    process_range(m_read_statements->memory_copy_time_range());
    process_range(m_read_statements->memory_alloc_time_range());

    reader_types::time_window_t window;
    if(global_min != std::numeric_limits<size_t>::max())
    {
        window.start = global_min;
        window.end   = global_max;
    }
    return window;
}

reader_types::event_counts_t
reader_t::impl::get_event_counts(const reader_types::time_window_t& window)
{
    const bool has_time = window.start.has_value() && window.end.has_value();

    auto get_count = [&](const auto& base_stmt, const auto& time_stmt) -> size_t {
        auto results =
            has_time ? time_stmt(window.end.value(), window.start.value()).to_vector()
                     : base_stmt().to_vector();
        return results.empty() ? 0 : results.front().count;
    };

    reader_types::event_counts_t counts;
    counts[reader_types::event_type_t::region] =
        get_count(m_read_statements->region_count(),
                  m_read_statements->region_count_time_filtered());
    counts[reader_types::event_type_t::kernel_dispatch] =
        get_count(m_read_statements->kernel_dispatch_count(),
                  m_read_statements->kernel_dispatch_count_time_filtered());
    counts[reader_types::event_type_t::memory_copy] =
        get_count(m_read_statements->memory_copy_count(),
                  m_read_statements->memory_copy_count_time_filtered());
    counts[reader_types::event_type_t::memory_allocate] =
        get_count(m_read_statements->memory_alloc_count(),
                  m_read_statements->memory_alloc_count_time_filtered());
    return counts;
}

}  // namespace profiler_hub
