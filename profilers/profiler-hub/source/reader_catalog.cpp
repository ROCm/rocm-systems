// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_catalog.hpp"
#include "common/debug.hpp"

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace profiler_hub
{

void
reader_catalog_t::build_all(data_storage::schema_v3::read_statements& stmts)
{
    build_string_list(stmts);
    build_nodes(stmts);
    build_processes(stmts);
    build_threads(stmts);

    build_agents(stmts);
    build_tracks(stmts);
    build_code_objects(stmts);
    build_kernel_symbols(stmts);
    build_streams(stmts);
    build_queues(stmts);
    build_pmc_infos(stmts);
}

void
reader_catalog_t::build_string_list(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement   = stmts.string_statement();
    const auto  string_list = statement().to_vector();

    string_utility.reserve(string_list.size());
    for(const auto& string : string_list)
    {
        string_utility.emplace(string.id, string.value);
    }
}

void
reader_catalog_t::build_nodes(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement      = stmts.node_info_statement();
    const auto  node_info_list = statement().to_vector();

    nodes.reserve(node_info_list.size());
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

        nodes.push_back(node_info_ptr);
        node_utility.emplace(node_info.node_id, node_info_ptr);
    }
}

void
reader_catalog_t::build_processes(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement         = stmts.process_info_statement();
    const auto  process_info_list = statement().to_vector();

    processes.reserve(process_info_list.size());
    for(const auto& process_info : process_info_list)
    {
        auto process_info_ptr         = std::make_shared<reader_types::process_info_t>();
        process_info_ptr->ppid        = process_info.ppid;
        process_info_ptr->pid         = process_info.pid;
        process_info_ptr->init        = process_info.init;
        process_info_ptr->fini        = process_info.fini;
        process_info_ptr->start       = process_info.start;
        process_info_ptr->end         = process_info.end;
        process_info_ptr->command     = process_info.command.value_or("");
        process_info_ptr->environment = process_info.environment;
        process_info_ptr->extdata     = process_info.extdata;

        const auto node_it = node_utility.find(process_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            process_info_ptr->node_info = node_it->second;
        }

        processes.push_back(process_info_ptr);
        process_utility.emplace(process_info.id, process_info_ptr);
    }
}

void
reader_catalog_t::build_threads(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement        = stmts.thread_info_statement();
    const auto  thread_info_list = statement().to_vector();

    threads.reserve(thread_info_list.size());
    for(const auto& thread_info : thread_info_list)
    {
        auto thread_info_ptr = std::make_shared<reader_types::thread_info_t>();
        thread_info_ptr->parent_process_id = thread_info.ppid;
        thread_info_ptr->thread_id         = thread_info.tid;
        thread_info_ptr->name              = thread_info.name.value_or("");
        thread_info_ptr->start             = thread_info.start;
        thread_info_ptr->end               = thread_info.end;
        thread_info_ptr->extdata           = thread_info.extdata;

        const auto node_it = node_utility.find(thread_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            thread_info_ptr->node_info = node_it->second;
        }

        const auto process_it = process_utility.find(thread_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            thread_info_ptr->process_info = process_it->second;
        }

        threads.push_back(thread_info_ptr);
        thread_utility.emplace(thread_info.id, thread_info_ptr);
    }
}

void
reader_catalog_t::build_agents(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement       = stmts.agent_info_statement();
    const auto  agent_info_list = statement().to_vector();

    agents.reserve(agent_info_list.size());
    for(const auto& agent_info : agent_info_list)
    {
        if(!agent_info.type.has_value() || !agent_info.type_index.has_value())
        {
            LOG_ERROR("Corrupted database detected. Agent type or type index is not "
                      "available for agent info with id: {}",
                      agent_info.id);
            continue;
        }

        auto agent_info_ptr            = std::make_shared<reader_types::agent_info_t>();
        agent_info_ptr->agent_type     = agent_info.type.value();
        agent_info_ptr->type_index     = agent_info.type_index.value();
        agent_info_ptr->absolute_index = agent_info.absolute_index;
        agent_info_ptr->logical_index  = agent_info.logical_index;
        agent_info_ptr->uuid           = agent_info.uuid;
        agent_info_ptr->name           = agent_info.name.value_or("");
        agent_info_ptr->model_name     = agent_info.model_name.value_or("");
        agent_info_ptr->vendor_name    = agent_info.vendor_name.value_or("");
        agent_info_ptr->product_name   = agent_info.product_name.value_or("");
        agent_info_ptr->user_name      = agent_info.user_name.value_or("");
        agent_info_ptr->extdata        = agent_info.extdata;

        auto node_it = node_utility.find(agent_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            agent_info_ptr->node_info = node_it->second;
        }

        auto process_it = process_utility.find(agent_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            agent_info_ptr->process_info = process_it->second;
        }

        agents.push_back(agent_info_ptr);
        agent_utility.emplace(agent_info.id, agent_info_ptr);
    }
}

void
reader_catalog_t::build_tracks(data_storage::schema_v3::read_statements& stmts)
{
    // No rocpd_track dependency: "thread" and "pmc_agent" tracks are derived
    // directly from grouped queries over the raw event tables, same as the
    // 4 optiq-parity category tracks below -- a track only exists if a
    // GROUP BY over real event data produces it, so no zero-event "ghost"
    // tracks are ever created (matches optiq's own discovery philosophy;
    // see rocprofvis_db_profile.cpp's CallBackAddTrack/CallBackLoadTrack,
    // fed by GROUP BY ... COUNT(*) queries).
    const auto thread_track_counts = discover_thread_tracks(stmts);

    // No real db id backs these thread tracks any more; get_events_for_track()
    // still binds a 4th "db_id" parameter for its (now-moot) sample-linked
    // UNION branch, so bind a value that can never collide with a real one.
    constexpr size_t no_db_id = std::numeric_limits<size_t>::max();

    size_t next_id = 0;

    tracks.reserve(thread_track_counts.size());
    for(const auto& [topo, count] : thread_track_counts)
    {
        auto track_info_ptr         = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id          = next_id++;
        track_info_ptr->name        = fmt::format("Thread {}", topo.tid);
        track_info_ptr->event_count = count;

        if(const auto node_it = node_utility.find(topo.nid);
           node_it != node_utility.end())
        {
            track_info_ptr->node_info = node_it->second;
        }
        if(const auto process_it = process_utility.find(topo.pid);
           process_it != process_utility.end())
        {
            track_info_ptr->process_info = process_it->second;
        }
        if(const auto thread_it = thread_utility.find(topo.tid);
           thread_it != thread_utility.end())
        {
            track_info_ptr->thread_info = thread_it->second;
        }

        tracks.push_back(track_info_ptr);
        track_utility.emplace(track_info_ptr->id, track_info_ptr);
        track_to_db_id.emplace(track_info_ptr, no_db_id);
        track_to_topology.emplace(track_info_ptr, topo);
        topology_to_track.emplace(topo, track_info_ptr);
    }

    // PMC/counter tracks: one row per (nid,agent_id,pmc_id), already grouped
    // directly off the counter-sample tables -- naturally one track per
    // device+counter, no per-track-id agent-split bookkeeping needed.
    for(const auto& row : stmts.pmc_track_statement()().to_vector())
    {
        auto track_ptr         = std::make_shared<reader_types::track_info_t>();
        track_ptr->id          = next_id++;
        track_ptr->name        = row.name;
        track_ptr->category    = reader_types::track_kind_t::pmc_agent;
        track_ptr->agent_id    = row.agent_id;
        track_ptr->pmc_id      = row.pmc_id;
        track_ptr->event_count = row.count;

        if(const auto node_it = node_utility.find(row.nid); node_it != node_utility.end())
        {
            track_ptr->node_info = node_it->second;
        }
        if(const auto process_it = process_utility.find(row.pid);
           process_it != process_utility.end())
        {
            track_ptr->process_info = process_it->second;
        }

        tracks.push_back(track_ptr);
        track_utility.emplace(track_ptr->id, track_ptr);
    }

    add_category_tracks(stmts, next_id);
}

void
reader_catalog_t::add_category_tracks(data_storage::schema_v3::read_statements& stmts,
                                      size_t& next_synthetic_id)
{
    auto add_agent_queue_track = [&](reader_types::track_kind_t kind,
                                     std::string                name,
                                     size_t                     nid,
                                     size_t                     agent_id,
                                     size_t                     queue_id,
                                     size_t                     count) {
        auto track_ptr         = std::make_shared<reader_types::track_info_t>();
        track_ptr->id          = next_synthetic_id++;
        track_ptr->category    = kind;
        track_ptr->name        = std::move(name);
        track_ptr->agent_id    = agent_id;
        track_ptr->queue_id    = queue_id;
        track_ptr->event_count = count;

        if(const auto node_it = node_utility.find(nid); node_it != node_utility.end())
        {
            track_ptr->node_info = node_it->second;
        }

        tracks.push_back(track_ptr);
        track_utility.emplace(track_ptr->id, track_ptr);
    };

    auto add_stream_track = [&](reader_types::track_kind_t kind,
                                std::string                name,
                                size_t                     nid,
                                size_t                     pid,
                                size_t                     stream_id,
                                size_t                     count) {
        auto track_ptr         = std::make_shared<reader_types::track_info_t>();
        track_ptr->id          = next_synthetic_id++;
        track_ptr->category    = kind;
        track_ptr->name        = std::move(name);
        track_ptr->stream_id   = stream_id;
        track_ptr->db_pid      = pid;
        track_ptr->event_count = count;

        if(const auto node_it = node_utility.find(nid); node_it != node_utility.end())
        {
            track_ptr->node_info = node_it->second;
        }
        if(const auto process_it = process_utility.find(pid);
           process_it != process_utility.end())
        {
            track_ptr->process_info = process_it->second;
        }

        tracks.push_back(track_ptr);
        track_utility.emplace(track_ptr->id, track_ptr);
    };

    const auto& category_statements = stmts.track_category_statements();

    for(const auto& row : category_statements.kernel_dispatch_agent_queue().to_vector())
    {
        add_agent_queue_track(
            reader_types::track_kind_t::kernel_dispatch_agent_queue,
            fmt::format("Kernel Dispatch [{}] Queue {}", row.agent_id, row.queue_id),
            row.nid,
            row.agent_id,
            row.queue_id,
            row.count);
    }

    for(const auto& row : category_statements.memory_allocate_agent_queue().to_vector())
    {
        add_agent_queue_track(
            reader_types::track_kind_t::memory_allocate_agent_queue,
            fmt::format("Memory Allocate [{}] Queue {}", row.agent_id, row.queue_id),
            row.nid,
            row.agent_id,
            row.queue_id,
            row.count);
    }

    for(const auto& row : category_statements.memory_copy_agent_queue().to_vector())
    {
        add_agent_queue_track(
            reader_types::track_kind_t::memory_copy_agent_queue,
            fmt::format("Memory Copy [{}] Queue {}", row.agent_id, row.queue_id),
            row.nid,
            row.agent_id,
            row.queue_id,
            row.count);
    }

    std::map<std::tuple<size_t, size_t, size_t>, size_t> stream_counts;
    auto accumulate_stream = [&](const auto& rows) {
        for(const auto& row : rows)
        {
            stream_counts[{ row.nid, row.pid, row.stream_id }] += row.count;
        }
    };
    accumulate_stream(category_statements.kernel_dispatch_stream().to_vector());
    accumulate_stream(category_statements.memory_allocate_stream().to_vector());
    accumulate_stream(category_statements.memory_copy_stream().to_vector());

    for(const auto& [key, count] : stream_counts)
    {
        const auto& [nid, pid, stream_id] = key;
        add_stream_track(reader_types::track_kind_t::stream,
                         fmt::format("Stream {}", stream_id),
                         nid,
                         pid,
                         stream_id,
                         count);
    }
}

std::unordered_map<topology_key_t, size_t, topology_key_hash_t>
reader_catalog_t::discover_thread_tracks(data_storage::schema_v3::read_statements& stmts)
{
    std::unordered_map<topology_key_t, size_t, topology_key_hash_t> key_counts;

    auto accumulate = [&](const auto& statement) {
        for(const auto& row : statement().to_vector())
        {
            topology_key_t key{ .nid = row.nid,
                                .pid = row.pid.value_or(0),
                                .tid = row.tid.value_or(0) };
            key_counts[key] += row.count;
        }
    };

    const auto& track_stmts = stmts.track_event_count_statements();
    accumulate(track_stmts.region);
    accumulate(track_stmts.kernel_dispatch);
    accumulate(track_stmts.memory_allocate);
    accumulate(track_stmts.memory_copy);

    return key_counts;
}

void
reader_catalog_t::build_kernel_symbols(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement               = stmts.kernel_symbol_info_statement();
    const auto  kernel_symbol_info_list = statement().to_vector();

    kernel_symbols.reserve(kernel_symbol_info_list.size());
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
        kernel_symbol_info_ptr->sgpr_count       = kernel_symbol_info.sgpr_count;
        kernel_symbol_info_ptr->arch_vgpr_count  = kernel_symbol_info.arch_vgpr_count;
        kernel_symbol_info_ptr->accum_vgpr_count = kernel_symbol_info.accum_vgpr_count;
        kernel_symbol_info_ptr->extdata          = kernel_symbol_info.extdata;

        auto node_it = node_utility.find(kernel_symbol_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            kernel_symbol_info_ptr->node_info = node_it->second;
        }

        auto process_it = process_utility.find(kernel_symbol_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            kernel_symbol_info_ptr->process_info = process_it->second;
        }

        auto code_object_it = code_object_utility.find(kernel_symbol_info.code_object_id);
        if(code_object_it != code_object_utility.end() && code_object_it->second)
        {
            kernel_symbol_info_ptr->code_object_info = code_object_it->second;
        }

        kernel_symbols.push_back(kernel_symbol_info_ptr);
        kernel_symbol_utility.emplace(kernel_symbol_info.id, kernel_symbol_info_ptr);
    }
}

void
reader_catalog_t::build_code_objects(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement             = stmts.code_object_info_statement();
    const auto  code_object_info_list = statement().to_vector();

    code_objects.reserve(code_object_info_list.size());
    for(const auto& code_object_info : code_object_info_list)
    {
        auto code_object_info_ptr = std::make_shared<reader_types::code_object_info_t>();
        code_object_info_ptr->id  = code_object_info.id;
        code_object_info_ptr->uri = code_object_info.uri.value_or("");
        code_object_info_ptr->load_base    = code_object_info.load_base;
        code_object_info_ptr->load_size    = code_object_info.load_size;
        code_object_info_ptr->load_delta   = code_object_info.load_delta;
        code_object_info_ptr->storage_type = code_object_info.storage_type.value_or("");
        code_object_info_ptr->extdata      = code_object_info.extdata;

        auto node_it = node_utility.find(code_object_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            code_object_info_ptr->node_info = node_it->second;
        }

        auto process_it = process_utility.find(code_object_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            code_object_info_ptr->process_info = process_it->second;
        }

        if(code_object_info.agent_id.has_value())
        {
            auto agent_it = agent_utility.find(code_object_info.agent_id.value());
            if(agent_it != agent_utility.end() && agent_it->second)
            {
                code_object_info_ptr->agent_info = agent_it->second;
            }
        }

        code_objects.push_back(code_object_info_ptr);
        code_object_utility.emplace(code_object_info.id, code_object_info_ptr);
    }
}

void
reader_catalog_t::build_streams(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement        = stmts.stream_info_statement();
    const auto  stream_info_list = statement().to_vector();

    streams.reserve(stream_info_list.size());
    for(const auto& stream_info : stream_info_list)
    {
        auto stream_info_ptr       = std::make_shared<reader_types::stream_info_t>();
        stream_info_ptr->stream_id = stream_info.id;
        stream_info_ptr->name      = stream_info.name.value_or("");
        stream_info_ptr->extdata   = stream_info.extdata;

        auto node_it = node_utility.find(stream_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            stream_info_ptr->node_info = node_it->second;
        }

        auto process_it = process_utility.find(stream_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            stream_info_ptr->process_info = process_it->second;
        }

        streams.push_back(stream_info_ptr);
        stream_utility.emplace(stream_info.id, stream_info_ptr);
    }
}

void
reader_catalog_t::build_queues(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement       = stmts.queue_info_statement();
    const auto  queue_info_list = statement().to_vector();

    queues.reserve(queue_info_list.size());
    for(const auto& queue_info : queue_info_list)
    {
        auto queue_info_ptr      = std::make_shared<reader_types::queue_info_t>();
        queue_info_ptr->queue_id = queue_info.id;
        queue_info_ptr->name     = queue_info.name.value_or("");
        queue_info_ptr->extdata  = queue_info.extdata;

        auto node_it = node_utility.find(queue_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            queue_info_ptr->node_info = node_it->second;
        }

        auto process_it = process_utility.find(queue_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            queue_info_ptr->process_info = process_it->second;
        }

        queues.push_back(queue_info_ptr);
        queue_utility.emplace(queue_info.id, queue_info_ptr);
    }
}

void
reader_catalog_t::build_pmc_infos(data_storage::schema_v3::read_statements& stmts)
{
    const auto& statement     = stmts.pmc_info_statement();
    const auto  pmc_info_list = statement().to_vector();

    pmc_infos.reserve(pmc_info_list.size());
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

        auto node_it = node_utility.find(pmc_info.nid);
        if(node_it != node_utility.end() && node_it->second)
        {
            pmc_info_ptr->node_info = node_it->second;
        }

        auto process_it = process_utility.find(pmc_info.pid);
        if(process_it != process_utility.end() && process_it->second)
        {
            pmc_info_ptr->process_info = process_it->second;
        }

        if(pmc_info.agent_id.has_value())
        {
            auto agent_it = agent_utility.find(pmc_info.agent_id.value());
            if(agent_it != agent_utility.end() && agent_it->second)
            {
                pmc_info_ptr->agent_info = agent_it->second;
            }
        }

        pmc_infos.push_back(pmc_info_ptr);
        pmc_utility.emplace(pmc_info.id, pmc_info_ptr);
    }
}

}  // namespace profiler_hub
