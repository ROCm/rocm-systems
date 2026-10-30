// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_catalog.hpp"

#include "reader_catalog.hpp"

#include <map>
#include <utility>

namespace profiler_hub::pytorch
{
std::shared_ptr<reader_catalog_t>
make_catalog(const loaded_trace_t& trace)
{
    auto catalog = std::make_shared<reader_catalog_t>();

    catalog->string_utility.reserve(trace.cats.size() + trace.names.size());
    for(std::size_t i = 0; i < trace.cats.size(); ++i)
    {
        catalog->string_utility.emplace(i, trace.cats[i]);
    }
    for(std::size_t i = 0; i < trace.names.size(); ++i)
    {
        catalog->string_utility.emplace(k_catalog_name_id_base + i, trace.names[i]);
    }
    return catalog;
}

void
fill_catalog(reader_catalog_t&                 catalog,
             const std::vector<store_track_t>& tracks,
             const loaded_trace_t&             trace)
{
    auto node = std::make_shared<reader_types::node_info_t>();
    catalog.nodes.push_back(node);
    catalog.node_utility.emplace(node->node_id, node);

    std::map<std::uint32_t, reader_types::process_info_ptr_t> process_by_pid;
    std::map<std::uint32_t, reader_types::agent_info_ptr_t>   agent_by_id;
    for(const auto& source : tracks)
    {
        process_by_pid.emplace(source.pid, nullptr);
        if(source.category == PH_TRACK_CATEGORY_STREAM)
        {
            agent_by_id.emplace(source.agent_id, nullptr);
        }
    }

    for(auto& [pid, process] : process_by_pid)
    {
        process       = std::make_shared<reader_types::process_info_t>();
        process->pid  = pid;
        const auto it = trace.process_names.find(static_cast<std::int32_t>(pid));
        if(it != trace.process_names.end())
        {
            process->command = it->second;
        }
        process->node_info = node;
        catalog.processes.push_back(process);
        catalog.process_utility.emplace(catalog.processes.size(), process);
    }

    // The track's agent_id is the device plus one, so 0 stays "not device-specific".
    for(auto& [agent_id, agent] : agent_by_id)
    {
        const std::size_t device = agent_id - 1;
        agent                    = std::make_shared<reader_types::agent_info_t>();
        agent->id                = agent_id;
        agent->agent_type        = "GPU";
        agent->type_index        = device;
        agent->absolute_index    = device;
        agent->name              = "GPU " + std::to_string(device);
        agent->node_info         = node;
        catalog.agents.push_back(agent);
        catalog.agent_utility.emplace(agent_id, agent);
    }

    for(const auto& source : tracks)
    {
        auto track          = std::make_shared<reader_types::track_info_t>();
        track->id           = source.id;
        track->name         = source.name;
        track->event_count  = source.events.size();
        track->agent_id     = source.agent_id;
        track->queue_id     = source.queue_id;
        track->stream_id    = source.stream_id;
        track->start_ts     = source.start_ts;
        track->end_ts       = source.end_ts;
        track->node_info    = node;
        track->process_info = process_by_pid.at(source.pid);

        if(source.category == PH_TRACK_CATEGORY_STREAM)
        {
            track->category = reader_types::track_kind_t::stream;
            track->db_pid   = source.pid;
        }
        else
        {
            track->category = reader_types::track_kind_t::thread;

            auto thread               = std::make_shared<reader_types::thread_info_t>();
            thread->thread_id         = source.tid;
            thread->parent_process_id = source.pid;
            const auto it =
                trace.thread_names.find({ static_cast<std::int32_t>(source.pid),
                                          static_cast<std::int32_t>(source.tid) });
            if(it != trace.thread_names.end())
            {
                thread->name = it->second;
            }
            thread->node_info    = node;
            thread->process_info = track->process_info;
            catalog.threads.push_back(thread);
            catalog.thread_utility.emplace(catalog.threads.size(), thread);
            track->thread_info = thread;
        }

        catalog.tracks.push_back(track);
        catalog.track_utility.emplace(track->id, track);
    }
}
}  // namespace profiler_hub::pytorch
