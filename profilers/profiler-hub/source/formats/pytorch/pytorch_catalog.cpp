// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pytorch_catalog.hpp"

#include "reader_catalog.hpp"

#include <map>
#include <memory>
#include <utility>

namespace profiler_hub::pytorch
{

std::size_t
fill_catalog(reader_catalog_t& catalog, pytorch_store_t& store)
{
    const std::size_t first_name_id = store.cats.size() + 1;
    catalog.strings_by_id.reserve(store.cats.size() + store.names.size());
    for(std::size_t i = 0; i < store.cats.size(); ++i)
    {
        catalog.strings_by_id.emplace(i + 1, std::move(store.cats[i]));
    }
    for(std::size_t i = 0; i < store.names.size(); ++i)
    {
        catalog.strings_by_id.emplace(first_name_id + i, std::move(store.names[i]));
    }
    std::vector<std::string>{}.swap(store.cats);
    std::vector<std::string>{}.swap(store.names);

    auto node      = std::make_shared<reader_types::node_info_t>();
    node->hostname = store.host_name;
    catalog.nodes.push_back(node);
    catalog.nodes_by_id.emplace(node->node_id, node);

    std::map<std::int64_t, reader_types::process_info_ptr_t> process_by_pid;
    std::map<std::uint64_t, reader_types::agent_info_ptr_t>  agent_by_id;
    for(const auto& source : store.tracks)
    {
        process_by_pid.emplace(source.pid, nullptr);
        if(source.category == reader_types::track_kind_t::stream)
        {
            agent_by_id.emplace(source.agent_id, nullptr);
        }
    }
    // Every deviceProperties entry is an agent, also one no event is on; an event device
    // without an entry still gets one. Entries match devices by id and nothing else.
    for(const auto& entry : store.device_names)
    {
        agent_by_id.emplace(static_cast<std::uint64_t>(entry.first) + 1, nullptr);
    }
    // A memory event names its device: a GPU is an agent like the stream tracks' ones,
    // and CPU memory (device type 0) gets a CPU agent.
    bool cpu_memory = false;
    for(const auto& sample : store.memory)
    {
        if(sample.device_type == 0)
        {
            cpu_memory = true;
        }
        else if(sample.device_id >= 0)
        {
            agent_by_id.emplace(static_cast<std::uint64_t>(sample.device_id) + 1,
                                nullptr);
        }
    }
    for(auto& [pid, process] : process_by_pid)
    {
        process       = std::make_shared<reader_types::process_info_t>();
        process->id   = catalog.processes.size() + 1;
        process->pid  = static_cast<std::size_t>(pid);
        const auto it = store.process_names.find(pid);
        if(it != store.process_names.end())
        {
            process->command = it->second;
        }
        process->node_info = node;
        catalog.processes.push_back(process);
        catalog.processes_by_id.emplace(process->id, process);
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
        const auto name = store.device_names.find(static_cast<std::int64_t>(device));
        const auto json = store.device_json.find(static_cast<std::int64_t>(device));
        if(name != store.device_names.end()) agent->product_name = name->second;
        agent->user_name = agent->product_name;
        agent->extdata   = json != store.device_json.end() ? json->second : "{}";
        catalog.agents.push_back(agent);
        catalog.agents_by_id.emplace(agent_id, agent);
    }
    if(cpu_memory)
    {
        auto agent            = std::make_shared<reader_types::agent_info_t>();
        agent->id             = agent_by_id.empty() ? 1 : agent_by_id.rbegin()->first + 1;
        agent->agent_type     = "CPU";
        agent->type_index     = 0;
        agent->absolute_index = 0;
        agent->name           = "CPU 0";
        agent->node_info      = node;
        store.cpu_agent_id    = agent->id;
        catalog.agents.push_back(agent);
        catalog.agents_by_id.emplace(agent->id, agent);
    }

    for(const auto& source : store.tracks)
    {
        auto track          = std::make_shared<reader_types::track_info_t>();
        track->id           = source.id;
        track->name         = source.name;
        track->event_count  = source.rows.size();
        track->agent_id     = source.agent_id;
        track->stream_id    = source.stream_id;
        track->category     = source.category;
        track->start_ts     = source.start_ts;
        track->end_ts       = source.end_ts;
        track->node_info    = node;
        track->process_info = process_by_pid.at(source.pid);
        if(source.category == reader_types::track_kind_t::stream)
        {
            track->db_pid        = track->process_info->id;
            auto stream          = std::make_shared<reader_types::stream_info_t>();
            stream->stream_id    = source.stream_id;
            stream->nid          = node->node_id;
            stream->pid          = track->process_info->id;
            stream->node_info    = node;
            stream->process_info = track->process_info;
            catalog.streams.push_back(stream);
        }
        else
        {
            auto thread               = std::make_shared<reader_types::thread_info_t>();
            thread->id                = catalog.threads.size() + 1;
            thread->thread_id         = static_cast<std::size_t>(source.tid);
            thread->parent_process_id = static_cast<std::size_t>(source.pid);
            const auto it = store.thread_names.find({ source.pid, source.tid });
            if(it != store.thread_names.end())
            {
                thread->name = it->second;
            }
            thread->node_info    = node;
            thread->process_info = track->process_info;
            catalog.threads.push_back(thread);
            catalog.threads_by_id.emplace(thread->id, thread);
            track->thread_info = thread;
        }
        catalog.tracks.push_back(track);
    }
    return first_name_id;
}

}  // namespace profiler_hub::pytorch
