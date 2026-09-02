// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "reader_impl.hpp"
#include "interval_layout.hpp"
#include "json_serializers.hpp"
#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"
#include "storage_impl.hpp"

#include "queries/select/table_select_query.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace profiler_hub
{

namespace
{
// cpu_thread is the only track type with real containment (HIP->HSA call nesting) ->
// `stack`; every other track type overlaps concurrently, not as parent/child -> `lane`.
// schema_version is "3" (v3) or "4.0.0"/"4.1.0" (v4); leading integer is the major
// version. Returns -1 when there is no leading digit, so the caller falls through to
// the metadata-less detection path instead of guessing.
int
schema_major_from_version(const std::string& version)
{
    std::size_t i = 0;
    while(i < version.size() && version[i] >= '0' && version[i] <= '9')
        ++i;
    if(i == 0) return -1;
    return std::stoi(version.substr(0, i));
}

reader_types::nesting_model_t
nesting_for(reader_types::track_type_t t)
{
    switch(t)
    {
        case reader_types::track_type_t::cpu_thread:
            return reader_types::nesting_model_t::stack;
        case reader_types::track_type_t::gpu_queue:
        case reader_types::track_type_t::dma:
        case reader_types::track_type_t::memory:
        case reader_types::track_type_t::stream:
        case reader_types::track_type_t::counter:
        case reader_types::track_type_t::kernel_dispatch_pmc:
        case reader_types::track_type_t::memory_activity:
        default: return reader_types::nesting_model_t::lane;
    }
}
}  // namespace

reader_t::impl::impl(std::unique_ptr<profiler_hub::storage_t> storage)
: m_storage(storage ? std::move(storage)
                    : throw std::invalid_argument(
                          "Provided pointer to a non-existing storage!"))
, m_backend(m_storage->m_impl->create_database(storage_t::impl::storage_type_t::read))
{
    // schema_version_major is not used: it is an unsubstituted template placeholder on
    // templated captures, not a concrete value; schema_version itself is always concrete.
    const auto uuid = m_backend->get_uuid();

    auto object_exists = [&](const std::string& name) {
        auto probe =
            m_backend->create_read_statement_executor<data_storage::count_result>(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='" + name + "';",
                &data_storage::count_result::count);
        auto rows = probe().to_vector();
        return !rows.empty() && rows.front().count > 0;
    };

    auto read_schema_major = [&](const std::string& metadata_object) {
        auto stmt =
            m_backend->create_read_statement_executor<data_storage::string_result>(
                "SELECT value FROM " + metadata_object + " WHERE tag='schema_version';",
                &data_storage::string_result::value);
        auto rows = stmt().to_vector();
        return rows.empty() ? -1 : schema_major_from_version(rows.front().value);
    };

    int schema_major = -1;
    if(object_exists("rocpd_metadata"))
        schema_major = read_schema_major("rocpd_metadata");
    else if(object_exists("rocpd_metadata_" + uuid))
        schema_major = read_schema_major("rocpd_metadata_" + uuid);

    if(schema_major >= 0)
    {
        m_is_v4 = (schema_major == 4);
    }
    else
    {
        auto detect =
            m_backend->create_read_statement_executor<data_storage::count_result>(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rocpd_"
                "timestamp_" +
                    uuid + "';",
                &data_storage::count_result::count);
        auto rows = detect().to_vector();
        m_is_v4   = !rows.empty() && rows.front().count > 0;
    }

    if(m_is_v4)
    {
        m_read_statements =
            std::make_shared<data_storage::schema_v4::read_statements>(m_backend, uuid);
    }
    else
    {
        m_read_statements =
            std::make_shared<data_storage::schema_v3::read_statements>(m_backend, uuid);
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
    m_track_info_list         = get_tracks();

    // Stamp each track's nesting_model once the list is built. max_lane is filled lazily
    // on first get_interval_track.
    for(auto& t : m_track_info_list)
    {
        if(t) t->nesting = nesting_for(t->type);
    }
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
            agent_info_ptr->id         = agent_info.id;
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
reader_t::impl::get_tracks()
{
    if(m_track_info_list.empty())
    {
        if(m_is_v4)
        {
            build_v4_tracks();
            return m_track_info_list;
        }

        // v3 rocpd_track is not a reliable cpu_thread registry, so cpu_thread/region
        // tracks are NOT taken from it — they are synthesized from rocpd_region in
        // synthesize_derived_tracks(). rocpd_track is used here ONLY to emit counter
        // tracks: a track is a counter track iff a PMC-backed rocpd_sample references it
        // (Q10 — v3 counters key on rocpd_track via rocpd_sample.track_id). Region
        // timer-sample tracks (no rocpd_pmc_event) are excluded by
        // distinct_sample_track_ids().
        std::unordered_set<size_t> counter_track_ids;
        for(const auto& r : m_read_statements->distinct_sample_track_ids()().to_vector())
        {
            counter_track_ids.insert(r.track_id);
        }

        std::unordered_map<size_t, size_t>      counter_track_pmc_ids;
        std::unordered_map<size_t, std::string> counter_track_names;
        for(const auto& r : m_read_statements->counter_track_names()().to_vector())
        {
            counter_track_pmc_ids.emplace(r.track_id, r.pmc_id);
            counter_track_names.emplace(r.track_id, r.name);
        }

        const auto& statement       = m_read_statements->track_info_statement();
        const auto  track_info_list = statement().to_vector();

        m_track_info_list.reserve(track_info_list.size());
        for(const auto& track_info : track_info_list)
        {
            const bool is_counter =
                counter_track_ids.find(track_info.id) != counter_track_ids.end();
            if(!is_counter) continue;

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
            track_info_ptr->id      = reader_types::track_id_t{ track_info.id };
            track_info_ptr->name    = track_name != nullptr ? track_name : "";
            track_info_ptr->extdata = track_info.extdata;
            track_info_ptr->type    = reader_types::track_type_t::counter;

            {
                auto nit = counter_track_names.find(track_info.id);
                if(nit != counter_track_names.end() && !nit->second.empty())
                {
                    track_info_ptr->name = nit->second;
                }
            }

            {
                auto pit = counter_track_pmc_ids.find(track_info.id);
                if(pit != counter_track_pmc_ids.end())
                {
                    auto pmit = m_pmc_info_utility.find(pit->second);
                    if(pmit != m_pmc_info_utility.end())
                        track_info_ptr->pmc_info = pmit->second;
                }
            }

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
            // v3 rocpd_track has no agent_id, so counter tracks get no agent_info.

            m_track_info_list.push_back(track_info_ptr);
            m_track_info_utility.emplace(track_info.id, track_info_ptr);
            m_track_ptr_to_db_id.emplace(track_info_ptr, track_info.id);

            topology_key_t topo{ track_info.nid,
                                 track_info.pid.value_or(0),
                                 track_info.tid.value_or(0) };
            m_track_ptr_to_topology.emplace(track_info_ptr, topo);
            m_topology_to_track_ptr.emplace(topo, track_info_ptr);

            track_query_info_t qi;
            qi.type          = track_info_ptr->type;
            qi.nid           = track_info.nid;
            qi.pid           = track_info.pid.value_or(0);
            qi.tid           = track_info.tid;
            qi.real_track_id = track_info.id;
            m_track_query_info.emplace(track_info.id, qi);
        }

        synthesize_derived_tracks();
    }

    return m_track_info_list;
}

// v3 has no rocpd_track rows for gpu_queue or dma; synthesized here from
// rocpd_kernel_dispatch / rocpd_memory_copy topology.
void
reader_t::impl::synthesize_derived_tracks()
{
    size_t next_id = 1;
    {
        const auto max_rows = m_read_statements->max_track_id()().to_vector();
        if(!max_rows.empty() && max_rows.front().max_id.has_value())
        {
            next_id = max_rows.front().max_id.value() + 1;
        }
    }

    // gpu_queue: one track per distinct (nid, pid, agent_id, queue_id).
    for(const auto& g : m_read_statements->distinct_gpu_queue_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::gpu_queue;

        auto node_it = m_node_info_utility.find(g.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(g.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        auto agent_it = m_agent_info_utility.find(g.agent_id);
        if(agent_it != m_agent_info_utility.end() && agent_it->second)
        {
            track_info_ptr->agent_info = agent_it->second;
        }
        auto queue_it = m_queue_info_utility.find(g.queue_id);
        if(queue_it != m_queue_info_utility.end() && queue_it->second)
        {
            track_info_ptr->queue_info = queue_it->second;
        }

        if(track_info_ptr->queue_info)
        {
            track_info_ptr->name = track_info_ptr->queue_info->name;
        }

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::gpu_queue;
        qi.nid      = g.nid;
        qi.pid      = g.pid;
        qi.agent_id = g.agent_id;
        qi.queue_id = g.queue_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // dma: one track per distinct (nid, pid, queue_id, dst_agent_id); NULL is a distinct
    // group value. Keyed on dst_agent_id to match Optiq's GetRocprofMemoryCopyTrackQuery
    // grouping (stream-level grouping is the separate `stream` track type). stream_info
    // is left nullopt here.
    for(const auto& d : m_read_statements->distinct_dma_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::dma;

        auto node_it = m_node_info_utility.find(d.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(d.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        if(d.dst_agent_id.has_value())
        {
            auto agent_it = m_agent_info_utility.find(d.dst_agent_id.value());
            if(agent_it != m_agent_info_utility.end() && agent_it->second)
            {
                track_info_ptr->agent_info = agent_it->second;
            }
        }

        track_info_ptr->name = "Memory copy";

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::dma;
        qi.nid      = d.nid;
        qi.pid      = d.pid;
        qi.queue_id = d.queue_id;
        qi.agent_id = d.dst_agent_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // memory: one track per distinct (nid, agent_id, queue_id, pid) in
    // rocpd_memory_allocate. Keyed to match Optiq's GetRocprofMemoryAllocTrackQuery GROUP
    // BY exactly. NULL agent_id / queue_id are distinct group values, not dropped.
    for(const auto& m : m_read_statements->distinct_memory_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::memory;

        auto node_it = m_node_info_utility.find(m.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(m.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        if(m.agent_id.has_value())
        {
            auto agent_it = m_agent_info_utility.find(m.agent_id.value());
            if(agent_it != m_agent_info_utility.end() && agent_it->second)
            {
                track_info_ptr->agent_info = agent_it->second;
            }
        }
        if(m.queue_id.has_value())
        {
            auto queue_it = m_queue_info_utility.find(m.queue_id.value());
            if(queue_it != m_queue_info_utility.end() && queue_it->second)
            {
                track_info_ptr->queue_info = queue_it->second;
            }
        }

        track_info_ptr->name = "Memory allocation";

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::memory;
        qi.nid      = m.nid;
        qi.pid      = m.pid;
        qi.agent_id = m.agent_id;
        qi.queue_id = m.queue_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // kernel_dispatch_pmc: one track per distinct (nid, agent_id, pmc_id, pid) from
    // rocpd_pmc_event JOIN rocpd_kernel_dispatch. Keyed to match Optiq's
    // GetRocprofPerformanceCountersTrackQuery GROUP BY.
    for(const auto& k : m_read_statements->distinct_kd_pmc_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::kernel_dispatch_pmc;

        auto node_it = m_node_info_utility.find(k.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(k.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        auto agent_it = m_agent_info_utility.find(k.agent_id);
        if(agent_it != m_agent_info_utility.end() && agent_it->second)
        {
            track_info_ptr->agent_info = agent_it->second;
        }

        auto pmc_it = m_pmc_info_utility.find(k.pmc_id);
        if(pmc_it != m_pmc_info_utility.end() && pmc_it->second)
        {
            track_info_ptr->pmc_info = pmc_it->second;
        }

        track_info_ptr->name = "Kernel dispatch PMC";

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::kernel_dispatch_pmc;
        qi.nid      = k.nid;
        qi.pid      = k.pid;
        qi.agent_id = k.agent_id;
        qi.pmc_id   = k.pmc_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // memory_activity: one scalar track per distinct (nid, pid, agent_id) from
    // rocpd_memory_allocate. Mirrors Optiq's GetRocprofMemoryActivity* per-agent
    // cumulative running sum (load_id 7).
    for(const auto& ma : m_read_statements->distinct_mem_activity_tracks()().to_vector())
    {
        // Skip NULL agent_id rows — FREE rows in v3 may have agent_id=NULL; they
        // participate in running-sum recovery but do not form their own tracks.
        if(!ma.agent_id.has_value()) continue;

        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::memory_activity;

        auto node_it = m_node_info_utility.find(ma.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(ma.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        if(ma.agent_id.has_value())
        {
            auto agent_it = m_agent_info_utility.find(ma.agent_id.value());
            if(agent_it != m_agent_info_utility.end() && agent_it->second)
            {
                track_info_ptr->agent_info = agent_it->second;
            }
        }

        track_info_ptr->name = "Memory activity";

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::memory_activity;
        qi.nid      = ma.nid;
        qi.pid      = ma.pid;
        qi.agent_id = ma.agent_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // stream: one track per distinct (nid, pid, stream_id), aggregating kernel_dispatch +
    // memory_copy + memory_allocate events sharing that stream. Additive to the gpu_queue
    // and dma tracks above — the same events also appear in their per-op tracks, matching
    // Optiq's Stream track.
    for(const auto& s : m_read_statements->distinct_stream_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::stream;

        auto node_it = m_node_info_utility.find(s.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(s.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        auto stream_it = m_stream_info_utility.find(s.stream_id);
        if(stream_it != m_stream_info_utility.end() && stream_it->second)
        {
            track_info_ptr->stream_info = stream_it->second;
        }

        if(track_info_ptr->stream_info && !track_info_ptr->stream_info->name.empty())
        {
            track_info_ptr->name = track_info_ptr->stream_info->name;
        }
        else
        {
            track_info_ptr->name = "Stream " + std::to_string(s.stream_id);
        }

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type      = reader_types::track_type_t::stream;
        qi.nid       = s.nid;
        qi.pid       = s.pid;
        qi.stream_id = s.stream_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // cpu_thread: one track per distinct (nid, pid, tid, is_sample) in rocpd_region.
    // A thread with both plain and sampled regions yields two tracks (main + sample),
    // mirroring roc-optiq's region-main / region-sample split.
    for(const auto& r : m_read_statements->distinct_region_tracks()().to_vector())
    {
        const bool is_sample = r.is_sample != 0;

        auto track_info_ptr         = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id          = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type        = reader_types::track_type_t::cpu_thread;
        track_info_ptr->region_kind = is_sample
                                          ? reader_types::region_track_kind_t::sample
                                          : reader_types::region_track_kind_t::main;

        auto node_it = m_node_info_utility.find(r.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(r.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        auto thread_it = m_thread_info_utility.find(r.tid);
        if(thread_it != m_thread_info_utility.end() && thread_it->second)
        {
            track_info_ptr->thread_info = thread_it->second;
        }

        std::string base_name;
        if(track_info_ptr->thread_info && !track_info_ptr->thread_info->name.empty())
        {
            base_name = track_info_ptr->thread_info->name;
        }
        else if(track_info_ptr->thread_info)
        {
            base_name =
                "Thread " + std::to_string(track_info_ptr->thread_info->thread_id);
        }
        else
        {
            base_name = "Thread";
        }
        track_info_ptr->name = is_sample ? base_name + " (samples)" : base_name;

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        // Topology registration so get_events_for_track() resolves this thread's
        // timeline events by (nid, pid, tid). db_id is the synthetic id: it can never
        // equal a real rocpd_sample.track_id, so the statement's "OR S.track_id = ?"
        // branch stays inert and only the topology match drives results.
        topology_key_t topo{ r.nid, r.pid, r.tid };
        m_track_ptr_to_topology.emplace(track_info_ptr, topo);
        m_topology_to_track_ptr.emplace(topo, track_info_ptr);
        m_track_ptr_to_db_id.emplace(track_info_ptr, track_info_ptr->id.value);

        track_query_info_t qi;
        qi.type             = reader_types::track_type_t::cpu_thread;
        qi.nid              = r.nid;
        qi.pid              = r.pid;
        qi.tid              = r.tid;
        qi.region_is_sample = is_sample;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }
}

// v4.0: every swimlane is a real rocpd_track row carrying the full identity tuple
// (nid, pid, tid, agent_id, queue_id, stream_id). There is nothing to synthesize —
// tracks are read directly and classified from which identity columns are populated,
// with counter tracks identified by being referenced from a PMC-backed rocpd_sample.
void
reader_t::impl::build_v4_tracks()
{
    // Counter tracks: any rocpd_track referenced by a PMC-backed rocpd_sample row (a
    // sample that joins rocpd_pmc_event); non-PMC sample tracks are excluded by
    // distinct_sample_track_ids().
    std::unordered_set<size_t> counter_track_ids;
    for(const auto& r : m_read_statements->distinct_sample_track_ids()().to_vector())
    {
        counter_track_ids.insert(r.track_id);
    }

    // Memory tracks: any rocpd_track referenced by rocpd_memory_allocate. Checked before
    // gpu_queue because both may carry agent_id + queue_id on their rocpd_track row.
    std::unordered_set<size_t> memory_alloc_track_ids;
    for(const auto& r : m_read_statements->memory_alloc_track_ids()().to_vector())
    {
        memory_alloc_track_ids.insert(r.track_id);
    }

    // Tracks appearing in both discovery sets: counter classification wins (is_counter
    // checked before is_memory), so an overlapping track silently loses its
    // memory-allocate events. This set is telemetry only — it does not change precedence.
    std::unordered_set<size_t> ambiguous_classification_ids;
    for(const auto& id : counter_track_ids)
    {
        if(memory_alloc_track_ids.count(id) > 0)
        {
            ambiguous_classification_ids.insert(id);
            std::cerr << "[profiler-hub] WARNING: track_id " << id
                      << " appears in both counter (rocpd_sample/pmc) and memory-allocate"
                         " (rocpd_memory_allocate) discovery sets; classifying as counter"
                         " (existing precedence). Memory-allocate events for this track"
                         " will not appear in the memory track type.\n";
        }
    }

    std::unordered_map<size_t, size_t>      counter_track_pmc_ids;
    std::unordered_map<size_t, std::string> counter_track_names;
    for(const auto& r : m_read_statements->counter_track_names()().to_vector())
    {
        counter_track_pmc_ids.emplace(r.track_id, r.pmc_id);
        counter_track_names.emplace(r.track_id, r.name);
    }

    const auto& statement       = m_read_statements->track_info_statement();
    const auto  track_info_list = statement().to_vector();

    m_track_info_list.reserve(track_info_list.size());
    for(const auto& track_info : track_info_list)
    {
        auto track_info_ptr     = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id      = reader_types::track_id_t{ track_info.id };
        track_info_ptr->extdata = track_info.extdata;

        std::string track_name;
        if(track_info.name_id.has_value())
        {
            auto sit = m_string_info_utility.find(track_info.name_id.value());
            if(sit != m_string_info_utility.end()) track_name = sit->second;
        }

        const bool is_counter =
            counter_track_ids.find(track_info.id) != counter_track_ids.end();
        const bool is_memory =
            memory_alloc_track_ids.find(track_info.id) != memory_alloc_track_ids.end();
        if(is_counter)
        {
            track_info_ptr->type = reader_types::track_type_t::counter;
            if(ambiguous_classification_ids.count(track_info.id) > 0)
                track_info_ptr->ambiguous_classification = true;
        }
        else if(is_memory)
        {
            track_info_ptr->type = reader_types::track_type_t::memory;
        }
        else if(track_info.queue_id.has_value())
        {
            track_info_ptr->type = reader_types::track_type_t::gpu_queue;
        }
        else if(track_info.stream_id.has_value())
        {
            track_info_ptr->type = reader_types::track_type_t::dma;
        }
        else
        {
            track_info_ptr->type = reader_types::track_type_t::cpu_thread;
        }

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

        // v4 rocpd_track carries agent_id directly, so counter and queue tracks resolve
        // agent_info directly from the track row (v3 has no such column — see above).
        if(track_info.agent_id.has_value())
        {
            auto agent_it = m_agent_info_utility.find(track_info.agent_id.value());
            if(agent_it != m_agent_info_utility.end() && agent_it->second)
            {
                track_info_ptr->agent_info = agent_it->second;
            }
        }

        if(track_info.queue_id.has_value())
        {
            auto queue_it = m_queue_info_utility.find(track_info.queue_id.value());
            if(queue_it != m_queue_info_utility.end() && queue_it->second)
            {
                track_info_ptr->queue_info = queue_it->second;
            }
        }

        if(track_info.stream_id.has_value())
        {
            auto stream_it = m_stream_info_utility.find(track_info.stream_id.value());
            if(stream_it != m_stream_info_utility.end() && stream_it->second)
            {
                track_info_ptr->stream_info = stream_it->second;
            }
        }

        track_info_ptr->name = track_name;
        if(is_counter)
        {
            auto nit = counter_track_names.find(track_info.id);
            if(nit != counter_track_names.end() && !nit->second.empty())
            {
                track_info_ptr->name = nit->second;
            }
            auto pit = counter_track_pmc_ids.find(track_info.id);
            if(pit != counter_track_pmc_ids.end())
            {
                auto pmit = m_pmc_info_utility.find(pit->second);
                if(pmit != m_pmc_info_utility.end())
                    track_info_ptr->pmc_info = pmit->second;
            }
        }
        else if(track_info_ptr->name.empty())
        {
            if(track_info_ptr->type == reader_types::track_type_t::gpu_queue &&
               track_info_ptr->queue_info)
            {
                track_info_ptr->name = track_info_ptr->queue_info->name;
            }
            else if(track_info_ptr->type == reader_types::track_type_t::memory)
            {
                track_info_ptr->name = "Memory allocation";
            }
            else if(track_info_ptr->type == reader_types::track_type_t::dma)
            {
                track_info_ptr->name = "Memory copy";
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

        // Routing for track-scoped queries: v4 anchors everything on the real
        // rocpd_track id, so real_track_id == the track's own id.
        track_query_info_t qi;
        qi.type          = track_info_ptr->type;
        qi.nid           = track_info.nid;
        qi.pid           = track_info.pid.value_or(0);
        qi.tid           = track_info.tid;
        qi.agent_id      = track_info.agent_id;
        qi.queue_id      = track_info.queue_id;
        qi.stream_id     = track_info.stream_id;
        qi.real_track_id = track_info.id;
        m_track_query_info.emplace(track_info.id, qi);
    }

    // stream: one track per distinct (nid, pid, stream_id), aggregating kernel_dispatch +
    // memory_copy + memory_allocate events sharing that stream. Unlike the 1:1 tracks
    // above, a stream spans multiple rocpd_track rows, so these are synthesized with ids
    // allocated above the max real rocpd_track id — never colliding with a real track.
    // Additive: the same events also appear in their per-op tracks, matching Optiq.
    size_t next_id = 1;
    for(const auto& track_info : track_info_list)
    {
        if(track_info.id >= next_id) next_id = track_info.id + 1;
    }
    for(const auto& s : m_read_statements->distinct_stream_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::stream;

        auto node_it = m_node_info_utility.find(s.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(s.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        auto stream_it = m_stream_info_utility.find(s.stream_id);
        if(stream_it != m_stream_info_utility.end() && stream_it->second)
        {
            track_info_ptr->stream_info = stream_it->second;
        }

        if(track_info_ptr->stream_info && !track_info_ptr->stream_info->name.empty())
        {
            track_info_ptr->name = track_info_ptr->stream_info->name;
        }
        else
        {
            track_info_ptr->name = "Stream " + std::to_string(s.stream_id);
        }

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type      = reader_types::track_type_t::stream;
        qi.nid       = s.nid;
        qi.pid       = s.pid;
        qi.stream_id = s.stream_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // kernel_dispatch_pmc: one track per distinct (nid, agent_id, pmc_id, pid) from
    // rocpd_pmc_event JOIN rocpd_kernel_dispatch JOIN rocpd_track. Keyed to match
    // Optiq's GetRocprofPerformanceCountersTrackQuery GROUP BY. Synthesized with ids
    // above max real id so they never collide with the real rocpd_track classification
    // above.
    for(const auto& k : m_read_statements->distinct_kd_pmc_tracks()().to_vector())
    {
        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::kernel_dispatch_pmc;

        auto node_it = m_node_info_utility.find(k.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(k.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        auto agent_it = m_agent_info_utility.find(k.agent_id);
        if(agent_it != m_agent_info_utility.end() && agent_it->second)
        {
            track_info_ptr->agent_info = agent_it->second;
        }
        auto pmc_it = m_pmc_info_utility.find(k.pmc_id);
        if(pmc_it != m_pmc_info_utility.end() && pmc_it->second)
        {
            track_info_ptr->pmc_info = pmc_it->second;
        }

        track_info_ptr->name = "Kernel dispatch PMC";

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::kernel_dispatch_pmc;
        qi.nid      = k.nid;
        qi.pid      = k.pid;
        qi.agent_id = k.agent_id;
        qi.pmc_id   = k.pmc_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }

    // memory_activity: one scalar track per distinct (nid, pid, agent_id) from
    // rocpd_memory_allocate JOIN rocpd_track. Synthesized above max real id.
    for(const auto& ma : m_read_statements->distinct_mem_activity_tracks()().to_vector())
    {
        if(!ma.agent_id.has_value()) continue;

        auto track_info_ptr  = std::make_shared<reader_types::track_info_t>();
        track_info_ptr->id   = reader_types::track_id_t{ next_id++ };
        track_info_ptr->type = reader_types::track_type_t::memory_activity;

        auto node_it = m_node_info_utility.find(ma.nid);
        if(node_it != m_node_info_utility.end() && node_it->second)
        {
            track_info_ptr->node_info = node_it->second;
        }
        auto process_it = m_process_info_utility.find(ma.pid);
        if(process_it != m_process_info_utility.end() && process_it->second)
        {
            track_info_ptr->process_info = process_it->second;
        }
        if(ma.agent_id.has_value())
        {
            auto agent_it = m_agent_info_utility.find(ma.agent_id.value());
            if(agent_it != m_agent_info_utility.end() && agent_it->second)
            {
                track_info_ptr->agent_info = agent_it->second;
            }
        }

        track_info_ptr->name = "Memory activity";

        m_track_info_list.push_back(track_info_ptr);
        m_track_info_utility.emplace(track_info_ptr->id.value, track_info_ptr);

        track_query_info_t qi;
        qi.type     = reader_types::track_type_t::memory_activity;
        qi.nid      = ma.nid;
        qi.pid      = ma.pid;
        qi.agent_id = ma.agent_id;
        m_track_query_info.emplace(track_info_ptr->id.value, qi);
    }
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

        // Build a set of pmc_ids that have >1 rocpd_pmc_event row per event_id.
        std::unordered_set<size_t> ambiguous_ids;
        for(const auto& row : m_read_statements->ambiguous_pmc_ids()().to_vector())
            ambiguous_ids.insert(row.pmc_id);

        m_pmc_info_list.reserve(pmc_info_list.size());
        for(const auto& pmc_info : pmc_info_list)
        {
            auto pmc_info_ptr    = std::make_shared<reader_types::pmc_info_t>();
            pmc_info_ptr->pmc_id = pmc_info.id;
            pmc_info_ptr->name   = pmc_info.name;

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

            pmc_info_ptr->ambiguous = (ambiguous_ids.count(pmc_info.id) > 0);

            m_pmc_info_list.push_back(pmc_info_ptr);
            m_pmc_info_utility.emplace(pmc_info.id, pmc_info_ptr);
        }
    }

    return m_pmc_info_list;
}

reader_types::timeline_event_list_t
reader_t::impl::build_timeline_events(
    const std::vector<data_storage::timeline_event_result>& results,
    reader_types::event_type_t                              type)
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

        if(result.category_name.has_value())
        {
            event.category = result.category_name.value();
        }

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
        [&](const data_storage::read_statements_base::timeline_event_statement_set& stmts,
            reader_types::event_type_t type) {
            std::vector<data_storage::timeline_event_result> results;
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
        [&](const data_storage::read_statements_base::timeline_event_statement_set& stmts,
            reader_types::event_type_t type) {
            std::vector<data_storage::timeline_event_result> results;
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

std::optional<data_storage::event_id_result>
reader_t::impl::resolve_event_metadata(const reader_types::timeline_event_t& event)
{
    auto db_id = event.unique_identifier.id;

    // The backend materializes and decodes call_stack / line_info into the
    // version-neutral event_id_result, so this path is identical for v3 and v4.
    std::vector<data_storage::event_id_result> results;
    switch(event.unique_identifier.type)
    {
        case reader_types::event_type_t::region:
            results = m_read_statements->region_event_id()(db_id);
            break;
        case reader_types::event_type_t::kernel_dispatch:
            results = m_read_statements->kernel_dispatch_event_id()(db_id);
            break;
        case reader_types::event_type_t::memory_copy:
            results = m_read_statements->memory_copy_event_id()(db_id);
            break;
        case reader_types::event_type_t::memory_allocate:
            results = m_read_statements->memory_alloc_event_id()(db_id);
            break;
        default: return std::nullopt;
    }

    if(results.empty()) return std::nullopt;
    return results.front();
}

reader_types::event_data_ptr_t
reader_t::impl::build_event_data(const data_storage::event_id_result& event_meta)
{
    auto event_data             = std::make_shared<reader_types::event_data_t>();
    event_data->stack_id        = event_meta.stack_id.value_or(0);
    event_data->parent_stack_id = event_meta.parent_stack_id.value_or(0);
    event_data->correlation_id  = event_meta.correlation_id.value_or(0);
    event_data->extdata         = event_meta.event_extdata;

    if(event_meta.category_name.has_value())
    {
        event_data->event_category = event_meta.category_name.value();
    }

    event_data->call_stack     = event_meta.call_stack;
    event_data->line_info_list = event_meta.line_info;

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

    return event_meta->call_stack;
}

reader_types::source_context_list_t
reader_t::impl::get_source_context(const reader_types::timeline_event_t& event)
{
    auto event_meta = resolve_event_metadata(event);
    if(!event_meta.has_value()) return {};

    return event_meta->line_info;
}

// Opaque-handle overloads: build a minimal timeline_event_t from the handle's private
// {row_id, type} and delegate to the overloads above. The decode lives entirely inside
// the reader, so event_id_t opacity is preserved -- no public type/row_id accessor is
// added, and the consumer only ever holds the opaque handle. Types with no stack or
// source context (sample / pmc_event) fall through to the delegates' empty-return case.
reader_types::call_stack_t
reader_t::impl::get_call_stack(const reader_types::event_id_t& id)
{
    reader_types::timeline_event_t ev{};
    ev.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                             reader_types::detail::event_id_access::type(id) };
    return get_call_stack(ev);
}

reader_types::source_context_list_t
reader_t::impl::get_source_context(const reader_types::event_id_t& id)
{
    reader_types::timeline_event_t ev{};
    ev.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                             reader_types::detail::event_id_access::type(id) };
    return get_source_context(ev);
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

// Opaque-handle overload; same bridge as get_call_stack(event_id_t) above.
reader_types::arg_data_list_t
reader_t::impl::get_arguments(const reader_types::event_id_t& id)
{
    reader_types::timeline_event_t ev{};
    ev.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                             reader_types::detail::event_id_access::type(id) };
    return get_arguments(ev);
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
reader_t::impl::get_time_range()
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

namespace
{
// Folds per-name-id GROUP BY rows into one event_summary_t per distinct display name:
// rows whose ids resolve to the same string are merged (multiple kernel_symbol ids can
// share a display_name). Sorted by name for deterministic output.
template <typename ResolveFn>
reader_types::event_summary_list_t
fold_summary_rows(const std::vector<data_storage::summary_result>& rows,
                  ResolveFn                                        resolve_name)
{
    std::map<std::string, reader_types::event_summary_t> by_name;
    for(const auto& r : rows)
    {
        if(r.count == 0) continue;
        std::string name =
            r.name_ref.has_value() ? resolve_name(r.name_ref.value()) : std::string{};
        auto it = by_name.find(name);
        if(it == by_name.end())
        {
            reader_types::event_summary_t s;
            s.name           = name;
            s.count          = r.count;
            s.total_duration = r.total_duration;
            s.min_duration   = r.min_duration;
            s.max_duration   = r.max_duration;
            by_name.emplace(std::move(name), s);
        }
        else
        {
            auto& s = it->second;
            s.count += r.count;
            s.total_duration += r.total_duration;
            s.min_duration = std::min(s.min_duration, r.min_duration);
            s.max_duration = std::max(s.max_duration, r.max_duration);
        }
    }

    reader_types::event_summary_list_t out;
    out.reserve(by_name.size());
    for(auto& [name, s] : by_name)
    {
        s.avg_duration = s.count > 0 ? s.total_duration / s.count : 0;
        out.push_back(std::move(s));
    }
    return out;
}
}  // namespace

reader_types::event_summary_list_t
reader_t::impl::get_kernel_summary(const reader_types::time_window_t& window)
{
    const bool has_time = window.start.has_value() && window.end.has_value();
    auto       rows     = has_time ? m_read_statements
                               ->kernel_summary_time_filtered()(window.end.value(),
                                                                window.start.value())
                               .to_vector()
                                   : m_read_statements->kernel_summary()().to_vector();

    return fold_summary_rows(rows, [&](size_t id) -> std::string {
        auto kit = m_kernel_symbol_info_utility.find(id);
        if(kit != m_kernel_symbol_info_utility.end() && kit->second)
        {
            return !kit->second->display_name.empty() ? kit->second->display_name
                                                      : kit->second->name;
        }
        return {};
    });
}

reader_types::event_summary_list_t
reader_t::impl::get_region_summary(const reader_types::time_window_t& window)
{
    const bool has_time = window.start.has_value() && window.end.has_value();
    auto       rows     = has_time ? m_read_statements
                               ->region_summary_time_filtered()(window.end.value(),
                                                                window.start.value())
                               .to_vector()
                                   : m_read_statements->region_summary()().to_vector();

    return fold_summary_rows(rows, [&](size_t id) -> std::string {
        auto sit = m_string_info_utility.find(id);
        return sit != m_string_info_utility.end() ? sit->second : std::string{};
    });
}

// ============================================================================
// Track-scoped queries (interval / scalar / flow)
// ============================================================================

namespace
{
// Event type of a homogeneous interval track, i.e. which per-type table its row ids
// index. Stream tracks are heterogeneous and resolve type per row from op_kind instead.
reader_types::event_type_t
interval_event_type_for(reader_types::track_type_t t)
{
    switch(t)
    {
        case reader_types::track_type_t::gpu_queue:
            return reader_types::event_type_t::kernel_dispatch;
        case reader_types::track_type_t::dma:
            return reader_types::event_type_t::memory_copy;
        case reader_types::track_type_t::memory:
            return reader_types::event_type_t::memory_allocate;
        case reader_types::track_type_t::kernel_dispatch_pmc:
            // A kd_pmc interval row is identified by the PAIR (kernel_dispatch_id,
            // pmc_id): both v3 and v4 kd_pmc interval SQL SELECT K.id (a
            // rocpd_kernel_dispatch.id), not a rocpd_pmc_event.id. The handle must be
            // typed as kernel_dispatch to resolve correctly; typing it as pmc_event
            // routes K.id to the wrong detail path (WHERE rocpd_pmc_event.id = ?). The
            // specific counter value / pmc_id is not recoverable from the handle alone --
            // an accepted limitation (a consumer clicking a specific pmc track already
            // has that pmc context).
            return reader_types::event_type_t::kernel_dispatch;
        case reader_types::track_type_t::cpu_thread:
        default: return reader_types::event_type_t::region;
    }
}

template <typename T>
void
paginate(std::vector<T>& v, const reader_types::pagination_t& pagination)
{
    if(pagination.offset.has_value())
    {
        auto off = pagination.offset.value();
        if(off >= v.size())
        {
            v.clear();
            return;
        }
        v.erase(v.begin(), v.begin() + static_cast<ptrdiff_t>(off));
    }
    if(pagination.limit.has_value() && pagination.limit.value() < v.size())
    {
        v.resize(pagination.limit.value());
    }
}
}  // namespace

reader_types::interval_entry_list_t
reader_t::impl::get_interval_track(size_t                              track_id,
                                   const reader_types::event_filter_t& filter)
{
    auto qit = m_track_query_info.find(track_id);
    if(qit == m_track_query_info.end()) return {};
    const auto& qi = qit->second;

    std::vector<data_storage::interval_row_result> rows;
    bool                                           name_from_kernel_symbol = false;
    // Stream tracks mix ops, so name resolution is chosen per-row from op_kind rather
    // than a single track-wide flag.
    bool is_stream = false;

    if(m_is_v4)
    {
        // v4.0: every track is a real rocpd_track row, so track-scoped interval reads
        // reduce to WHERE <table>.track_id = real_track_id (start/end resolved through
        // the rocpd_timestamp spine inside the statement).
        switch(qi.type)
        {
            case reader_types::track_type_t::cpu_thread:
                rows = m_read_statements->region_interval_track_v4()(qi.real_track_id)
                           .to_vector();
                break;
            case reader_types::track_type_t::gpu_queue:
                rows = m_read_statements
                           ->kernel_dispatch_interval_track_v4()(qi.real_track_id)
                           .to_vector();
                name_from_kernel_symbol = true;
                break;
            case reader_types::track_type_t::dma:
                rows =
                    m_read_statements->memory_copy_interval_track_v4()(qi.real_track_id)
                        .to_vector();
                break;
            case reader_types::track_type_t::memory:
                rows =
                    m_read_statements->memory_alloc_interval_track_v4()(qi.real_track_id)
                        .to_vector();
                break;
            case reader_types::track_type_t::kernel_dispatch_pmc:
                rows = m_read_statements
                           ->kd_pmc_interval_track()(qi.nid,
                                                     qi.pid,
                                                     qi.agent_id.value_or(0),
                                                     qi.pmc_id.value_or(0))
                           .to_vector();
                name_from_kernel_symbol = true;
                break;
            case reader_types::track_type_t::stream:
                rows = m_read_statements
                           ->stream_interval_track()(qi.stream_id.value(),
                                                     qi.stream_id.value(),
                                                     qi.stream_id.value())
                           .to_vector();
                is_stream = true;
                break;
            case reader_types::track_type_t::counter:
            default:
                // A counter track is scalar-only; an interval query returns nothing.
                return {};
        }
    }
    else
    {
        switch(qi.type)
        {
            case reader_types::track_type_t::cpu_thread:
                rows = (qi.region_is_sample
                            ? m_read_statements->region_interval_track_sample()
                            : m_read_statements->region_interval_track_main())(
                           qi.nid, qi.pid, qi.tid.value_or(0))
                           .to_vector();
                break;
            case reader_types::track_type_t::gpu_queue:
                rows = m_read_statements
                           ->kernel_dispatch_interval_track()(qi.nid,
                                                              qi.pid,
                                                              qi.agent_id.value_or(0),
                                                              qi.queue_id.value_or(0))
                           .to_vector();
                name_from_kernel_symbol = true;
                break;
            case reader_types::track_type_t::dma:
            {
                const bool has_q = qi.queue_id.has_value();
                const bool has_a = qi.agent_id.has_value();
                if(has_q && has_a)
                {
                    rows =
                        m_read_statements
                            ->memory_copy_interval_qa()(
                                qi.nid, qi.pid, qi.queue_id.value(), qi.agent_id.value())
                            .to_vector();
                }
                else if(has_q)
                {
                    rows = m_read_statements
                               ->memory_copy_interval_q_only()(
                                   qi.nid, qi.pid, qi.queue_id.value())
                               .to_vector();
                }
                else if(has_a)
                {
                    rows = m_read_statements
                               ->memory_copy_interval_a_only()(
                                   qi.nid, qi.pid, qi.agent_id.value())
                               .to_vector();
                }
                else
                {
                    rows =
                        m_read_statements->memory_copy_interval_neither()(qi.nid, qi.pid)
                            .to_vector();
                }
                break;
            }
            case reader_types::track_type_t::memory:
            {
                const bool has_q = qi.queue_id.has_value();
                const bool has_a = qi.agent_id.has_value();
                if(has_q && has_a)
                {
                    rows =
                        m_read_statements
                            ->memory_alloc_interval_qa()(
                                qi.nid, qi.pid, qi.agent_id.value(), qi.queue_id.value())
                            .to_vector();
                }
                else if(has_q)
                {
                    rows = m_read_statements
                               ->memory_alloc_interval_q_only()(
                                   qi.nid, qi.pid, qi.queue_id.value())
                               .to_vector();
                }
                else if(has_a)
                {
                    rows = m_read_statements
                               ->memory_alloc_interval_a_only()(
                                   qi.nid, qi.pid, qi.agent_id.value())
                               .to_vector();
                }
                else
                {
                    rows =
                        m_read_statements->memory_alloc_interval_neither()(qi.nid, qi.pid)
                            .to_vector();
                }
                break;
            }
            case reader_types::track_type_t::kernel_dispatch_pmc:
                rows = m_read_statements
                           ->kd_pmc_interval_track()(qi.nid,
                                                     qi.pid,
                                                     qi.agent_id.value_or(0),
                                                     qi.pmc_id.value_or(0))
                           .to_vector();
                name_from_kernel_symbol = true;
                break;
            case reader_types::track_type_t::stream:
                rows = m_read_statements
                           ->stream_interval_track()(qi.stream_id.value(),
                                                     qi.stream_id.value(),
                                                     qi.stream_id.value())
                           .to_vector();
                is_stream = true;
                break;
            case reader_types::track_type_t::counter:
            default:
                // A counter track is scalar-only; an interval query returns nothing.
                return {};
        }
    }

    reader_types::interval_entry_list_t events;
    events.reserve(rows.size());
    for(const auto& r : rows)
    {
        reader_types::interval_entry_t ev;
        ev.start = r.start;
        ev.end   = r.end;

        // Category is resolved to its display string in each backend's SQL (v3 via
        // rocpd_string, v4 via rocpd_info_category), so just carry it through.
        if(r.category.has_value()) ev.category = r.category.value();

        // Stream tracks are heterogeneous: per-row op_kind (kernel_dispatch=1,
        // memory_copy=2, memory_allocate=3) picks both the event type encoded in the
        // handle (routing to the right get_*_details()) and the name table for this row.
        // Homogeneous tracks take their type from the track.
        const reader_types::event_type_t etype =
            (is_stream && r.op_kind.has_value())
                ? static_cast<reader_types::event_type_t>(r.op_kind.value())
                : interval_event_type_for(qi.type);
        ev.id = reader_types::detail::event_id_access::make(etype, r.id);

        const bool use_kernel_symbol =
            is_stream ? (r.op_kind ==
                         static_cast<size_t>(reader_types::event_type_t::kernel_dispatch))
                      : name_from_kernel_symbol;

        if(r.name_ref.has_value())
        {
            if(use_kernel_symbol)
            {
                auto kit = m_kernel_symbol_info_utility.find(r.name_ref.value());
                if(kit != m_kernel_symbol_info_utility.end() && kit->second)
                {
                    ev.display_name = !kit->second->display_name.empty()
                                          ? kit->second->display_name
                                          : kit->second->name;
                }
            }
            else
            {
                auto sit = m_string_info_utility.find(r.name_ref.value());
                if(sit != m_string_info_utility.end()) ev.display_name = sit->second;
            }
        }

        events.push_back(std::move(ev));
    }

    // Rows arrive ORDER BY start ascending. Compute lane packing (all tracks) +
    // containment (stack tracks only) over the full track so lanes/levels are stable
    // regardless of any time-window filter applied afterwards. Cache peak concurrency on
    // the track so height consumers can read it via get_tracks() (Optiq
    // level->max_lane migration target).
    const auto max_lane = detail::compute_interval_layout(events, nesting_for(qi.type));
    auto       tit      = m_track_info_utility.find(track_id);
    if(tit != m_track_info_utility.end()) tit->second->max_lane = max_lane;

    // Time-window filter uses OVERLAP, not containment: a bar is kept iff its extent
    // [ev.start, ev.end] intersects the window [lo, hi] (boundary-inclusive), matching
    // the get_events_* SQL predicate and get_flows_in_window. The window is deliberately
    // NOT pushed into SQL: full-track lane/parent layout is computed above, over the
    // entire track, before this filter, so lanes stay stable across pans/zooms; pushing
    // the window into the query would relayout each windowed read and make lanes jump.
    if(filter.time_window.start.has_value() || filter.time_window.end.has_value())
    {
        const auto&                         lo = filter.time_window.start;
        const auto&                         hi = filter.time_window.end;
        reader_types::interval_entry_list_t filtered;
        filtered.reserve(events.size());
        for(auto& ev : events)
        {
            if(lo.has_value() && ev.end < lo.value()) continue;  // entirely before window
            if(hi.has_value() && ev.start > hi.value())
                continue;  // entirely after window
            filtered.push_back(std::move(ev));
        }
        events = std::move(filtered);
    }

    paginate(events, filter.pagination);
    return events;
}

reader_types::scalar_sample_list_t
reader_t::impl::get_scalar_track(size_t                              track_id,
                                 const reader_types::event_filter_t& filter)
{
    auto qit = m_track_query_info.find(track_id);
    if(qit == m_track_query_info.end()) return {};
    const auto& qi = qit->second;

    if(qi.type == reader_types::track_type_t::memory_activity)
    {
        // Compute per-agent cumulative running sum from all rocpd_memory_allocate rows
        // for (nid, pid), ordered by start. Mirrors Optiq's C++ synthesis (rocprof.cpp
        // CallbackCaptureMemoryActivity + CreateMemoryActivityTable). FREE agent_id and
        // size are recovered via address self-join to the most recent prior ALLOC at the
        // same address. REALLOC/RECLAIM are no-ops.
        auto raw_rows =
            m_read_statements->mem_activity_raw_track()(qi.nid, qi.pid).to_vector();

        std::unordered_map<size_t, std::pair<std::optional<size_t>, size_t>> addr_map;
        std::unordered_map<size_t, int64_t>                                  running;
        constexpr size_t null_agent_key = std::numeric_limits<size_t>::max();

        auto agent_key = [&](std::optional<size_t> a) -> size_t {
            return a.has_value() ? a.value() : null_agent_key;
        };
        const size_t target_key = agent_key(qi.agent_id);

        reader_types::scalar_sample_list_t events;

        for(const auto& r : raw_rows)
        {
            if(r.type == "ALLOC")
            {
                const size_t ak = agent_key(r.agent_id);
                running[ak] += static_cast<int64_t>(r.size);
                if(r.address.has_value())
                    addr_map[r.address.value()] = { r.agent_id, r.size };
                if(ak == target_key)
                {
                    if(filter.time_window.start.has_value() &&
                       r.start < filter.time_window.start.value())
                        continue;
                    if(filter.time_window.end.has_value() &&
                       r.start > filter.time_window.end.value())
                        continue;
                    reader_types::scalar_sample_t ev;
                    ev.id = reader_types::detail::event_id_access::make(
                        reader_types::event_type_t::memory_allocate, r.id);
                    ev.timestamp = r.start;
                    ev.value     = static_cast<double>(running[ak]);
                    events.push_back(ev);
                }
            }
            else if(r.type == "FREE")
            {
                std::optional<size_t> freed_agent = r.agent_id;
                size_t                freed_size  = r.size;
                if(r.address.has_value())
                {
                    auto it = addr_map.find(r.address.value());
                    if(it != addr_map.end())
                    {
                        if(!freed_agent.has_value()) freed_agent = it->second.first;
                        freed_size = it->second.second;
                        addr_map.erase(it);
                    }
                }
                const size_t ak = agent_key(freed_agent);
                running[ak] -= static_cast<int64_t>(freed_size);
                if(ak == target_key)
                {
                    if(filter.time_window.start.has_value() &&
                       r.start < filter.time_window.start.value())
                        continue;
                    if(filter.time_window.end.has_value() &&
                       r.start > filter.time_window.end.value())
                        continue;
                    reader_types::scalar_sample_t ev;
                    ev.id = reader_types::detail::event_id_access::make(
                        reader_types::event_type_t::memory_allocate, r.id);
                    ev.timestamp = r.start;
                    ev.value     = static_cast<double>(running[ak]);
                    events.push_back(ev);
                }
            }
        }

        paginate(events, filter.pagination);
        return events;
    }

    if(qi.type != reader_types::track_type_t::counter) return {};

    auto rows = m_read_statements->scalar_track()(qi.real_track_id).to_vector();

    reader_types::scalar_sample_list_t events;
    events.reserve(rows.size());
    for(const auto& r : rows)
    {
        if(filter.time_window.start.has_value() &&
           r.timestamp < filter.time_window.start.value())
            continue;
        if(filter.time_window.end.has_value() &&
           r.timestamp > filter.time_window.end.value())
            continue;

        reader_types::scalar_sample_t ev;
        ev.id = reader_types::detail::event_id_access::make(
            reader_types::event_type_t::sample, r.id);
        ev.timestamp = r.timestamp;
        ev.value     = r.value;
        events.push_back(ev);
    }

    paginate(events, filter.pagination);
    return events;
}

reader_types::track_stats_t
reader_t::impl::get_track_stats(size_t track_id)
{
    auto qit = m_track_query_info.find(track_id);
    if(qit == m_track_query_info.end()) return {};
    const auto& qi = qit->second;

    // memory_activity stats are computed from the same running-sum pass as
    // get_scalar_track, ensuring count/bounds exactly match the scalar series.
    if(qi.type == reader_types::track_type_t::memory_activity)
    {
        auto events = get_scalar_track(track_id, reader_types::event_filter_t{});
        if(events.empty()) return {};
        reader_types::track_stats_t stats;
        stats.min_ts = events.front().timestamp;
        stats.max_ts = events.back().timestamp;
        stats.count  = events.size();
        return stats;
    }

    // Cheap MIN/MAX/COUNT aggregates over exactly the rows get_interval_track /
    // get_scalar_track would return for this track. Routing mirrors those methods so
    // bounds/count agree with a full slice load without materializing event rows.
    std::vector<data_storage::track_stats_result> rows;

    if(m_is_v4)
    {
        switch(qi.type)
        {
            case reader_types::track_type_t::cpu_thread:
                rows = m_read_statements->region_stats_track_v4()(qi.real_track_id)
                           .to_vector();
                break;
            case reader_types::track_type_t::gpu_queue:
                rows =
                    m_read_statements->kernel_dispatch_stats_track_v4()(qi.real_track_id)
                        .to_vector();
                break;
            case reader_types::track_type_t::dma:
                rows = m_read_statements->memory_copy_stats_track_v4()(qi.real_track_id)
                           .to_vector();
                break;
            case reader_types::track_type_t::memory:
                rows = m_read_statements->memory_alloc_stats_track_v4()(qi.real_track_id)
                           .to_vector();
                break;
            case reader_types::track_type_t::kernel_dispatch_pmc:
                rows = m_read_statements
                           ->kd_pmc_stats_track()(qi.nid,
                                                  qi.pid,
                                                  qi.agent_id.value_or(0),
                                                  qi.pmc_id.value_or(0))
                           .to_vector();
                break;
            case reader_types::track_type_t::stream:
                rows = m_read_statements
                           ->stream_stats_track()(qi.stream_id.value(),
                                                  qi.stream_id.value(),
                                                  qi.stream_id.value())
                           .to_vector();
                break;
            case reader_types::track_type_t::counter:
                rows = m_read_statements->scalar_stats()(qi.real_track_id).to_vector();
                break;
            default: return {};
        }
    }
    else
    {
        switch(qi.type)
        {
            case reader_types::track_type_t::cpu_thread:
                rows =
                    (qi.region_is_sample ? m_read_statements->region_stats_track_sample()
                                         : m_read_statements->region_stats_track_main())(
                        qi.nid, qi.pid, qi.tid.value_or(0))
                        .to_vector();
                break;
            case reader_types::track_type_t::gpu_queue:
                rows = m_read_statements
                           ->kernel_dispatch_stats_track()(qi.nid,
                                                           qi.pid,
                                                           qi.agent_id.value_or(0),
                                                           qi.queue_id.value_or(0))
                           .to_vector();
                break;
            case reader_types::track_type_t::dma:
            {
                const bool has_q = qi.queue_id.has_value();
                const bool has_a = qi.agent_id.has_value();
                if(has_q && has_a)
                {
                    rows =
                        m_read_statements
                            ->memory_copy_stats_qa()(
                                qi.nid, qi.pid, qi.queue_id.value(), qi.agent_id.value())
                            .to_vector();
                }
                else if(has_q)
                {
                    rows = m_read_statements
                               ->memory_copy_stats_q_only()(
                                   qi.nid, qi.pid, qi.queue_id.value())
                               .to_vector();
                }
                else if(has_a)
                {
                    rows = m_read_statements
                               ->memory_copy_stats_a_only()(
                                   qi.nid, qi.pid, qi.agent_id.value())
                               .to_vector();
                }
                else
                {
                    rows = m_read_statements->memory_copy_stats_neither()(qi.nid, qi.pid)
                               .to_vector();
                }
                break;
            }
            case reader_types::track_type_t::memory:
            {
                const bool has_q = qi.queue_id.has_value();
                const bool has_a = qi.agent_id.has_value();
                if(has_q && has_a)
                {
                    rows =
                        m_read_statements
                            ->memory_alloc_stats_qa()(
                                qi.nid, qi.pid, qi.agent_id.value(), qi.queue_id.value())
                            .to_vector();
                }
                else if(has_q)
                {
                    rows = m_read_statements
                               ->memory_alloc_stats_q_only()(
                                   qi.nid, qi.pid, qi.queue_id.value())
                               .to_vector();
                }
                else if(has_a)
                {
                    rows = m_read_statements
                               ->memory_alloc_stats_a_only()(
                                   qi.nid, qi.pid, qi.agent_id.value())
                               .to_vector();
                }
                else
                {
                    rows = m_read_statements->memory_alloc_stats_neither()(qi.nid, qi.pid)
                               .to_vector();
                }
                break;
            }
            case reader_types::track_type_t::kernel_dispatch_pmc:
                rows = m_read_statements
                           ->kd_pmc_stats_track()(qi.nid,
                                                  qi.pid,
                                                  qi.agent_id.value_or(0),
                                                  qi.pmc_id.value_or(0))
                           .to_vector();
                break;
            case reader_types::track_type_t::stream:
                rows = m_read_statements
                           ->stream_stats_track()(qi.stream_id.value(),
                                                  qi.stream_id.value(),
                                                  qi.stream_id.value())
                           .to_vector();
                break;
            case reader_types::track_type_t::counter:
                rows = m_read_statements->scalar_stats()(qi.real_track_id).to_vector();
                break;
            default: return {};
        }
    }

    reader_types::track_stats_t stats;
    if(!rows.empty())
    {
        // Aggregate query yields exactly one row. min_ts/max_ts are nullopt when the
        // track has no events (SQL MIN/MAX over an empty set), matching count == 0.
        stats.min_ts = rows.front().min_ts;
        stats.max_ts = rows.front().max_ts;
        stats.count  = rows.front().count;
    }
    return stats;
}

reader_types::flow_list_t
reader_t::impl::get_flows(const reader_types::event_filter_t& filter)
{
    reader_types::flow_list_t flows;

    const bool has_window =
        filter.time_window.start.has_value() || filter.time_window.end.has_value();
    const size_t lo = filter.time_window.start.value_or(0);
    const size_t hi = filter.time_window.end.value_or(std::numeric_limits<size_t>::max());

    // Flow SOURCE. Edges are sourced from stack_id (+ parent_stack_id for lineage), NOT
    // the correlation_id column. correlation_id is populated only 0-0.86% of the time (an
    // external id pushed via roctx -- 8/935 events on the measured capture, all
    // group-size-1, 0 cross-track pairs), so flows sourced from it would yield a nearly
    // empty cross-track graph, whereas the stack_id clique yields 100% clean cross-track
    // pairs. This builder makes an UNDIRECTED stack_id clique with a heuristic direction,
    // grouped by flow_id = stack_id; it does NOT implement a stricter directed,
    // cross-track-ONLY model -- the region/sibling sets can still include same-track
    // edges.
    //
    // Direction is a heuristic; the stricter directed cross-track-only model remains
    // under consideration, so this behaviour is provisional.

    // De-duplicate each unordered endpoint pair to ONE directed edge. The same-type sets
    // (region->region, kd/mc/ma siblings) emit both (a,b) and (b,a); cross-type sets emit
    // one direction. Key is the endpoint pair normalized to (min,max) so both orderings
    // collapse. Insertion order into `flows` follows the run() call order below.
    std::set<std::pair<reader_types::event_id_t, reader_types::event_id_t>> seen;

    auto run = [&](const data_storage::read_statements_base::flow_statement_set& set,
                   reader_types::event_type_t source_type,
                   reader_types::event_type_t dest_type,
                   reader_types::flow_kind_t  kind) {
        auto rows =
            has_window ? set.time_filtered(lo, hi).to_vector() : set.base().to_vector();
        for(const auto& r : rows)
        {
            const auto a =
                reader_types::detail::event_id_access::make(source_type, r.source_id);
            const auto b =
                reader_types::detail::event_id_access::make(dest_type, r.dest_id);

            const auto key = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
            if(!seen.insert(key).second) continue;

            // Direction. Prefer parent_stack_id lineage — the endpoint whose stack_id
            // equals the other's parent_stack_id is the parent (src). The clique join
            // constrains both endpoints to the SAME stack_id, so this only fires on a
            // self-parent row and in practice yields to the start-ts fallback (earlier
            // start = src; ties broken by handle order for determinism).
            reader_types::event_id_t src{};
            reader_types::event_id_t dst{};
            if(r.dest_parent.has_value() && *r.dest_parent == r.stack_id)
            {
                src = a;  // source endpoint is the parent of the dest endpoint
                dst = b;
            }
            else if(r.source_parent.has_value() && *r.source_parent == r.stack_id)
            {
                src = b;  // dest endpoint is the parent of the source endpoint
                dst = a;
            }
            else if(r.source_start != r.dest_start)
            {
                const bool a_first = r.source_start < r.dest_start;
                src                = a_first ? a : b;
                dst                = a_first ? b : a;
            }
            else
            {
                src = key.first;  // equal starts: deterministic tie-break by handle order
                dst = key.second;
            }

            // flow_id groups a causal chain. All edges of one stack_id clique share
            // flow_id = that stack_id, so a multi-hop lineage is recoverable by grouping
            // on flow_id and sorting by src start.
            flows.push_back(reader_types::flow_edge_t{
                src, dst, reader_types::detail::flow_id_access::make(r.stack_id), kind });
        }
    };

    // kind is fixed by each set's endpoint-type pairing (order-independent, so
    // orientation never changes it).
    using et = reader_types::event_type_t;
    using fk = reader_types::flow_kind_t;
    run(m_read_statements->region_to_kernel_dispatch_flows(),
        et::region,
        et::kernel_dispatch,
        fk::launch_to_dispatch);
    run(m_read_statements->region_to_memory_copy_flows(),
        et::region,
        et::memory_copy,
        fk::copy_submit_to_exec);
    run(m_read_statements->region_to_memory_allocate_flows(),
        et::region,
        et::memory_allocate,
        fk::copy_submit_to_exec);
    run(m_read_statements->region_to_region_flows(), et::region, et::region, fk::generic);
    run(m_read_statements->kernel_dispatch_sibling_flows(),
        et::kernel_dispatch,
        et::kernel_dispatch,
        fk::stream_dependency);
    run(m_read_statements->memory_copy_sibling_flows(),
        et::memory_copy,
        et::memory_copy,
        fk::stream_dependency);
    run(m_read_statements->memory_allocate_sibling_flows(),
        et::memory_allocate,
        et::memory_allocate,
        fk::stream_dependency);

    return flows;
}

reader_types::flow_list_t
reader_t::impl::get_flows_for_event(const reader_types::event_id_t& id)
{
    // Adjacent edges: those with `id` as either endpoint. Cheap post-filter over the
    // full built edge list (the clique join is not indexable by a single endpoint).
    reader_types::flow_list_t out;
    for(auto& f : get_flows({}))
        if(f.source == id || f.dest == id) out.push_back(f);
    return out;
}

reader_types::flow_list_t
reader_t::impl::get_flows_for_chain(const reader_types::flow_id_t& flow_id)
{
    // Chain members: every edge sharing this flow_id. Cheap post-filter over the full
    // built edge list (flow_id is assigned during the clique build, not a query key).
    reader_types::flow_list_t out;
    for(auto& f : get_flows({}))
        if(f.flow_id == flow_id) out.push_back(f);
    return out;
}

reader_types::flow_list_t
reader_t::impl::get_flows_in_window(const std::vector<reader_types::track_id_t>& tracks,
                                    const reader_types::time_window_t&           window,
                                    uint32_t max_edges)
{
    auto edges = get_flows({});
    if(edges.empty()) return edges;

    // A flow_edge_t carries no timestamps and no track_id, so this selector cannot
    // post-filter get_flows({}) the way the adjacency/chain selectors do. Flow endpoints
    // are always interval events (region/kd/mc/ma), so resolve each endpoint's (start,
    // end) and its track membership by sweeping the interval tracks once. This goes
    // through get_interval_track, which already hides the v3-synthesized-vs-v4-column
    // track-id difference, so both backends yield identical selector semantics.
    struct endpoint_geom
    {
        reader_types::timestamp_t start{};
        reader_types::timestamp_t end{};
        bool                      seen{ false };
    };
    std::unordered_map<reader_types::event_id_t, endpoint_geom> geom;
    std::unordered_map<reader_types::event_id_t, std::vector<reader_types::track_id_t>>
        on_tracks;
    for(const auto& t : get_tracks())
    {
        if(!t) continue;
        switch(t->type)
        {
            case reader_types::track_type_t::counter:
            case reader_types::track_type_t::memory_activity: continue;  // scalar-only
            default: break;
        }
        for(const auto& ev : get_interval_track(t->id.value, {}))
        {
            auto& g = geom[ev.id];
            if(!g.seen)
            {
                g.start = ev.start;
                g.end   = ev.end;
                g.seen  = true;
            }
            on_tracks[ev.id].push_back(
                t->id);  // multi-track: kd is on gpu_queue + pmc + stream
        }
    }

    const std::set<reader_types::track_id_t> track_set(tracks.begin(), tracks.end());
    const bool has_window = window.start.has_value() || window.end.has_value();
    const auto wlo        = window.start.value_or(0);
    const auto whi =
        window.end.value_or(std::numeric_limits<reader_types::timestamp_t>::max());

    reader_types::flow_list_t filtered;
    for(const auto& e : edges)
    {
        const auto& gs = geom[e.source];
        const auto& gd = geom[e.dest];

        // Window overlap uses the edge's full temporal extent
        // [min(src.start,dst.start), max(src.end,dst.end)] (conservative superset of the
        // src.end->dst.start arrow), included iff it intersects the window; empty window
        // = no filter.
        if(has_window)
        {
            const auto elo = std::min(gs.start, gd.start);
            const auto ehi = std::max(gs.end, gd.end);
            if(elo > whi || ehi < wlo) continue;
        }

        // An edge is kept iff AT LEAST ONE endpoint sits on a listed track, so a
        // cross-track arrow with one endpoint off-screen still surfaces its visible half;
        // empty tracks = all.
        if(!track_set.empty())
        {
            bool touches = false;
            for(auto id : { e.source, e.dest })
            {
                for(auto tid : on_tracks[id])
                    if(track_set.count(tid))
                    {
                        touches = true;
                        break;
                    }
                if(touches) break;
            }
            if(!touches) continue;
        }

        filtered.push_back(e);
    }

    // Decimation keeps the highest arrow-span-latency (dst.start - src.end, clamped at 0)
    // edges, tie-broken by (source, dest) handle order so an identical window yields an
    // identical ranking (stable across pans); max_edges == 0 = uncapped. This ranking is
    // a reversible default, and the decimation contract itself remains provisional.
    if(max_edges == 0 || filtered.size() <= max_edges) return filtered;

    auto latency = [&](const reader_types::flow_edge_t& e) -> reader_types::timestamp_t {
        const auto& gs = geom[e.source];
        const auto& gd = geom[e.dest];
        return gd.start > gs.end ? gd.start - gs.end : 0;  // clamp: size_t is unsigned
    };
    std::sort(
        filtered.begin(),
        filtered.end(),
        [&](const reader_types::flow_edge_t& a, const reader_types::flow_edge_t& b) {
            const auto la = latency(a);
            const auto lb = latency(b);
            if(la != lb) return la > lb;
            if(!(a.source == b.source)) return a.source < b.source;
            return a.dest < b.dest;
        });
    filtered.resize(max_edges);
    return filtered;
}

// ============================================================================
// Detail methods by event handle
// ============================================================================

reader_types::pmc_event_data_t
reader_t::impl::build_pmc_event_data(const data_storage::scalar_detail_result& row)
{
    reader_types::pmc_event_data_t data;
    data.value            = row.value;
    data.sample.timestamp = row.timestamp;

    auto tit = m_track_info_utility.find(row.track_id);
    if(tit != m_track_info_utility.end()) data.sample.track = tit->second;

    // v3 has no event-by-event_id read statement, so data.event is left null here.
    return data;
}

std::optional<reader_types::pmc_event_data_t>
reader_t::impl::get_pmc_event_details(const reader_types::event_id_t& id)
{
    if(reader_types::detail::event_id_access::type(id) !=
       reader_types::event_type_t::pmc_event)
        return std::nullopt;
    auto rows =
        m_read_statements
            ->pmc_event_detail()(reader_types::detail::event_id_access::row_id(id))
            .to_vector();
    if(rows.empty()) return std::nullopt;
    return build_pmc_event_data(rows.front());
}

std::optional<reader_types::region_data_t>
reader_t::impl::get_region_details(const reader_types::event_id_t& id)
{
    if(reader_types::detail::event_id_access::type(id) !=
       reader_types::event_type_t::region)
        return std::nullopt;
    reader_types::timeline_event_t event{};
    event.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                                reader_types::event_type_t::region };
    return get_region_details(event);
}

std::optional<reader_types::kernel_dispatch_data_t>
reader_t::impl::get_kernel_dispatch_details(const reader_types::event_id_t& id)
{
    if(reader_types::detail::event_id_access::type(id) !=
       reader_types::event_type_t::kernel_dispatch)
        return std::nullopt;
    reader_types::timeline_event_t event{};
    event.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                                reader_types::event_type_t::kernel_dispatch };
    return get_kernel_dispatch_details(event);
}

std::optional<reader_types::memory_copy_data_t>
reader_t::impl::get_memory_copy_details(const reader_types::event_id_t& id)
{
    if(reader_types::detail::event_id_access::type(id) !=
       reader_types::event_type_t::memory_copy)
        return std::nullopt;
    reader_types::timeline_event_t event{};
    event.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                                reader_types::event_type_t::memory_copy };
    return get_memory_copy_details(event);
}

std::optional<reader_types::memory_alloc_data_t>
reader_t::impl::get_memory_alloc_details(const reader_types::event_id_t& id)
{
    if(reader_types::detail::event_id_access::type(id) !=
       reader_types::event_type_t::memory_allocate)
        return std::nullopt;
    reader_types::timeline_event_t event{};
    event.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                                reader_types::event_type_t::memory_allocate };
    return get_memory_alloc_details(event);
}

// ============================================================================
// Unified event detail
// ============================================================================

// Collapses the seven typed get_*_details public methods into this one path. The typed
// methods survive as private impl helpers, reused here for their SQL + FK resolution;
// only their public surface is gone. Optiq is the sole consumer, migrating to
// get_event_info.
std::optional<reader_types::event_info_t>
reader_t::impl::get_event_info(const reader_types::event_id_t& id)
{
    using reader_types::event_type_t;

    reader_types::event_info_t detail;
    detail.id = id;

    // Property-key naming = the source struct's field name, snake_case; region
    // call-arguments are keyed by the argument's own name. Values are typed (unsigned
    // counts/ids/addresses -> uint64_t, counter value -> double, args -> string).
    // Optional-field policy: an absent std::optional scalar or a null linked entity is
    // OMITTED from the bag (not emitted as monostate/nullptr_t), applied uniformly.
    auto push_u = [&](std::string key, size_t v) {
        detail.properties.push_back({ std::move(key), static_cast<uint64_t>(v) });
    };
    auto push_opt_u = [&](std::string key, const std::optional<size_t>& v) {
        if(v.has_value()) push_u(std::move(key), v.value());
    };
    // Fold rocpd_arg rows into the property bag as name-keyed values. Args key on the
    // shared rocpd_event row, so region / kernel_dispatch / memory_copy / memory_allocate
    // all carry them; point events (sample / pmc_event) do not.
    auto fold_args = [&]() {
        reader_types::timeline_event_t ev{};
        ev.unique_identifier = { reader_types::detail::event_id_access::row_id(id),
                                 reader_types::detail::event_id_access::type(id) };
        for(const auto& a : get_arguments(ev))
            detail.properties.push_back({ a->name, a->value });
    };

    switch(reader_types::detail::event_id_access::type(id))
    {
        case event_type_t::region:
        {
            auto d = get_region_details(id);
            if(!d) return std::nullopt;
            detail.name = d->name;
            if(d->event) detail.category = d->event->event_category;
            detail.ts = d->start_timestamp;
            detail.te = d->end_timestamp;

            // Freeform region call-arguments as string-valued properties, keyed by name.
            // (args are not populated by get_region_details; fetch them separately.)
            fold_args();
            break;
        }
        case event_type_t::kernel_dispatch:
        {
            auto d = get_kernel_dispatch_details(id);
            if(!d) return std::nullopt;
            detail.name = d->name;
            if(d->event) detail.category = d->event->event_category;
            detail.ts = d->start_timestamp;
            detail.te = d->end_timestamp;
            push_u("dispatch_id", d->dispatch_id);
            push_u("workgroup_size_x", d->workgroup_size_x);
            push_u("workgroup_size_y", d->workgroup_size_y);
            push_u("workgroup_size_z", d->workgroup_size_z);
            push_u("grid_size_x", d->grid_size_x);
            push_u("grid_size_y", d->grid_size_y);
            push_u("grid_size_z", d->grid_size_z);
            push_opt_u("private_segment_size", d->private_segment_size);
            push_opt_u("group_segment_size", d->group_segment_size);
            // Linked entities collapse to their integer id (kernel_symbol_id,
            // code_object_id, node_id, process_id, thread_id, ...), NOT the resolved
            // sub-struct -- a deliberate collapse loss; the consumer does a follow-up
            // lookup by id. Entities the detail row does not carry (kd agent/
            // stream/queue) are simply absent under the omit policy.
            if(d->kernel_symbol_info)
                push_u("kernel_symbol_id", d->kernel_symbol_info->id);
            if(d->code_object_info) push_u("code_object_id", d->code_object_info->id);
            if(d->node_info) push_u("node_id", d->node_info->node_id);
            if(d->process_info) push_u("process_id", d->process_info->pid);
            if(d->thread_info) push_u("thread_id", d->thread_info->thread_id);
            if(d->agent_info) push_u("agent_id", d->agent_info->id);
            if(d->stream_info) push_u("stream_id", d->stream_info->stream_id);
            if(d->queue_info) push_u("queue_id", d->queue_info->queue_id);
            fold_args();
            break;
        }
        case event_type_t::memory_copy:
        {
            auto d = get_memory_copy_details(id);
            if(!d) return std::nullopt;
            detail.name = d->name;
            if(d->event) detail.category = d->event->event_category;
            detail.ts = d->start_timestamp;
            detail.te = d->end_timestamp;
            push_u("size", d->size);
            push_opt_u("src_address", d->src_address);
            push_opt_u("dst_address", d->dst_address);
            if(!d->region_name.empty())
                detail.properties.push_back({ "region_name", d->region_name });
            if(d->src_agent_id) push_u("src_agent_id", d->src_agent_id->id);
            if(d->dst_agent_id) push_u("dst_agent_id", d->dst_agent_id->id);
            if(d->node_info) push_u("node_id", d->node_info->node_id);
            if(d->process_info) push_u("process_id", d->process_info->pid);
            if(d->thread_info) push_u("thread_id", d->thread_info->thread_id);
            if(d->stream_info) push_u("stream_id", d->stream_info->stream_id);
            if(d->queue_info) push_u("queue_id", d->queue_info->queue_id);
            fold_args();
            break;
        }
        case event_type_t::memory_allocate:
        {
            auto d = get_memory_alloc_details(id);
            if(!d) return std::nullopt;
            // memory_allocate has no name field -> empty header name.
            if(d->event) detail.category = d->event->event_category;
            detail.ts = d->start_timestamp;
            detail.te = d->end_timestamp;
            if(!d->type.empty()) detail.properties.push_back({ "type", d->type });
            if(!d->level.empty()) detail.properties.push_back({ "level", d->level });
            push_opt_u("address", d->address);
            push_u("size", d->size);
            if(d->node_info) push_u("node_id", d->node_info->node_id);
            if(d->process_info) push_u("process_id", d->process_info->pid);
            if(d->thread_info) push_u("thread_id", d->thread_info->thread_id);
            if(d->agent_info) push_u("agent_id", d->agent_info->id);
            if(d->stream_info) push_u("stream_id", d->stream_info->stream_id);
            if(d->queue_info) push_u("queue_id", d->queue_info->queue_id);
            fold_args();
            break;
        }
        case event_type_t::sample:
        {
            // A sample-typed id is unambiguously a counter sample -- get_scalar_track
            // (counter arm) is the sole event_type_t::sample mint site. Its detail is
            // the counter name (from the track) + value (from scalar_detail()); it is a
            // point event so te stays nullopt.
            auto rows =
                m_read_statements
                    ->scalar_detail()(reader_types::detail::event_id_access::row_id(id))
                    .to_vector();
            if(rows.empty()) return std::nullopt;
            auto d    = build_pmc_event_data(rows.front());
            detail.ts = d.sample.timestamp;  // point event: te stays nullopt
            if(d.sample.track) detail.name = d.sample.track->name;
            detail.properties.push_back({ "value", d.value });
            break;
        }
        case event_type_t::pmc_event:
        {
            auto d = get_pmc_event_details(id);
            if(!d) return std::nullopt;
            if(d->event) detail.category = d->event->event_category;
            detail.ts = d->sample.timestamp;  // point event: te stays nullopt
            detail.properties.push_back({ "value", d->value });
            break;
        }
        default: return std::nullopt;
    }

    return detail;
}

}  // namespace profiler_hub
