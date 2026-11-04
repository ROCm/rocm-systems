// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocpd_v2_reader.hpp"

#include "catalog_aggregate_store.hpp"

#include <fmt/format.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace profiler_hub
{

using reader_types::track_kind_t;
using rocpd_v2::k_node_id;

namespace
{
struct string_row_t
{
    size_t                     id{};
    std::optional<std::string> text;
};

struct id_row_t
{
    size_t id{};
};

struct pid_tid_row_t
{
    size_t pid{};
    size_t tid{};
};

struct gpu_queue_row_t
{
    size_t gpu_id{};
    size_t queue_id{};
};

struct thread_track_row_t
{
    size_t                pid{};
    size_t                tid{};
    size_t                count{};
    std::optional<size_t> min_start;
    std::optional<size_t> max_end;
};

struct op_track_row_t
{
    size_t                gpu_id{};
    size_t                queue_id{};
    size_t                is_copy{};
    size_t                count{};
    std::optional<size_t> min_start;
    std::optional<size_t> max_end;
};

struct monitor_track_row_t
{
    size_t                device_id{};
    std::string           monitor_type;
    size_t                count{};
    std::optional<size_t> min_start;
    std::optional<size_t> max_end;
    std::optional<double> min_value;
    std::optional<double> max_value;
};

struct monitor_key_row_t
{
    size_t      device_id{};
    std::string monitor_type;
};

struct owner_row_t
{
    size_t pid{};
    size_t gpu_id{};
    size_t queue_id{};
};

struct gpu_queue_key_t
{
    size_t gpu{};
    size_t queue{};

    bool operator==(const gpu_queue_key_t&) const = default;
};

struct gpu_queue_key_hash_t
{
    size_t operator()(const gpu_queue_key_t& key) const
    {
        size_t hash = std::hash<size_t>{}(key.gpu);
        hash ^= std::hash<size_t>{}(key.queue) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

// One process owns a (gpu, queue). A queue id is reused by every GPU, so the
// process cannot be recovered from the queue id alone.
struct queue_ownership_t
{
    std::unordered_map<gpu_queue_key_t, size_t, gpu_queue_key_hash_t> by_queue;
    std::unordered_map<size_t, size_t>                                by_gpu;
};

queue_ownership_t
queue_ownership(data_storage::sqlite_backend& backend)
{
    queue_ownership_t ownership;
    backend
        .create_read_statement_executor<owner_row_t>(
            "SELECT MIN(A.pid), O.gpuId, O.queueId FROM rocpd_api_ops AO "
            "INNER JOIN rocpd_api A ON A.id = AO.api_id "
            "INNER JOIN rocpd_op O ON O.id = AO.op_id "
            "GROUP BY O.gpuId, O.queueId",
            &owner_row_t::pid,
            &owner_row_t::gpu_id,
            &owner_row_t::queue_id)()
        .for_each([&](const owner_row_t& row) {
            ownership.by_queue.emplace(gpu_queue_key_t{ row.gpu_id, row.queue_id }, row.pid);
            ownership.by_gpu.emplace(row.gpu_id, row.pid);
        });
    return ownership;
}

reader_types::process_info_ptr_t
process_of(const reader_catalog_t& catalog, size_t pid)
{
    const auto found = catalog.processes_by_id.find(pid);
    return found != catalog.processes_by_id.end() ? found->second : nullptr;
}

// pmc_infos and pmc_agent tracks are numbered in this order, so a track's
// pmc_id is the id of its pmc_info.
constexpr std::string_view k_monitor_groups =
    "FROM rocpd_monitor WHERE deviceId > 0 GROUP BY deviceId, monitorType ORDER BY "
    "deviceId, monitorType";
}  // namespace

void
rocpd_v2_reader::build_catalog_category(catalog_category_t category,
                                        reader_catalog_t&  catalog)
{
    switch(category)
    {
        case catalog_category_t::string_list:
            build_string_list(catalog);
            break;
        case catalog_category_t::nodes:
            build_nodes(catalog);
            break;
        case catalog_category_t::processes:
            if(load_catalog_aggregates(catalog)) return;
            build_processes(catalog);
            break;
        case catalog_category_t::threads:
            if(catalog.derived_aggregates_ready) return;
            build_threads(catalog);
            break;
        case catalog_category_t::agents:
            if(catalog.derived_aggregates_ready) return;
            build_agents(catalog);
            break;
        case catalog_category_t::tracks:
            if(catalog.derived_aggregates_ready) return;
            build_tracks(catalog);
            break;
        case catalog_category_t::queues:
            if(catalog.derived_aggregates_ready) return;
            build_queues(catalog);
            break;
        case catalog_category_t::pmc_infos:
            if(catalog.derived_aggregates_ready) return;
            build_pmc_infos(catalog);
            break;
        // The schema records no streams, code objects, or kernel symbols.
        case catalog_category_t::stream_hardware_links:
        case catalog_category_t::code_objects:
        case catalog_category_t::kernel_symbols:
        case catalog_category_t::streams:
            break;
    }
}

void
rocpd_v2_reader::build_string_list(reader_catalog_t& catalog)
{
    std::unordered_map<std::string_view, size_t> first_id_by_text;
    m_backend
        ->create_read_statement_executor<string_row_t>(
            "SELECT id, string FROM rocpd_string ORDER BY id",
            &string_row_t::id,
            &string_row_t::text)()
        .for_each([&](const string_row_t& row) {
            catalog.strings_by_id.emplace(row.id, row.text.value_or(""));
        });

    // Keys point into strings_by_id, which no longer changes.
    first_id_by_text.reserve(catalog.strings_by_id.size());
    for(const auto& [id, text] : catalog.strings_by_id)
    {
        const auto [found, inserted] = first_id_by_text.emplace(text, id);
        if(!inserted && id < found->second) found->second = id;
    }
    for(const auto& [id, text] : catalog.strings_by_id)
    {
        const size_t first = first_id_by_text.at(text);
        if(first != id) catalog.canonical_string_ids.emplace(id, first);
    }
}

void
rocpd_v2_reader::build_nodes(reader_catalog_t& catalog)
{
    auto node       = std::make_shared<reader_types::node_info_t>();
    node->node_id   = k_node_id;
    node->hash      = std::hash<std::string>{}(database_path());
    node->hostname  = std::filesystem::path(database_path()).stem().string();
    node->system_name = node->hostname;
    catalog.nodes.push_back(node);
    catalog.nodes_by_id.emplace(node->node_id, node);
}

void
rocpd_v2_reader::build_processes(reader_catalog_t& catalog)
{
    const auto node = catalog.nodes_by_id.find(k_node_id);
    m_backend
        ->create_read_statement_executor<id_row_t>(
            "SELECT pid FROM rocpd_api GROUP BY pid ORDER BY pid", &id_row_t::id)()
        .for_each([&](const id_row_t& row) {
            auto process = std::make_shared<reader_types::process_info_t>();
            process->id  = row.id;
            process->pid = row.id;
            if(node != catalog.nodes_by_id.end()) process->node_info = node->second;
            catalog.processes.push_back(process);
            catalog.processes_by_id.emplace(process->id, process);
        });
}

void
rocpd_v2_reader::build_threads(reader_catalog_t& catalog)
{
    const auto node    = catalog.nodes_by_id.find(k_node_id);
    size_t     next_id = 1;
    m_backend
        ->create_read_statement_executor<pid_tid_row_t>(
            "SELECT pid, tid FROM rocpd_api GROUP BY pid, tid ORDER BY pid, tid",
            &pid_tid_row_t::pid,
            &pid_tid_row_t::tid)()
        .for_each([&](const pid_tid_row_t& row) {
            auto thread               = std::make_shared<reader_types::thread_info_t>();
            thread->id                = next_id++;
            thread->parent_process_id = row.pid;
            thread->thread_id         = row.tid;
            if(node != catalog.nodes_by_id.end()) thread->node_info = node->second;
            if(const auto process = catalog.processes_by_id.find(row.pid);
               process != catalog.processes_by_id.end())
            {
                thread->process_info = process->second;
            }
            catalog.threads.push_back(thread);
            catalog.threads_by_id.emplace(thread->id, thread);
        });
}

void
rocpd_v2_reader::build_agents(reader_catalog_t& catalog)
{
    const auto node = catalog.nodes_by_id.find(k_node_id);
    m_backend
        ->create_read_statement_executor<id_row_t>(
            "SELECT gpuId FROM rocpd_op GROUP BY gpuId UNION SELECT deviceId FROM "
            "rocpd_monitor WHERE deviceId > 0 GROUP BY deviceId ORDER BY 1",
            &id_row_t::id)()
        .for_each([&](const id_row_t& row) {
            auto agent        = std::make_shared<reader_types::agent_info_t>();
            agent->id         = row.id;
            agent->agent_type = "GPU";
            agent->type_index = row.id;
            agent->name       = fmt::format("GPU {}", row.id);
            if(node != catalog.nodes_by_id.end()) agent->node_info = node->second;
            catalog.agents.push_back(agent);
            catalog.agents_by_id.emplace(agent->id, agent);
        });
}

void
rocpd_v2_reader::build_queues(reader_catalog_t& catalog)
{
    const auto node     = catalog.nodes_by_id.find(k_node_id);
    const auto owners   = queue_ownership(*m_backend);
    m_backend
        ->create_read_statement_executor<gpu_queue_row_t>(
            "SELECT gpuId, queueId FROM rocpd_op GROUP BY gpuId, queueId ORDER BY gpuId, "
            "queueId",
            &gpu_queue_row_t::gpu_id,
            &gpu_queue_row_t::queue_id)()
        .for_each([&](const gpu_queue_row_t& row) {
            auto queue      = std::make_shared<reader_types::queue_info_t>();
            queue->queue_id = row.queue_id;
            queue->nid      = k_node_id;
            queue->name     = fmt::format("GPU {} Queue {}", row.gpu_id, row.queue_id);
            if(const auto owner = owners.by_queue.find(gpu_queue_key_t{ row.gpu_id, row.queue_id });
               owner != owners.by_queue.end())
            {
                queue->pid          = owner->second;
                queue->process_info = process_of(catalog, owner->second);
            }
            if(node != catalog.nodes_by_id.end()) queue->node_info = node->second;
            catalog.queues.push_back(queue);
        });
}

void
rocpd_v2_reader::build_pmc_infos(reader_catalog_t& catalog)
{
    const auto node    = catalog.nodes_by_id.find(k_node_id);
    size_t     next_id = 1;
    m_backend
        ->create_read_statement_executor<monitor_key_row_t>(
            fmt::format("SELECT deviceId, monitorType {}", k_monitor_groups),
            &monitor_key_row_t::device_id,
            &monitor_key_row_t::monitor_type)()
        .for_each([&](const monitor_key_row_t& row) {
            auto pmc  = std::make_shared<reader_types::pmc_info_t>();
            pmc->id   = next_id++;
            pmc->name = row.monitor_type;
            pmc->symbol = row.monitor_type;
            if(const auto agent = catalog.agents_by_id.find(row.device_id);
               agent != catalog.agents_by_id.end())
            {
                pmc->agent_info = agent->second;
            }
            if(node != catalog.nodes_by_id.end()) pmc->node_info = node->second;
            catalog.pmc_infos.push_back(pmc);
        });
}

void
rocpd_v2_reader::build_tracks(reader_catalog_t& catalog)
{
    const auto node    = catalog.nodes_by_id.find(k_node_id);
    const auto owners  = queue_ownership(*m_backend);
    size_t     next_id = 0;

    const auto new_track = [&](std::string name, track_kind_t kind, size_t count,
                               const std::optional<size_t>& min_start,
                               const std::optional<size_t>& max_end) {
        auto track         = std::make_shared<reader_types::track_info_t>();
        track->id          = next_id++;
        track->name        = std::move(name);
        track->category    = kind;
        track->event_count = count;
        track->start_ts    = min_start.value_or(0);
        track->end_ts      = max_end.value_or(0);
        if(node != catalog.nodes_by_id.end()) track->node_info = node->second;
        catalog.tracks.push_back(track);
        return track;
    };

    std::unordered_map<size_t, std::unordered_map<size_t, reader_types::thread_info_ptr_t>>
        threads_by_pid;
    for(const auto& thread : catalog.threads)
    {
        threads_by_pid[thread->parent_process_id.value_or(0)][thread->thread_id] = thread;
    }

    m_backend
        ->create_read_statement_executor<thread_track_row_t>(
            "SELECT pid, tid, COUNT(*), MIN(start), MAX(end) FROM rocpd_api GROUP BY pid, "
            "tid ORDER BY pid, tid",
            &thread_track_row_t::pid,
            &thread_track_row_t::tid,
            &thread_track_row_t::count,
            &thread_track_row_t::min_start,
            &thread_track_row_t::max_end)()
        .for_each([&](const thread_track_row_t& row) {
            auto track = new_track(fmt::format("Thread {}", row.tid),
                                   track_kind_t::thread,
                                   row.count,
                                   row.min_start,
                                   row.max_end);
            if(const auto process = catalog.processes_by_id.find(row.pid);
               process != catalog.processes_by_id.end())
            {
                track->process_info = process->second;
            }
            if(const auto process = threads_by_pid.find(row.pid);
               process != threads_by_pid.end())
            {
                if(const auto thread = process->second.find(row.tid);
                   thread != process->second.end())
                {
                    track->thread_info = thread->second;
                }
            }
            const topology_key_t topology{ k_node_id, row.pid, row.tid };
            catalog.topology_by_track.emplace(track, topology);
            catalog.track_by_topology.emplace(topology, track);
        });

    size_t next_pmc_id = 1;
    m_backend
        ->create_read_statement_executor<monitor_track_row_t>(
            fmt::format("SELECT deviceId, monitorType, COUNT(*), MIN(start), MAX(end), "
                        "MIN(CAST(value AS REAL)), MAX(CAST(value AS REAL)) {}",
                        k_monitor_groups),
            &monitor_track_row_t::device_id,
            &monitor_track_row_t::monitor_type,
            &monitor_track_row_t::count,
            &monitor_track_row_t::min_start,
            &monitor_track_row_t::max_end,
            &monitor_track_row_t::min_value,
            &monitor_track_row_t::max_value)()
        .for_each([&](const monitor_track_row_t& row) {
            auto track = new_track(
                row.monitor_type, track_kind_t::pmc_agent, row.count, row.min_start, row.max_end);
            track->agent_id = row.device_id;
            track->pmc_id   = next_pmc_id++;
            if(const auto owner = owners.by_gpu.find(row.device_id);
               owner != owners.by_gpu.end())
            {
                track->process_info = process_of(catalog, owner->second);
            }
            if(row.min_value.has_value() && row.max_value.has_value())
            {
                track->value_range =
                    reader_types::value_range_t{ row.min_value.value(), row.max_value.value() };
            }
        });

    m_backend
        ->create_read_statement_executor<op_track_row_t>(
            fmt::format("SELECT gpuId, queueId, opType_id IN {} AS is_copy, COUNT(*), "
                        "MIN(start), MAX(end) FROM rocpd_op GROUP BY is_copy, gpuId, queueId "
                        "ORDER BY is_copy, gpuId, queueId",
                        rocpd_v2::k_copy_op_types),
            &op_track_row_t::gpu_id,
            &op_track_row_t::queue_id,
            &op_track_row_t::is_copy,
            &op_track_row_t::count,
            &op_track_row_t::min_start,
            &op_track_row_t::max_end)()
        .for_each([&](const op_track_row_t& row) {
            const bool is_copy = row.is_copy != 0;
            const auto kind    = is_copy ? track_kind_t::memory_copy_agent_queue
                                         : track_kind_t::kernel_dispatch_agent_queue;
            auto track = new_track(fmt::format("{} [{}] Queue {}",
                                               is_copy ? "Memory Copy" : "Kernel Dispatch",
                                               row.gpu_id,
                                               row.queue_id),
                                   kind,
                                   row.count,
                                   row.min_start,
                                   row.max_end);
            track->agent_id = row.gpu_id;
            track->queue_id = row.queue_id;
            if(const auto owner = owners.by_queue.find(gpu_queue_key_t{ row.gpu_id, row.queue_id });
               owner != owners.by_queue.end())
            {
                track->process_info = process_of(catalog, owner->second);
            }
            catalog.agent_queue_track_by_key.emplace(
                agent_queue_track_key_t{ kind, k_node_id, row.gpu_id, row.queue_id }, track);
        });
}

}  // namespace profiler_hub
