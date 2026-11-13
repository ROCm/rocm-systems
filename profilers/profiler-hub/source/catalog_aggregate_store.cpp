// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "catalog_aggregate_store.hpp"

#include "cache_directory.hpp"
#include "debug.hpp"
#if PROFILER_HUB_ENABLE_ROCPD_V2
#    include "formats/rocpd_v2/rocpd_v2_reader.hpp"
#endif
#include "reader_catalog.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace profiler_hub
{
namespace
{
#if PROFILER_HUB_ENABLE_ROCPD_V2
constexpr uint32_t k_format_version = 1;
constexpr char     k_magic[4]       = { 'P', 'H', 'C', '1' };
#endif
#if PROFILER_HUB_ENABLE_ROCPD_V3
constexpr uint32_t k_derived_format_version = 1;
constexpr char     k_derived_magic[4]       = { 'P', 'H', 'D', '1' };
#endif

struct source_identity_t
{
    std::string path;
    uint64_t    size{};
    int64_t     mtime{};
};

#if PROFILER_HUB_ENABLE_ROCPD_V2
struct stored_track_t
{
    uint64_t    id{};
    uint32_t    kind{};
    std::string name;
    uint64_t    event_count{};
    uint64_t    start_ts{};
    uint64_t    end_ts{};
    uint64_t    agent_id{};
    uint64_t    queue_id{};
    uint64_t    pmc_id{};
    bool        has_process{};
    uint64_t    pid{};
    bool        has_topology{};
    uint64_t    topo_pid{};
    uint64_t    topo_tid{};
    bool        has_thread{};
    uint64_t    thread_row_id{};
    bool        has_value{};
    double      value_min{};
    double      value_max{};
};
#endif

bool
same_magic(const char* found, const char* expected)
{
    for(size_t index = 0; index < 4; ++index)
    {
        if(found[index] != expected[index]) return false;
    }
    return true;
}

#if PROFILER_HUB_ENABLE_ROCPD_V3
bool
agent_queue_kind(reader_types::track_kind_t kind)
{
    switch(kind)
    {
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
        case reader_types::track_kind_t::memory_allocate_agent_queue:
        case reader_types::track_kind_t::memory_copy_agent_queue:
        case reader_types::track_kind_t::hip_event_agent_queue: return true;
        case reader_types::track_kind_t::thread:
        case reader_types::track_kind_t::thread_sample:
        case reader_types::track_kind_t::pmc_agent:
        case reader_types::track_kind_t::pmc_kernel:
        case reader_types::track_kind_t::pmc_memory:
        case reader_types::track_kind_t::stream: return false;
    }
    return false;
}

bool
known_derived_kind(uint32_t kind, reader_types::track_kind_t& out)
{
    const auto value = static_cast<reader_types::track_kind_t>(kind);
    switch(value)
    {
        case reader_types::track_kind_t::thread:
        case reader_types::track_kind_t::thread_sample:
        case reader_types::track_kind_t::pmc_agent:
        case reader_types::track_kind_t::pmc_kernel:
        case reader_types::track_kind_t::pmc_memory:
        case reader_types::track_kind_t::kernel_dispatch_agent_queue:
        case reader_types::track_kind_t::memory_allocate_agent_queue:
        case reader_types::track_kind_t::memory_copy_agent_queue:
        case reader_types::track_kind_t::stream:
        case reader_types::track_kind_t::hip_event_agent_queue: out = value; return true;
    }
    return false;
}

uint32_t
fit_count(size_t size)
{
    if(size > std::numeric_limits<uint32_t>::max())
    {
        throw std::runtime_error{ "catalog derived file is too large" };
    }
    return static_cast<uint32_t>(size);
}
#endif

bool
stat_source(const std::string& path, source_identity_t& identity)
{
    const std::filesystem::path file{ path };
    if(!std::filesystem::is_regular_file(file)) return false;
    identity.path  = path;
    identity.size  = static_cast<uint64_t>(std::filesystem::file_size(file));
    identity.mtime = static_cast<int64_t>(
        std::filesystem::last_write_time(file).time_since_epoch().count());
    return true;
}

std::string
catalog_directory(const reader_catalog_t& catalog)
{
    const std::string cache_directory = trace_cache_directory(
        catalog.aggregate_cache_root, catalog.aggregate_cache_paths);
    if(cache_directory.empty()) return {};
    return (std::filesystem::path{ cache_directory } / "catalog").string();
}

#if PROFILER_HUB_ENABLE_ROCPD_V2
std::filesystem::path
aggregate_file(const reader_catalog_t& catalog)
{
    const std::string directory = catalog_directory(catalog);
    if(directory.empty()) return {};
    return std::filesystem::path{ directory } /
           ("aggregates-" + std::to_string(catalog.aggregate_cache_source) + ".bin");
}
#endif

#if PROFILER_HUB_ENABLE_ROCPD_V3
std::filesystem::path
derived_file(const reader_catalog_t& catalog)
{
    const std::string directory = catalog_directory(catalog);
    if(directory.empty()) return {};
    return std::filesystem::path{ directory } /
           ("derived-" + std::to_string(catalog.aggregate_cache_source) + ".bin");
}
#endif

class binary_writer
{
public:
    explicit binary_writer(const std::filesystem::path& path)
    : m_out{ path, std::ios::binary | std::ios::trunc }
    {
        if(!m_out) throw std::runtime_error{ "catalog aggregate file was not created" };
    }

    void bytes(const void* data, size_t size)
    {
        m_out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if(!m_out) throw std::runtime_error{ "catalog aggregate file was not written" };
    }

    void u8(uint8_t value) { bytes(&value, sizeof(value)); }
    void u32(uint32_t value) { bytes(&value, sizeof(value)); }
    void u64(uint64_t value) { bytes(&value, sizeof(value)); }
    void f64(double value) { bytes(&value, sizeof(value)); }

    void text(std::string_view value)
    {
        if(value.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error{ "catalog aggregate string is too long" };
        }
        u32(static_cast<uint32_t>(value.size()));
        if(!value.empty()) bytes(value.data(), value.size());
    }

private:
    std::ofstream m_out;
};

class binary_reader
{
public:
    explicit binary_reader(const std::filesystem::path& path)
    : m_in{ path, std::ios::binary }
    {
        if(!m_in) throw std::runtime_error{ "catalog aggregate file was not opened" };
    }

    void bytes(void* data, size_t size)
    {
        m_in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
        if(!m_in) throw std::runtime_error{ "catalog aggregate file is truncated" };
    }

    uint8_t u8()
    {
        uint8_t value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    uint32_t u32()
    {
        uint32_t value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    uint64_t u64()
    {
        uint64_t value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    double f64()
    {
        double value = 0;
        bytes(&value, sizeof(value));
        return value;
    }

    std::string text()
    {
        const uint32_t size = u32();
        if(size > (1U << 20U))
        {
            throw std::runtime_error{ "catalog aggregate string is too long" };
        }
        std::string value(size, '\0');
        if(size != 0) bytes(value.data(), size);
        return value;
    }

private:
    std::ifstream m_in;
};

#if PROFILER_HUB_ENABLE_ROCPD_V2
bool
known_kind(uint32_t kind, reader_types::track_kind_t& out)
{
    switch(kind)
    {
        case static_cast<uint32_t>(reader_types::track_kind_t::thread):
        case static_cast<uint32_t>(reader_types::track_kind_t::pmc_agent):
        case static_cast<uint32_t>(
            reader_types::track_kind_t::kernel_dispatch_agent_queue):
        case static_cast<uint32_t>(reader_types::track_kind_t::memory_copy_agent_queue):
            out = static_cast<reader_types::track_kind_t>(kind);
            return true;
        default: return false;
    }
}

uint64_t
queue_gpu_id(const reader_types::queue_info_t& queue)
{
    constexpr std::string_view k_prefix = "GPU ";
    constexpr std::string_view k_mark   = " Queue ";
    if(queue.name.size() <= k_prefix.size() ||
       queue.name.compare(0, k_prefix.size(), k_prefix) != 0)
    {
        throw std::runtime_error{ "catalog aggregate queue name has no gpu id" };
    }
    const size_t mark = queue.name.find(k_mark);
    if(mark == std::string::npos || mark <= k_prefix.size())
    {
        throw std::runtime_error{ "catalog aggregate queue name has no gpu id" };
    }
    return static_cast<uint64_t>(
        std::stoull(queue.name.substr(k_prefix.size(), mark - k_prefix.size())));
}
#endif

void
rename_into_place(const std::filesystem::path& temporary,
                  const std::filesystem::path& final_path)
{
    std::error_code error;
    std::filesystem::rename(temporary, final_path, error);
    if(error)
    {
        std::filesystem::remove(final_path, error);
        std::filesystem::rename(temporary, final_path);
    }
}

}  // namespace

#if PROFILER_HUB_ENABLE_ROCPD_V2
bool
load_catalog_aggregates(reader_catalog_t& catalog)
{
    static_assert(std::endian::native == std::endian::little);
    if(catalog.derived_aggregates_ready) return true;

    try
    {
        const std::filesystem::path path = aggregate_file(catalog);
        if(path.empty() || !std::filesystem::is_regular_file(path)) return false;
        if(catalog.aggregate_cache_source >= catalog.aggregate_cache_paths.size())
            return false;

        source_identity_t live;
        if(!stat_source(catalog.aggregate_cache_paths[catalog.aggregate_cache_source],
                        live))
        {
            return false;
        }
        const auto node = catalog.nodes_by_id.find(rocpd_v2::k_node_id);
        if(node == catalog.nodes_by_id.end()) return false;

        binary_reader in{ path };
        char          magic[4] = {};
        in.bytes(magic, 4);
        if(!same_magic(magic, k_magic) || in.u32() != k_format_version) return false;
        if(in.u32() != catalog.aggregate_cache_source) return false;
        if(in.u64() != live.size || in.u64() != static_cast<uint64_t>(live.mtime))
            return false;
        if(in.text() != live.path) return false;

        const uint32_t                                process_count = in.u32();
        std::vector<reader_types::process_info_ptr_t> processes;
        processes.reserve(process_count);
        for(uint32_t index = 0; index < process_count; ++index)
        {
            auto process       = std::make_shared<reader_types::process_info_t>();
            process->pid       = static_cast<size_t>(in.u64());
            process->id        = process->pid;
            process->node_info = node->second;
            processes.push_back(std::move(process));
        }

        const uint32_t                               thread_count = in.u32();
        std::vector<reader_types::thread_info_ptr_t> threads;
        threads.reserve(thread_count);
        std::unordered_map<size_t, reader_types::process_info_ptr_t> process_by_pid;
        for(const auto& process : processes)
        {
            process_by_pid.emplace(process->pid, process);
        }
        for(uint32_t index = 0; index < thread_count; ++index)
        {
            auto thread               = std::make_shared<reader_types::thread_info_t>();
            thread->id                = static_cast<size_t>(in.u64());
            thread->parent_process_id = static_cast<size_t>(in.u64());
            thread->thread_id         = static_cast<size_t>(in.u64());
            thread->node_info         = node->second;
            if(const auto process =
                   process_by_pid.find(thread->parent_process_id.value_or(0));
               process != process_by_pid.end())
            {
                thread->process_info = process->second;
            }
            threads.push_back(std::move(thread));
        }

        const uint32_t                              agent_count = in.u32();
        std::vector<reader_types::agent_info_ptr_t> agents;
        agents.reserve(agent_count);
        for(uint32_t index = 0; index < agent_count; ++index)
        {
            auto agent        = std::make_shared<reader_types::agent_info_t>();
            agent->id         = static_cast<size_t>(in.u64());
            agent->type_index = agent->id;
            agent->agent_type = "GPU";
            agent->name       = "GPU " + std::to_string(agent->id);
            agent->node_info  = node->second;
            agents.push_back(std::move(agent));
        }

        const uint32_t                              queue_count = in.u32();
        std::vector<reader_types::queue_info_ptr_t> queues;
        queues.reserve(queue_count);
        for(uint32_t index = 0; index < queue_count; ++index)
        {
            const uint64_t gpu_id    = in.u64();
            const uint64_t queue_id  = in.u64();
            const bool     has_owner = in.u8() != 0;
            const uint64_t owner     = in.u64();
            auto           queue     = std::make_shared<reader_types::queue_info_t>();
            queue->queue_id          = static_cast<size_t>(queue_id);
            queue->nid               = rocpd_v2::k_node_id;
            queue->name =
                "GPU " + std::to_string(gpu_id) + " Queue " + std::to_string(queue_id);
            queue->node_info = node->second;
            if(has_owner)
            {
                queue->pid = static_cast<size_t>(owner);
                if(const auto process = process_by_pid.find(queue->pid);
                   process != process_by_pid.end())
                {
                    queue->process_info = process->second;
                }
            }
            queues.push_back(std::move(queue));
        }

        std::unordered_map<size_t, reader_types::agent_info_ptr_t> agent_by_id;
        for(const auto& agent : agents)
        {
            agent_by_id.emplace(agent->id, agent);
        }
        std::unordered_map<size_t, reader_types::thread_info_ptr_t> thread_by_id;
        for(const auto& thread : threads)
        {
            thread_by_id.emplace(thread->id, thread);
        }

        const uint32_t                            pmc_count = in.u32();
        std::vector<reader_types::pmc_info_ptr_t> pmc_infos;
        pmc_infos.reserve(pmc_count);
        for(uint32_t index = 0; index < pmc_count; ++index)
        {
            auto pmc                 = std::make_shared<reader_types::pmc_info_t>();
            pmc->id                  = static_cast<size_t>(in.u64());
            const uint64_t device_id = in.u64();
            pmc->name                = in.text();
            pmc->symbol              = pmc->name;
            pmc->node_info           = node->second;
            if(const auto agent = agent_by_id.find(static_cast<size_t>(device_id));
               agent != agent_by_id.end())
            {
                pmc->agent_info = agent->second;
            }
            pmc_infos.push_back(std::move(pmc));
        }

        const uint32_t                              track_count = in.u32();
        std::vector<reader_types::track_info_ptr_t> tracks;
        std::vector<stored_track_t>                 stored_tracks;
        tracks.reserve(track_count);
        stored_tracks.reserve(track_count);
        for(uint32_t index = 0; index < track_count; ++index)
        {
            stored_track_t stored;
            stored.id            = in.u64();
            stored.kind          = in.u32();
            stored.name          = in.text();
            stored.event_count   = in.u64();
            stored.start_ts      = in.u64();
            stored.end_ts        = in.u64();
            stored.agent_id      = in.u64();
            stored.queue_id      = in.u64();
            stored.pmc_id        = in.u64();
            stored.has_process   = in.u8() != 0;
            stored.pid           = in.u64();
            stored.has_topology  = in.u8() != 0;
            stored.topo_pid      = in.u64();
            stored.topo_tid      = in.u64();
            stored.has_thread    = in.u8() != 0;
            stored.thread_row_id = in.u64();
            stored.has_value     = in.u8() != 0;
            stored.value_min     = in.f64();
            stored.value_max     = in.f64();

            reader_types::track_kind_t kind{};
            if(!known_kind(stored.kind, kind)) return false;
            auto track         = std::make_shared<reader_types::track_info_t>();
            track->id          = static_cast<size_t>(stored.id);
            track->name        = stored.name;
            track->category    = kind;
            track->event_count = static_cast<size_t>(stored.event_count);
            track->start_ts    = static_cast<size_t>(stored.start_ts);
            track->end_ts      = static_cast<size_t>(stored.end_ts);
            track->agent_id    = static_cast<size_t>(stored.agent_id);
            track->queue_id    = static_cast<size_t>(stored.queue_id);
            track->pmc_id      = static_cast<size_t>(stored.pmc_id);
            track->node_info   = node->second;
            if(stored.has_process)
            {
                if(const auto process =
                       process_by_pid.find(static_cast<size_t>(stored.pid));
                   process != process_by_pid.end())
                {
                    track->process_info = process->second;
                }
            }
            if(stored.has_thread)
            {
                if(const auto thread =
                       thread_by_id.find(static_cast<size_t>(stored.thread_row_id));
                   thread != thread_by_id.end())
                {
                    track->thread_info = thread->second;
                }
            }
            if(stored.has_value)
            {
                track->value_range =
                    reader_types::value_range_t{ stored.value_min, stored.value_max };
            }
            tracks.push_back(std::move(track));
            stored_tracks.push_back(std::move(stored));
        }

        std::unordered_map<size_t, reader_types::process_info_ptr_t> processes_by_id;
        std::unordered_map<size_t, reader_types::thread_info_ptr_t>  threads_by_id;
        std::unordered_map<size_t, reader_types::agent_info_ptr_t>   agents_by_id;
        std::unordered_map<topology_key_t,
                           reader_types::track_info_ptr_t,
                           topology_key_hash_t>
            track_by_topology;
        std::unordered_map<reader_types::track_info_ptr_t, topology_key_t>
            topology_by_track;
        std::unordered_map<agent_queue_track_key_t,
                           reader_types::track_info_ptr_t,
                           agent_queue_track_key_hash_t>
            agent_queue_track_by_key;
        for(const auto& process : processes)
            processes_by_id.emplace(process->id, process);
        for(const auto& thread : threads)
            threads_by_id.emplace(thread->id, thread);
        for(const auto& agent : agents)
            agents_by_id.emplace(agent->id, agent);
        for(size_t index = 0; index < tracks.size(); ++index)
        {
            const auto&           track  = tracks[index];
            const stored_track_t& stored = stored_tracks[index];
            if(stored.has_topology)
            {
                const topology_key_t topology{ rocpd_v2::k_node_id,
                                               static_cast<size_t>(stored.topo_pid),
                                               static_cast<size_t>(stored.topo_tid) };
                topology_by_track.emplace(track, topology);
                track_by_topology.emplace(topology, track);
            }
            if(track->category ==
                   reader_types::track_kind_t::kernel_dispatch_agent_queue ||
               track->category == reader_types::track_kind_t::memory_copy_agent_queue)
            {
                agent_queue_track_by_key.emplace(
                    agent_queue_track_key_t{ track->category,
                                             rocpd_v2::k_node_id,
                                             track->agent_id,
                                             track->queue_id },
                    track);
            }
        }

        catalog.processes                = std::move(processes);
        catalog.threads                  = std::move(threads);
        catalog.agents                   = std::move(agents);
        catalog.queues                   = std::move(queues);
        catalog.pmc_infos                = std::move(pmc_infos);
        catalog.tracks                   = std::move(tracks);
        catalog.processes_by_id          = std::move(processes_by_id);
        catalog.threads_by_id            = std::move(threads_by_id);
        catalog.agents_by_id             = std::move(agents_by_id);
        catalog.track_by_topology        = std::move(track_by_topology);
        catalog.topology_by_track        = std::move(topology_by_track);
        catalog.agent_queue_track_by_key = std::move(agent_queue_track_by_key);
        catalog.derived_aggregates_ready = true;
        LOG_INFO("[Profiler-Hub] Using catalog aggregates {}", path.string());
        return true;
    } catch(const std::exception& error)
    {
        LOG_WARNING("[Profiler-Hub] Catalog aggregates were not used: {}", error.what());
        return false;
    }
}

void
save_catalog_aggregates(const reader_catalog_t& catalog)
{
    static_assert(std::endian::native == std::endian::little);
    try
    {
        const std::filesystem::path path = aggregate_file(catalog);
        if(path.empty()) return;
        if(catalog.aggregate_cache_source >= catalog.aggregate_cache_paths.size()) return;

        source_identity_t live;
        if(!stat_source(catalog.aggregate_cache_paths[catalog.aggregate_cache_source],
                        live))
        {
            return;
        }

        std::filesystem::create_directories(path.parent_path());
        const std::filesystem::path temporary = path.string() + ".tmp";
        {
            binary_writer out{ temporary };
            out.bytes(k_magic, 4);
            out.u32(k_format_version);
            out.u32(catalog.aggregate_cache_source);
            out.u64(live.size);
            out.u64(static_cast<uint64_t>(live.mtime));
            out.text(live.path);

            out.u32(static_cast<uint32_t>(catalog.processes.size()));
            for(const auto& process : catalog.processes)
            {
                out.u64(static_cast<uint64_t>(process->pid));
            }

            out.u32(static_cast<uint32_t>(catalog.threads.size()));
            for(const auto& thread : catalog.threads)
            {
                out.u64(static_cast<uint64_t>(thread->id));
                out.u64(static_cast<uint64_t>(thread->parent_process_id.value_or(0)));
                out.u64(static_cast<uint64_t>(thread->thread_id));
            }

            out.u32(static_cast<uint32_t>(catalog.agents.size()));
            for(const auto& agent : catalog.agents)
            {
                out.u64(static_cast<uint64_t>(agent->id));
            }

            out.u32(static_cast<uint32_t>(catalog.queues.size()));
            for(const auto& queue : catalog.queues)
            {
                const bool has_owner = queue->process_info != nullptr;
                out.u64(queue_gpu_id(*queue));
                out.u64(static_cast<uint64_t>(queue->queue_id));
                out.u8(has_owner ? 1 : 0);
                out.u64(has_owner ? static_cast<uint64_t>(queue->pid) : 0);
            }

            out.u32(static_cast<uint32_t>(catalog.pmc_infos.size()));
            for(const auto& pmc : catalog.pmc_infos)
            {
                out.u64(static_cast<uint64_t>(pmc->id));
                out.u64(pmc->agent_info ? static_cast<uint64_t>(pmc->agent_info->id) : 0);
                out.text(pmc->name);
            }

            out.u32(static_cast<uint32_t>(catalog.tracks.size()));
            for(const auto& track : catalog.tracks)
            {
                const auto topology     = catalog.topology_by_track.find(track);
                const bool has_topology = topology != catalog.topology_by_track.end();
                out.u64(static_cast<uint64_t>(track->id));
                out.u32(static_cast<uint32_t>(track->category));
                out.text(track->name);
                out.u64(static_cast<uint64_t>(track->event_count));
                out.u64(static_cast<uint64_t>(track->start_ts));
                out.u64(static_cast<uint64_t>(track->end_ts));
                out.u64(static_cast<uint64_t>(track->agent_id));
                out.u64(static_cast<uint64_t>(track->queue_id));
                out.u64(static_cast<uint64_t>(track->pmc_id));
                out.u8(track->process_info ? 1 : 0);
                out.u64(track->process_info
                            ? static_cast<uint64_t>(track->process_info->pid)
                            : 0);
                out.u8(has_topology ? 1 : 0);
                out.u64(has_topology ? static_cast<uint64_t>(topology->second.pid) : 0);
                out.u64(has_topology ? static_cast<uint64_t>(topology->second.tid) : 0);
                out.u8(track->thread_info ? 1 : 0);
                out.u64(track->thread_info ? static_cast<uint64_t>(track->thread_info->id)
                                           : 0);
                out.u8(track->value_range ? 1 : 0);
                out.f64(track->value_range ? track->value_range->min : 0);
                out.f64(track->value_range ? track->value_range->max : 0);
            }
        }
        rename_into_place(temporary, path);
        LOG_INFO("[Profiler-Hub] Writing catalog aggregates {}", path.string());
    } catch(const std::exception& error)
    {
        LOG_WARNING("[Profiler-Hub] Catalog aggregates were not saved: {}", error.what());
    }
}
#else
bool
load_catalog_aggregates(reader_catalog_t&)
{
    return false;
}

void
save_catalog_aggregates(const reader_catalog_t&)
{}
#endif

#if PROFILER_HUB_ENABLE_ROCPD_V3
bool
load_derived_tracks(reader_catalog_t& catalog)
{
    static_assert(std::endian::native == std::endian::little);
    if(catalog.derived_aggregates_ready) return true;

    try
    {
        const std::filesystem::path path = derived_file(catalog);
        if(path.empty() || !std::filesystem::is_regular_file(path)) return false;
        if(catalog.aggregate_cache_source >= catalog.aggregate_cache_paths.size())
            return false;

        source_identity_t live;
        if(!stat_source(catalog.aggregate_cache_paths[catalog.aggregate_cache_source],
                        live))
        {
            return false;
        }

        binary_reader in{ path };
        char          magic[4] = {};
        in.bytes(magic, 4);
        if(!same_magic(magic, k_derived_magic) || in.u32() != k_derived_format_version)
        {
            return false;
        }
        if(in.u32() != catalog.aggregate_cache_source) return false;
        if(in.u64() != live.size || in.u64() != static_cast<uint64_t>(live.mtime))
            return false;
        if(in.text() != live.path) return false;

        const uint32_t                              track_count = in.u32();
        std::vector<reader_types::track_info_ptr_t> tracks;
        tracks.reserve(track_count);
        std::unordered_map<size_t, reader_types::track_info_ptr_t> tracks_by_id;
        std::unordered_map<topology_key_t,
                           reader_types::track_info_ptr_t,
                           topology_key_hash_t>
            track_by_topology;
        std::unordered_map<reader_types::track_info_ptr_t, topology_key_t>
                                                                   topology_by_track;
        std::unordered_map<reader_types::track_info_ptr_t, size_t> db_id_by_track;
        std::unordered_map<size_t, reader_types::track_info_ptr_t> sample_track_by_db_id;
        std::unordered_map<stream_track_key_t,
                           reader_types::track_info_ptr_t,
                           stream_track_key_hash_t>
            stream_track_by_key;
        std::unordered_map<agent_queue_track_key_t,
                           reader_types::track_info_ptr_t,
                           agent_queue_track_key_hash_t>
            agent_queue_track_by_key;
        std::unordered_map<size_t, std::vector<memory_activity_point_t>> memory_activity;
        uint64_t memory_first_id = 0;
        uint64_t memory_last_id  = 0;
        bool     memory_has_ids  = false;

        for(uint32_t index = 0; index < track_count; ++index)
        {
            reader_types::track_kind_t kind{};
            const uint64_t             id = in.u64();
            if(!known_derived_kind(in.u32(), kind)) return false;
            auto track             = std::make_shared<reader_types::track_info_t>();
            track->id              = static_cast<size_t>(id);
            track->category        = kind;
            track->name            = in.text();
            track->extdata         = in.text();
            track->event_count     = static_cast<size_t>(in.u64());
            track->agent_id        = static_cast<size_t>(in.u64());
            track->queue_id        = static_cast<size_t>(in.u64());
            track->stream_id       = static_cast<size_t>(in.u64());
            track->db_pid          = static_cast<size_t>(in.u64());
            track->pmc_id          = static_cast<size_t>(in.u64());
            track->start_ts        = static_cast<size_t>(in.u64());
            track->end_ts          = static_cast<size_t>(in.u64());
            const bool   has_value = in.u8() != 0;
            const double value_min = in.f64();
            const double value_max = in.f64();
            if(has_value)
            {
                track->value_range = reader_types::value_range_t{ value_min, value_max };
            }

            const size_t nid     = static_cast<size_t>(in.u64());
            const bool   has_pid = in.u8() != 0;
            const size_t pid     = static_cast<size_t>(in.u64());
            const bool   has_tid = in.u8() != 0;
            const size_t tid     = static_cast<size_t>(in.u64());
            if(const auto node = catalog.nodes_by_id.find(nid);
               node != catalog.nodes_by_id.end())
            {
                track->node_info = node->second;
            }
            if(has_pid)
            {
                if(const auto process = catalog.processes_by_id.find(pid);
                   process != catalog.processes_by_id.end())
                {
                    track->process_info = process->second;
                }
            }
            if(has_tid)
            {
                if(const auto thread = catalog.threads_by_id.find(tid);
                   thread != catalog.threads_by_id.end())
                {
                    track->thread_info = thread->second;
                }
            }

            const bool           has_topology = in.u8() != 0;
            const topology_key_t topology{ static_cast<size_t>(in.u64()),
                                           static_cast<size_t>(in.u64()),
                                           static_cast<size_t>(in.u64()) };
            const bool           has_db_id  = in.u8() != 0;
            const uint64_t       db_id      = in.u64();
            const bool           has_sample = in.u8() != 0;
            const uint64_t       sample_id  = in.u64();
            if(has_topology)
            {
                topology_by_track.emplace(track, topology);
                track_by_topology.emplace(topology, track);
            }
            if(has_db_id)
            {
                db_id_by_track.emplace(track, static_cast<size_t>(db_id));
            }
            if(has_sample)
            {
                sample_track_by_db_id.emplace(static_cast<size_t>(sample_id), track);
            }
            if(agent_queue_kind(kind))
            {
                agent_queue_track_by_key.emplace(
                    agent_queue_track_key_t{
                        kind, nid, track->agent_id, track->queue_id },
                    track);
            }
            if(kind == reader_types::track_kind_t::stream)
            {
                stream_track_by_key.emplace(
                    stream_track_key_t{ nid, track->db_pid, track->stream_id }, track);
            }

            const uint32_t point_count = in.u32();
            if(point_count > 0)
            {
                std::vector<memory_activity_point_t> points;
                points.reserve(point_count);
                for(uint32_t point_index = 0; point_index < point_count; ++point_index)
                {
                    memory_activity_point_t point;
                    point.id    = in.u64();
                    point.start = in.u64();
                    point.end   = in.u64();
                    point.value = in.f64();
                    if(!memory_has_ids)
                    {
                        memory_first_id = point.id;
                        memory_last_id  = point.id;
                        memory_has_ids  = true;
                    }
                    else
                    {
                        if(point.id < memory_first_id) memory_first_id = point.id;
                        if(point.id > memory_last_id) memory_last_id = point.id;
                    }
                    points.push_back(point);
                }
                memory_activity.emplace(track->id, std::move(points));
            }

            tracks_by_id.emplace(track->id, track);
            tracks.push_back(std::move(track));
        }

        const uint32_t                            link_count = in.u32();
        reader_types::stream_hardware_link_list_t links;
        links.reserve(link_count);
        for(uint32_t index = 0; index < link_count; ++index)
        {
            reader_types::stream_hardware_link_t link;
            if(!known_derived_kind(in.u32(), link.category)) return false;
            link.nid               = static_cast<size_t>(in.u64());
            link.pid               = static_cast<size_t>(in.u64());
            link.stream_id         = static_cast<size_t>(in.u64());
            const bool   has_agent = in.u8() != 0;
            const size_t agent     = static_cast<size_t>(in.u64());
            const bool   has_queue = in.u8() != 0;
            const size_t queue     = static_cast<size_t>(in.u64());
            if(has_agent) link.agent_id = agent;
            if(has_queue) link.queue_id = queue;
            const bool     has_stream   = in.u8() != 0;
            const uint64_t stream_id    = in.u64();
            const bool     has_hardware = in.u8() != 0;
            const uint64_t hardware_id  = in.u64();
            if(has_stream)
            {
                if(const auto found = tracks_by_id.find(static_cast<size_t>(stream_id));
                   found != tracks_by_id.end())
                {
                    link.stream_track = found->second;
                }
            }
            if(has_hardware)
            {
                if(const auto found = tracks_by_id.find(static_cast<size_t>(hardware_id));
                   found != tracks_by_id.end())
                {
                    link.hardware_track = found->second;
                }
            }
            links.push_back(std::move(link));
        }

        catalog.tracks                     = std::move(tracks);
        catalog.track_by_topology          = std::move(track_by_topology);
        catalog.topology_by_track          = std::move(topology_by_track);
        catalog.db_id_by_track             = std::move(db_id_by_track);
        catalog.sample_track_by_db_id      = std::move(sample_track_by_db_id);
        catalog.stream_track_by_key        = std::move(stream_track_by_key);
        catalog.agent_queue_track_by_key   = std::move(agent_queue_track_by_key);
        catalog.stream_hardware_links      = std::move(links);
        catalog.m_memory_activity          = std::move(memory_activity);
        catalog.m_memory_activity_first_id = memory_first_id;
        catalog.m_memory_activity_last_id  = memory_last_id;
        catalog.m_memory_activity_has_ids  = memory_has_ids;
        catalog.derived_aggregates_ready   = true;
        LOG_INFO("[Profiler-Hub] Using derived tracks {}", path.string());
        return true;
    } catch(const std::exception& error)
    {
        LOG_WARNING("[Profiler-Hub] Derived tracks were not used: {}", error.what());
        return false;
    }
}

void
save_derived_tracks(const reader_catalog_t& catalog)
{
    static_assert(std::endian::native == std::endian::little);
    try
    {
        const std::filesystem::path path = derived_file(catalog);
        if(path.empty()) return;
        if(catalog.aggregate_cache_source >= catalog.aggregate_cache_paths.size()) return;

        source_identity_t live;
        if(!stat_source(catalog.aggregate_cache_paths[catalog.aggregate_cache_source],
                        live))
        {
            return;
        }

        std::unordered_map<reader_types::track_info_ptr_t, size_t> sample_id_by_track;
        sample_id_by_track.reserve(catalog.sample_track_by_db_id.size());
        for(const auto& entry : catalog.sample_track_by_db_id)
        {
            sample_id_by_track.emplace(entry.second, entry.first);
        }
        std::unordered_map<reader_types::track_info_ptr_t, size_t> nid_by_track;
        for(const auto& entry : catalog.agent_queue_track_by_key)
        {
            nid_by_track.emplace(entry.second, entry.first.nid);
        }
        for(const auto& entry : catalog.stream_track_by_key)
        {
            nid_by_track.emplace(entry.second, entry.first.nid);
        }
        for(const auto& entry : catalog.topology_by_track)
        {
            if(entry.second.nid != std::numeric_limits<size_t>::max())
            {
                nid_by_track.emplace(entry.first, entry.second.nid);
            }
        }

        std::filesystem::create_directories(path.parent_path());
        const std::filesystem::path temporary = path.string() + ".tmp";
        {
            binary_writer out{ temporary };
            out.bytes(k_derived_magic, 4);
            out.u32(k_derived_format_version);
            out.u32(catalog.aggregate_cache_source);
            out.u64(live.size);
            out.u64(static_cast<uint64_t>(live.mtime));
            out.text(live.path);

            out.u32(fit_count(catalog.tracks.size()));
            for(const auto& track : catalog.tracks)
            {
                const auto topology     = catalog.topology_by_track.find(track);
                const bool has_topology = topology != catalog.topology_by_track.end();
                const auto db_id        = catalog.db_id_by_track.find(track);
                const bool has_db_id    = db_id != catalog.db_id_by_track.end();
                const auto sample_id    = sample_id_by_track.find(track);
                const bool has_sample   = sample_id != sample_id_by_track.end();
                const auto nid          = nid_by_track.find(track);
                const auto points       = catalog.m_memory_activity.find(track->id);

                out.u64(static_cast<uint64_t>(track->id));
                out.u32(static_cast<uint32_t>(track->category));
                out.text(track->name);
                out.text(track->extdata);
                out.u64(static_cast<uint64_t>(track->event_count));
                out.u64(static_cast<uint64_t>(track->agent_id));
                out.u64(static_cast<uint64_t>(track->queue_id));
                out.u64(static_cast<uint64_t>(track->stream_id));
                out.u64(static_cast<uint64_t>(track->db_pid));
                out.u64(static_cast<uint64_t>(track->pmc_id));
                out.u64(static_cast<uint64_t>(track->start_ts));
                out.u64(static_cast<uint64_t>(track->end_ts));
                out.u8(track->value_range ? 1 : 0);
                out.f64(track->value_range ? track->value_range->min : 0);
                out.f64(track->value_range ? track->value_range->max : 0);
                const uint64_t stored_nid =
                    nid != nid_by_track.end() ? static_cast<uint64_t>(nid->second)
                    : track->node_info ? static_cast<uint64_t>(track->node_info->node_id)
                                       : 0;
                out.u64(stored_nid);
                out.u8(track->process_info ? 1 : 0);
                out.u64(track->process_info
                            ? static_cast<uint64_t>(track->process_info->id)
                            : 0);
                out.u8(track->thread_info ? 1 : 0);
                out.u64(track->thread_info ? static_cast<uint64_t>(track->thread_info->id)
                                           : 0);
                out.u8(has_topology ? 1 : 0);
                out.u64(has_topology ? static_cast<uint64_t>(topology->second.nid) : 0);
                out.u64(has_topology ? static_cast<uint64_t>(topology->second.pid) : 0);
                out.u64(has_topology ? static_cast<uint64_t>(topology->second.tid) : 0);
                out.u8(has_db_id ? 1 : 0);
                out.u64(has_db_id ? static_cast<uint64_t>(db_id->second) : 0);
                out.u8(has_sample ? 1 : 0);
                out.u64(has_sample ? static_cast<uint64_t>(sample_id->second) : 0);

                if(points == catalog.m_memory_activity.end())
                {
                    out.u32(0);
                }
                else
                {
                    out.u32(fit_count(points->second.size()));
                    for(const memory_activity_point_t& point : points->second)
                    {
                        out.u64(point.id);
                        out.u64(point.start);
                        out.u64(point.end);
                        out.f64(point.value);
                    }
                }
            }

            out.u32(fit_count(catalog.stream_hardware_links.size()));
            for(const reader_types::stream_hardware_link_t& link :
                catalog.stream_hardware_links)
            {
                out.u32(static_cast<uint32_t>(link.category));
                out.u64(static_cast<uint64_t>(link.nid));
                out.u64(static_cast<uint64_t>(link.pid));
                out.u64(static_cast<uint64_t>(link.stream_id));
                out.u8(link.agent_id ? 1 : 0);
                out.u64(link.agent_id ? static_cast<uint64_t>(link.agent_id.value()) : 0);
                out.u8(link.queue_id ? 1 : 0);
                out.u64(link.queue_id ? static_cast<uint64_t>(link.queue_id.value()) : 0);
                out.u8(link.stream_track ? 1 : 0);
                out.u64(link.stream_track ? static_cast<uint64_t>(link.stream_track->id)
                                          : 0);
                out.u8(link.hardware_track ? 1 : 0);
                out.u64(link.hardware_track
                            ? static_cast<uint64_t>(link.hardware_track->id)
                            : 0);
            }
        }
        rename_into_place(temporary, path);
        LOG_INFO("[Profiler-Hub] Writing derived tracks {}", path.string());
    } catch(const std::exception& error)
    {
        LOG_WARNING("[Profiler-Hub] Derived tracks were not saved: {}", error.what());
    }
}
#else
bool
load_derived_tracks(reader_catalog_t&)
{
    return false;
}

void
save_derived_tracks(const reader_catalog_t&)
{}
#endif

}  // namespace profiler_hub
