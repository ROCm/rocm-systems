#include "ph_ctx.hpp"
#include "debug.hpp"
#include "ph_future.hpp"
#include "populate_reader_catalog.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "reader_catalog.hpp"
#include "track_read_options.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace
{

std::string
existing_trace_path(std::string_view path)
{
    std::string file_path{ path };
    if(!std::filesystem::is_regular_file(file_path))
    {
        throw std::runtime_error("trace file does not exist: " + file_path);
    }
    return file_path;
}

ph_track_category_t
to_c_track_category(profiler_hub::reader_types::track_kind_t kind)
{
    using profiler_hub::reader_types::track_kind_t;
    switch(kind)
    {
        case track_kind_t::thread: return PH_TRACK_CATEGORY_THREAD;
        case track_kind_t::thread_sample: return PH_TRACK_CATEGORY_THREAD_SAMPLE;
        case track_kind_t::pmc_agent: return PH_TRACK_CATEGORY_PMC_AGENT;
        case track_kind_t::kernel_dispatch_agent_queue:
            return PH_TRACK_CATEGORY_KERNEL_DISPATCH_AGENT_QUEUE;
        case track_kind_t::memory_allocate_agent_queue:
            return PH_TRACK_CATEGORY_MEMORY_ALLOCATE_AGENT_QUEUE;
        case track_kind_t::memory_copy_agent_queue:
            return PH_TRACK_CATEGORY_MEMORY_COPY_AGENT_QUEUE;
        case track_kind_t::stream: return PH_TRACK_CATEGORY_STREAM;
    }
    return PH_TRACK_CATEGORY_THREAD;
}

}  // namespace

size_t
ph_ctx::default_thread_pool_size()
{
    return std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
}

size_t
ph_ctx::default_connection_count()
{
    return 8;
}

ph_ctx::ph_ctx(std::string_view trace_path)
: m_file_path{ existing_trace_path(trace_path) }
, m_catalog{ std::make_shared<profiler_hub::reader_catalog_t>() }
{}

ph_ctx::~ph_ctx()
{
    if(!m_load_task) return;

    std::ignore = m_load_task->cancel();
    m_load_task->wait();

    // A load that never got a worker did not run, so nobody ended its future.
    set_load_state_if_loading(load_state_t::cancelled);
    m_load_future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
}

void
ph_ctx::load()
{
    try
    {
        std::ignore = read_trace([] { return false; }, {});
    } catch(...)
    {
        set_load_state(load_state_t::failed);
        throw;
    }
    set_load_state(load_state_t::ready);
}

void
ph_ctx::load_async(std::shared_ptr<ph_future> future)
{
    m_load_future = future;
    m_load_task   = m_thread_pool.submit([this, future](const std::stop_token& token) {
        const auto stop_requested = [&] {
            return token.stop_requested() || future->stop_token().stop_requested();
        };
        const auto progress = [&](double value) { future->report_progress(value); };

        try
        {
            if(read_trace(stop_requested, progress))
            {
                set_load_state(load_state_t::ready);
                future->finish(PH_FUTURE_FINISHED, PH_RESULT_SUCCESS);
            }
            else
            {
                set_load_state(load_state_t::cancelled);
                future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
            }
        } catch(...)
        {
            set_load_state(load_state_t::failed);
            future->finish(PH_FUTURE_ERROR, PH_RESULT_CONTEXT_ALLOCATION_FAILED);
        }
    });
}

void
ph_ctx::run_async(std::shared_ptr<ph_future>   future,
                  std::function<ph_result_t()> operation)
{
    m_operations.add(future);
    try
    {
        std::ignore =
            m_thread_pool.submit([this, future, operation = std::move(operation)](
                                     const std::stop_token& token) {
                if(token.stop_requested() || future->stop_token().stop_requested())
                {
                    future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
                }
                else
                {
                    const auto result = operation();
                    future->finish(result == PH_RESULT_SUCCESS ? PH_FUTURE_FINISHED
                                                               : PH_FUTURE_ERROR,
                                   result);
                }
                m_operations.remove(future);
            });
    } catch(...)
    {
        m_operations.remove(future);
        throw;
    }
}

ph_result_t
ph_ctx::wait_until_ready()
{
    std::unique_lock lock{ m_state_mutex };
    m_state_cv.wait(lock, [this] { return m_state != load_state_t::loading; });

    switch(m_state)
    {
        case load_state_t::ready: return PH_RESULT_SUCCESS;
        case load_state_t::cancelled: return PH_RESULT_CANCELLED;
        case load_state_t::loading:
        case load_state_t::failed: break;
    }
    return PH_RESULT_INVALID_CONTEXT;
}

void
ph_ctx::set_load_state(load_state_t state)
{
    {
        const std::scoped_lock lock{ m_state_mutex };
        m_state = state;
    }
    m_state_cv.notify_all();
}

void
ph_ctx::set_load_state_if_loading(load_state_t state)
{
    {
        const std::scoped_lock lock{ m_state_mutex };
        if(m_state != load_state_t::loading) return;
        m_state = state;
    }
    m_state_cv.notify_all();
}

bool
ph_ctx::read_trace(const stop_requested_fn& stop_requested, const progress_fn& progress)
{
    if(stop_requested()) return false;

    profiler_hub::storage_t version_probe{ m_file_path, "" };
    const auto              version = version_probe.get_storage_version();
    m_schema_version                = { .major = version.major,
                                        .minor = version.minor,
                                        .patch = version.patch };

    populate_reader_catalog(m_thread_pool, m_connection_pool, *m_catalog);
    if(stop_requested()) return false;

    initialize_track_list();
    if(!load_all_tracks(stop_requested, progress)) return false;

    initialize_node_agents();
    initialize_node_processes();
    initialize_node_info();
    return true;
}

ph_schema_version_t
ph_ctx::get_schema_version()
{
    return m_schema_version;
}

ph_track_list_t
ph_ctx::get_track_list()
{
    LOG_DEBUG("[Profiler-Hub] Get track list");
    return ph_track_list_t{ .list_size = static_cast<std::uint32_t>(m_c_tracks.size()),
                            .tracks    = m_c_tracks.data() };
}

ph_node_t
ph_ctx::get_node()
{
    return *m_c_node;
}

ph_event_list_t
ph_ctx::get_track_events(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    LOG_DEBUG("[Profiler-Hub] Get track events. Track id {}, time slice [{} - {}]",
              track_id,
              start_ts,
              end_ts);
    const auto track_it = m_track_by_id.find(track_id);
    if(track_it == m_track_by_id.end())
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    return m_track_reader.events(track_it->second, start_ts, end_ts);
}

ph_sample_list_t
ph_ctx::get_track_samples(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    LOG_DEBUG("[Profiler-Hub] Get track samples. Track id {}, time slice [{} - {}]",
              track_id,
              start_ts,
              end_ts);
    const auto track_it = m_track_by_id.find(track_id);
    if(track_it == m_track_by_id.end())
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    return m_track_reader.samples(track_it->second, start_ts, end_ts);
}

void
ph_ctx::initialize_track_list()
{
    const auto& all_tracks = m_catalog->tracks;

    m_c_tracks.reserve(all_tracks.size());

    for(const auto& track : all_tracks)
    {
        if(track->event_count == 0)
        {
            continue;
        }

        m_track_by_id.emplace(static_cast<std::uint32_t>(track->id), track);
        m_c_tracks.push_back(ph_track_t{
            .id          = static_cast<std::uint32_t>(track->id),
            .track_name  = track->name.c_str(),
            .nid         = track->node_info
                               ? static_cast<std::uint32_t>(track->node_info->node_id)
                               : 0,
            .pid         = track->process_info
                               ? static_cast<std::uint32_t>(track->process_info->pid)
                               : 0,
            .tid         = track->thread_info  // Some tracks have no thread (e.g. per
                                       // agent+queue category tracks)
                               ? static_cast<std::uint32_t>(track->thread_info->thread_id)
                               : 0,
            .event_count = static_cast<std::uint32_t>(std::min<size_t>(
                track->event_count, std::numeric_limits<std::uint32_t>::max())),
            .agent_id    = static_cast<std::uint32_t>(track->agent_id),
            .category    = to_c_track_category(track->category),
            .queue_id    = static_cast<std::uint32_t>(track->queue_id),
            .stream_id   = static_cast<std::uint32_t>(track->stream_id),
            .start_ts    = static_cast<std::uint64_t>(track->start_ts),
            .end_ts      = static_cast<std::uint64_t>(track->end_ts),
            .value_range = track->value_range ? ph_value_range_t{ track->value_range->min,
                                                                  track->value_range->max,
                                                                  1 }
                                              : ph_value_range_t{},
        });
    }
}

bool
ph_ctx::load_all_tracks(const stop_requested_fn& stop_requested,
                        const progress_fn&       progress)
{
    const auto total_events = std::accumulate(
        m_c_tracks.begin(),
        m_c_tracks.end(),
        uint64_t{ 0 },
        [](uint64_t sum, const auto& track) { return sum + track.event_count; });

    uint64_t loaded_events = 0;
    for(auto& c_track : m_c_tracks)
    {
        if(stop_requested()) return false;

        const auto& track = m_track_by_id.at(c_track.id);
        if(track->category == profiler_hub::reader_types::track_kind_t::pmc_agent)
        {
            std::ignore = m_track_reader.samples(track, 0, 0);
        }
        else
        {
            std::ignore           = m_track_reader.events(track, 0, 0);
            c_track.nesting_depth = m_track_reader.nesting_depth(track);
        }

        loaded_events += c_track.event_count;
        if(progress && total_events != 0)
        {
            progress(static_cast<double>(loaded_events) /
                     static_cast<double>(total_events));
        }
    }
    return true;
}

void
ph_ctx::initialize_node_info()
{
    m_c_node = std::make_unique<ph_node_t>();
    m_nodes  = m_catalog->nodes;

    m_c_node->info = ph_node_info_t{ .id            = 0,
                                     .machine_id    = "",
                                     .system_name   = "",
                                     .hostname      = "",
                                     .release       = "",
                                     .version       = "",
                                     .hardware_name = "",
                                     .domain_name   = "" };

    if(!m_nodes.empty())
    {
        const auto& node = m_nodes[0];
        m_c_node->info   = {
              .id            = static_cast<uint32_t>(node->node_id),
              .machine_id    = node->machine_id.c_str(),
              .system_name   = node->system_name.c_str(),
              .hostname      = node->hostname.c_str(),
              .release       = node->release.c_str(),
              .version       = node->version.c_str(),
              .hardware_name = node->hardware_name.c_str(),
              .domain_name   = node->domain_name.c_str(),
        };
    }

    m_c_node->agents =
        ph_agent_list_t{ .list_size = static_cast<std::uint32_t>(m_c_agents.size()),
                         .agents    = m_c_agents.data() };

    m_c_node->process_list =
        ph_process_list_t{ .list_size = static_cast<std::uint32_t>(m_c_processes.size()),
                           .processes = m_c_processes.data() };

    m_c_node->track_list = get_track_list();
}

void
ph_ctx::initialize_node_agents()
{
    m_agents = m_catalog->agents;
    m_c_agents.reserve(m_agents.size());

    for(const auto& agent : m_agents)
    {
        m_c_agents.push_back(ph_agent_t{
            .id         = static_cast<std::uint32_t>(agent->type_index),
            .agent_type = agent->agent_type.c_str(),
            .absolute_index =
                static_cast<std::uint32_t>(agent->absolute_index.value_or(0)),
            .logical_index = static_cast<std::uint32_t>(agent->logical_index.value_or(0)),
            .uuid          = static_cast<std::uint32_t>(agent->uuid.value_or(0)),
            .name          = agent->name.c_str(),
            .model_name    = agent->model_name.c_str(),
            .vendor_name   = agent->vendor_name.c_str(),
            .product_name  = agent->product_name.c_str(),
            .user_name     = agent->user_name.c_str(),
        });
    }
}

void
ph_ctx::initialize_node_processes()
{
    m_processes = m_catalog->processes;
    m_c_processes.reserve(m_processes.size());

    for(const auto& process : m_processes)
    {
        m_c_processes.push_back(ph_process_t{
            .id      = static_cast<std::uint32_t>(process->pid),
            .command = process->command.c_str(),
        });
    }
}
