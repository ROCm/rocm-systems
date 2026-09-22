#include "profiler_hub_ctx.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "profiler_hub_future.hpp"

#include <algorithm>
#include <optional>
#include <thread>
#include <tuple>

namespace
{

std::optional<profiler_hub::reader_types::track_info_ptr_t>
find_track(profiler_hub::reader_t& reader, uint32_t track_id)
{
    const auto tracks = reader.get_all_tracks();
    const auto track  = std::ranges::find_if(tracks, [track_id](const auto& item) {
        return static_cast<uint32_t>(item->id) == track_id;
    });

    return track != tracks.end() ? std::make_optional(*track) : std::nullopt;
}

}  // namespace

size_t
ph_ctx::default_thread_pool_size()
{
    const auto hw = std::thread::hardware_concurrency();
    return std::max<size_t>(1, hw / 2);
}

ph_ctx::ph_ctx(std::string_view trace_path)
: m_file_path{ trace_path }
{
    profiler_hub::storage_t version_probe{ m_file_path, "" };
    const auto              version = version_probe.get_storage_version();
    m_schema_version                = { .major = version.major,
                                        .minor = version.minor,
                                        .patch = version.patch };

    initialize_track_list();
    initialize_node_agents();
    initilaize_node_info();
}

ph_ctx::~ph_ctx()
{
    std::scoped_lock lock{ m_futures_mutex };
    for(auto* future : m_live_futures)
    {
        std::ignore = future->m_handle.cancel();
        future->m_handle.wait();
    }
}

void
ph_ctx::register_future(ph_future* future)
{
    std::scoped_lock lock{ m_futures_mutex };
    m_live_futures.insert(future);
}

void
ph_ctx::unregister_future(ph_future* future)
{
    std::scoped_lock lock{ m_futures_mutex };
    m_live_futures.erase(future);
}

bool
ph_ctx::owns_future(ph_future* future) const
{
    std::scoped_lock lock{ m_futures_mutex };
    return m_live_futures.contains(future);
}

ph_schema_version_t
ph_ctx::get_storage_version()
{
    return m_schema_version;
}

ph_track_list_t
ph_ctx::get_track_list()
{
    return ph_track_list_t{ .list_size = static_cast<std::uint32_t>(m_c_tracks.size()),
                            .tracks    = m_c_tracks.data() };
}

ph_node_t
ph_ctx::get_node()
{
    return *m_c_node;
}

ph_event_list_t
ph_ctx::core_get_track_events(profiler_hub::common::connection& conn,
                              uint32_t                          track_id,
                              uint64_t                          start_ts,
                              uint64_t                          end_ts)
{
    const auto track = find_track(conn.reader(), track_id);
    if(!track.has_value())
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    profiler_hub::reader_types::event_filter_t filter;
    if(start_ts != 0 || end_ts != 0)
    {
        filter.time_window.start = start_ts;
        filter.time_window.end   = end_ts;
    }

    auto events = conn.reader().get_events_for_track(track.value(), filter);

    std::scoped_lock lock{ m_track_results_mutex };
    auto&            result = m_track_events_results.emplace_back();
    result.events           = std::move(events);
    result.c_events.reserve(result.events.size());
    for(const auto& event : result.events)
    {
        result.c_events.push_back(ph_event_t{
            .start = event.start_timestamp,
            .end   = event.end_timestamp,
            .name  = event.display_name.empty() ? "" : event.display_name.data(),
        });
    }

    return ph_event_list_t{ .list_size =
                                static_cast<std::uint32_t>(result.c_events.size()),
                            .events = result.c_events.data() };
}

ph_sample_list_t
ph_ctx::core_get_track_samples(profiler_hub::common::connection& conn,
                               uint32_t                          track_id,
                               uint64_t                          start_ts,
                               uint64_t                          end_ts)
{
    const auto track = find_track(conn.reader(), track_id);
    if(!track.has_value())
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    profiler_hub::reader_types::event_filter_t filter;
    if(start_ts != 0 || end_ts != 0)
    {
        filter.time_window.start = start_ts;
        filter.time_window.end   = end_ts;
    }

    auto samples = conn.reader().get_counter_events_for_track(track.value(), filter);

    std::scoped_lock lock{ m_track_results_mutex };
    auto&            c_samples = m_track_samples_results.emplace_back();
    c_samples.reserve(samples.size());
    for(const auto& sample : samples)
    {
        c_samples.push_back(
            ph_sample_t{ .timestamp = sample.timestamp, .value = sample.value });
    }

    return ph_sample_list_t{ .list_size = static_cast<std::uint32_t>(c_samples.size()),
                             .samples   = c_samples.data() };
}

ph_event_list_t
ph_ctx::get_track_events(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    if(!m_track_by_id.contains(track_id))
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    return m_connection_pool.run_sync([&](profiler_hub::common::connection& conn) {
        return core_get_track_events(conn, track_id, start_ts, end_ts);
    });
}

ph_sample_list_t
ph_ctx::get_track_samples(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    if(!m_track_by_id.contains(track_id))
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    return m_connection_pool.run_sync([&](profiler_hub::common::connection& conn) {
        return core_get_track_samples(conn, track_id, start_ts, end_ts);
    });
}

void
ph_ctx::initialize_track_list()
{
    const auto all_tracks =
        m_connection_pool.run_sync([](profiler_hub::common::connection& conn) {
            return conn.reader().get_all_tracks();
        });

    m_tracks.reserve(all_tracks.size());
    m_c_tracks.reserve(all_tracks.size());

    for(const auto& track : all_tracks)
    {
        if(track->event_count == 0)
        {
            continue;
        }

        m_tracks.push_back(track);
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
            .event_count = static_cast<std::uint32_t>(track->event_count),
            .agent_id    = static_cast<std::uint32_t>(track->agent_id),
        });
    }
}

void
ph_ctx::initilaize_node_info()
{
    m_c_node = std::make_unique<ph_node_t>();
    m_nodes  = m_connection_pool.run_sync([](profiler_hub::common::connection& conn) {
        return conn.reader().get_all_nodes();
    });
    const auto& node = m_nodes[0];

    m_c_node->info = {
        .id            = static_cast<uint32_t>(node->node_id),
        .machine_id    = node->machine_id.c_str(),
        .system_name   = node->system_name.c_str(),
        .hostname      = node->hostname.c_str(),
        .release       = node->release.c_str(),
        .version       = node->version.c_str(),
        .hardware_name = node->hardware_name.c_str(),
        .domain_name   = node->domain_name.c_str(),
    };

    m_c_node->agents =
        ph_agent_list_t{ .list_size = static_cast<std::uint32_t>(m_c_agents.size()),
                         .agents    = m_c_agents.data() };

    m_c_node->track_list = get_track_list();
}

void
ph_ctx::initialize_node_agents()
{
    m_agents = m_connection_pool.run_sync([](profiler_hub::common::connection& conn) {
        return conn.reader().get_all_agents();
    });
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
