#include "profiler_hub_ctx.hpp"
#include "common/natural_merge_sort.hpp"
#include "debug.hpp"
#include "populate_reader_catalog.hpp"
#include "profiler-hub/cpp/storage.hpp"
#include "profiler_hub_future.hpp"
#include "reader_catalog.hpp"
#include "track_read_options.hpp"
#include "track_window.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
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

std::uint32_t
to_list_size(size_t size)
{
    if(size > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("result has more elements than the C API can report");
    }
    return static_cast<std::uint32_t>(size);
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
    const auto hw = std::thread::hardware_concurrency();
    return std::max<size_t>(1,
                            profiler_hub::parse_size(std::getenv("PH_POOL_THREADS"),
                                                     std::max<size_t>(1, hw / 2)));
}

size_t
ph_ctx::default_connection_count()
{
    return std::max<size_t>(1,
                            profiler_hub::parse_size(std::getenv("PH_CONNECTIONS"), 8));
}

ph_ctx::ph_ctx(std::string_view trace_path)
: m_file_path{ existing_trace_path(trace_path) }
, m_read_options{ profiler_hub::track_read_options::from_env(
      [](const char* name) { return std::getenv(name); }) }
, m_catalog{ std::make_shared<profiler_hub::reader_catalog_t>() }
{
    profiler_hub::storage_t version_probe{ m_file_path, "" };
    const auto              version = version_probe.get_storage_version();
    m_schema_version                = { .major = version.major,
                                        .minor = version.minor,
                                        .patch = version.patch };

    populate_reader_catalog(m_thread_pool, m_connection_pool, *m_catalog);

    initialize_track_list();
    initialize_node_agents();
    initialize_node_processes();
    initilaize_node_info();
}

ph_ctx::~ph_ctx()
{
    std::unordered_set<ph_future*> live;
    {
        std::scoped_lock lock{ m_futures_mutex };
        m_closing = true;
        live.swap(m_live_futures);
    }

    for(auto* future : live)
    {
        std::ignore = future->m_handle.cancel();
        future->m_handle.wait();
        delete future;
    }
}

bool
ph_ctx::register_future(ph_future* future)
{
    std::scoped_lock lock{ m_futures_mutex };
    if(m_closing) return false;
    m_live_futures.insert(future);
    return true;
}

bool
ph_ctx::unregister_future(ph_future* future)
{
    std::scoped_lock lock{ m_futures_mutex };
    return m_live_futures.erase(future) != 0;
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
ph_ctx::core_get_track_events(profiler_hub::common::connection&                   conn,
                              const profiler_hub::reader_types::track_info_ptr_t& track,
                              uint64_t start_ts,
                              uint64_t end_ts)
{
    const auto filter = profiler_hub::make_window_filter(start_ts, end_ts);

    track_events_result_t result;
    result.events = conn.reader().get_events_for_track(track, filter);
    result.c_events.reserve(result.events.size());
    for(const auto& event : result.events)
    {
        result.c_events.push_back(ph_event_t{
            .start = event.start_timestamp,
            .end   = event.end_timestamp,
            .name  = event.display_name.empty() ? "" : event.display_name.data(),
        });
    }

    std::scoped_lock lock{ m_track_results_mutex };
    auto&            stored = m_track_events_results.emplace_back(std::move(result));

    return ph_event_list_t{ .list_size = to_list_size(stored.c_events.size()),
                            .events    = stored.c_events.data() };
}

std::vector<ph_event_t>
ph_ctx::build_thread_track_events(
    profiler_hub::common::connection&                   conn,
    const profiler_hub::reader_types::track_info_ptr_t& track,
    size_t                                              parts)
{
    using profiler_hub::reader_types::event_type_t;

    constexpr std::array types{ event_type_t::region,
                                event_type_t::kernel_dispatch,
                                event_type_t::memory_allocate,
                                event_type_t::memory_copy };

    struct work_item
    {
        event_type_t            type;
        size_t                  begin;
        size_t                  end;
        std::vector<ph_event_t> out;
    };

    std::array<std::optional<std::pair<size_t, size_t>>, types.size()> spans{};
    size_t                                                             total_span = 0;
    for(size_t i = 0; i < types.size(); ++i)
    {
        spans[i] = conn.reader().get_event_id_span(types[i]);
        if(spans[i].has_value())
        {
            total_span += spans[i]->second - spans[i]->first + 1;
        }
    }

    std::vector<work_item> items;
    for(size_t i = 0; i < types.size(); ++i)
    {
        if(!spans[i].has_value()) continue;
        const size_t low   = spans[i]->first;
        const size_t high  = spans[i]->second + 1;
        const size_t share = (high - low) * parts;
        const size_t count = std::max<size_t>(1, (share + total_span / 2) / total_span);
        const size_t step  = (high - low + count - 1) / count;
        for(size_t begin = low; begin < high; begin += step)
        {
            items.push_back({ types[i], begin, std::min(begin + step, high), {} });
        }
    }

    const auto visitor = [](void*                                      context,
                            profiler_hub::reader_types::timestamp_ns_t start,
                            profiler_hub::reader_types::timestamp_ns_t end,
                            std::string_view                           name) {
        static_cast<std::vector<ph_event_t>*>(context)->push_back(ph_event_t{
            .start = start, .end = end, .name = name.empty() ? "" : name.data() });
    };

    std::atomic<size_t> next{ 0 };
    std::atomic<bool>   failed{ false };
    std::mutex          error_mutex;
    std::exception_ptr  first_error;
    const auto          worker = [&](profiler_hub::common::connection& worker_conn) {
        try
        {
            for(size_t i = next.fetch_add(1); i < items.size() && !failed.load();
                i        = next.fetch_add(1))
            {
                auto& item = items[i];
                worker_conn.reader().visit_track_events_in_id_range(
                    track, item.type, item.begin, item.end, visitor, &item.out);
            }
        } catch(...)
        {
            const std::scoped_lock lock{ error_mutex };
            if(!first_error) first_error = std::current_exception();
            failed.store(true);
        }
    };

    const auto helper = [&](const std::stop_token&) {
        if(next.load() >= items.size()) return;
        auto lease = m_connection_pool.try_acquire();
        if(lease.has_value()) worker(**lease);
    };

    struct helper_group
    {
        std::vector<profiler_hub::common::thread_pool::task_handle> handles;

        ~helper_group()
        {
            for(const auto& handle : handles)
            {
                std::ignore = handle.cancel();
                handle.wait();
            }
        }
    };

    {
        helper_group helpers;
        const size_t wanted = std::min(parts, items.size());
        for(size_t i = 1; i < wanted; ++i)
        {
            helpers.handles.push_back(m_thread_pool.submit(helper));
        }
        worker(conn);
    }

    if(first_error) std::rethrow_exception(first_error);

    size_t total = 0;
    for(const auto& item : items)
    {
        total += item.out.size();
    }

    std::vector<ph_event_t> events;
    events.reserve(total);

    std::vector<size_t> table_ends;
    for(size_t i = 0; i < items.size(); ++i)
    {
        events.insert(events.end(), items[i].out.begin(), items[i].out.end());
        std::vector<ph_event_t>().swap(items[i].out);
        if(i + 1 == items.size() || items[i + 1].type != items[i].type)
        {
            table_ends.push_back(events.size());
        }
    }

    const auto by_start = [](const ph_event_t& lhs, const ph_event_t& rhs) {
        return lhs.start < rhs.start;
    };

    size_t table_begin = 0;
    for(const size_t table_end : table_ends)
    {
        profiler_hub::common::natural_merge_sort(
            events.begin() + static_cast<std::ptrdiff_t>(table_begin),
            events.begin() + static_cast<std::ptrdiff_t>(table_end),
            by_start);
        table_begin = table_end;
    }
    for(size_t i = 1; i < table_ends.size(); ++i)
    {
        std::inplace_merge(events.begin(),
                           events.begin() +
                               static_cast<std::ptrdiff_t>(table_ends[i - 1]),
                           events.begin() + static_cast<std::ptrdiff_t>(table_ends[i]),
                           by_start);
    }

    return events;
}

std::vector<ph_event_t>
ph_ctx::build_sorted_track_events(
    profiler_hub::common::connection&                   conn,
    const profiler_hub::reader_types::track_info_ptr_t& track)
{
    if(track->category == profiler_hub::reader_types::track_kind_t::thread &&
       track->event_count >= m_read_options.parallel_read_min_events)
    {
        return build_thread_track_events(conn, track, m_read_options.parallel_read_parts);
    }

    const auto events = conn.reader().get_events_for_track(track, {});

    std::vector<ph_event_t> sorted;
    sorted.reserve(events.size());
    for(const auto& event : events)
    {
        sorted.push_back(ph_event_t{
            .start = event.start_timestamp,
            .end   = event.end_timestamp,
            .name  = event.display_name.empty() ? "" : event.display_name.data(),
        });
    }

    profiler_hub::common::natural_merge_sort(
        sorted.begin(), sorted.end(), [](const ph_event_t& lhs, const ph_event_t& rhs) {
            return lhs.start < rhs.start;
        });

    return sorted;
}

std::vector<ph_sample_t>
ph_ctx::build_sorted_track_samples(
    profiler_hub::common::connection&                   conn,
    const profiler_hub::reader_types::track_info_ptr_t& track)
{
    const auto samples = conn.reader().get_counter_events_for_track(track, {});

    std::vector<ph_sample_t> sorted;
    sorted.reserve(samples.size());
    for(const auto& sample : samples)
    {
        sorted.push_back(
            ph_sample_t{ .timestamp = sample.timestamp, .value = sample.value });
    }

    profiler_hub::common::natural_merge_sort(
        sorted.begin(), sorted.end(), [](const ph_sample_t& lhs, const ph_sample_t& rhs) {
            return lhs.timestamp < rhs.timestamp;
        });

    return sorted;
}

ph_sample_list_t
ph_ctx::get_cached_track_samples(
    const profiler_hub::reader_types::track_info_ptr_t& track)
{
    track_samples_entry* entry = nullptr;
    {
        std::scoped_lock lock{ m_track_cache_mutex };
        auto&            slot = m_track_samples_cache[static_cast<uint32_t>(track->id)];
        if(!slot) slot = std::make_unique<track_samples_entry>();
        entry = slot.get();
    }

    std::call_once(entry->once, [&] {
        entry->samples =
            m_connection_pool.run_sync([&](profiler_hub::common::connection& conn) {
                return build_sorted_track_samples(conn, track);
            });
    });

    return ph_sample_list_t{ .list_size = to_list_size(entry->samples.size()),
                             .samples   = entry->samples.data() };
}

ph_event_list_t
ph_ctx::get_cached_track_events(const profiler_hub::reader_types::track_info_ptr_t& track)
{
    track_events_entry* entry = nullptr;
    {
        std::scoped_lock lock{ m_track_cache_mutex };
        auto&            slot = m_track_events_cache[static_cast<uint32_t>(track->id)];
        if(!slot) slot = std::make_unique<track_events_entry>();
        entry = slot.get();
    }

    std::call_once(entry->once, [&] {
        entry->events =
            m_connection_pool.run_sync([&](profiler_hub::common::connection& conn) {
                return build_sorted_track_events(conn, track);
            });
    });

    return ph_event_list_t{ .list_size = to_list_size(entry->events.size()),
                            .events    = entry->events.data() };
}

ph_sample_list_t
ph_ctx::core_get_track_samples(profiler_hub::common::connection&                   conn,
                               const profiler_hub::reader_types::track_info_ptr_t& track,
                               uint64_t start_ts,
                               uint64_t end_ts)
{
    const auto filter = profiler_hub::make_window_filter(start_ts, end_ts);

    const auto samples = conn.reader().get_counter_events_for_track(track, filter);

    std::vector<ph_sample_t> c_samples;
    c_samples.reserve(samples.size());
    for(const auto& sample : samples)
    {
        c_samples.push_back(
            ph_sample_t{ .timestamp = sample.timestamp, .value = sample.value });
    }

    std::scoped_lock lock{ m_track_results_mutex };
    auto&            stored = m_track_samples_results.emplace_back(std::move(c_samples));

    return ph_sample_list_t{ .list_size = to_list_size(stored.size()),
                             .samples   = stored.data() };
}

ph_event_list_t
ph_ctx::get_track_events(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    LOG_DEBUG("[Profiler-Hub] Get track events. Track id {}, time slice [{} - {}]",
              track_id,
              start_ts,
              end_ts);
    const auto track_it = m_track_by_id.find(track_id);
    if(track_it == m_track_by_id.end() ||
       track_it->second->category == profiler_hub::reader_types::track_kind_t::pmc_agent)
    {
        return ph_event_list_t{ .list_size = 0, .events = nullptr };
    }

    if(start_ts == 0 && end_ts == 0)
    {
        return get_cached_track_events(track_it->second);
    }

    return m_connection_pool.run_sync([&](profiler_hub::common::connection& conn) {
        return core_get_track_events(conn, track_it->second, start_ts, end_ts);
    });
}

ph_sample_list_t
ph_ctx::get_track_samples(uint32_t track_id, uint64_t start_ts, uint64_t end_ts)
{
    LOG_DEBUG("[Profiler-Hub] Get track samples. Track id {}, time slice [{} - {}]",
              track_id,
              start_ts,
              end_ts);
    const auto track_it = m_track_by_id.find(track_id);
    if(track_it == m_track_by_id.end() ||
       track_it->second->category != profiler_hub::reader_types::track_kind_t::pmc_agent)
    {
        return ph_sample_list_t{ .list_size = 0, .samples = nullptr };
    }

    if(start_ts == 0 && end_ts == 0)
    {
        return get_cached_track_samples(track_it->second);
    }

    return m_connection_pool.run_sync([&](profiler_hub::common::connection& conn) {
        return core_get_track_samples(conn, track_it->second, start_ts, end_ts);
    });
}

void
ph_ctx::initialize_track_list()
{
    const auto& all_tracks = m_catalog->tracks;

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
            .event_count = static_cast<std::uint32_t>(std::min<size_t>(
                track->event_count, std::numeric_limits<std::uint32_t>::max())),
            .agent_id    = static_cast<std::uint32_t>(track->agent_id),
            .category    = to_c_track_category(track->category),
            .queue_id    = static_cast<std::uint32_t>(track->queue_id),
            .stream_id   = static_cast<std::uint32_t>(track->stream_id),
            .start_ts    = static_cast<std::uint64_t>(track->start_ts),
            .end_ts      = static_cast<std::uint64_t>(track->end_ts),
        });
    }
}

void
ph_ctx::initilaize_node_info()
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
