#pragma once

#include "common/connection_pool.hpp"
#include "common/thread_pool.hpp"
#include "profiler-hub/c_interface/profiler_hub_types.h"
#include "profiler-hub/reader.hpp"
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ph_future;

struct ph_ctx
{
    explicit ph_ctx(std::string_view trace_path);
    ~ph_ctx();

    ph_ctx(const ph_ctx&)            = delete;
    ph_ctx& operator=(const ph_ctx&) = delete;
    ph_ctx(ph_ctx&&)                 = delete;
    ph_ctx& operator=(ph_ctx&&)      = delete;

    ph_schema_version_t get_storage_version();
    ph_track_list_t     get_track_list();
    ph_node_t           get_node();
    bool has_track(uint32_t track_id) const { return m_track_by_id.contains(track_id); }
    ph_event_list_t  get_track_events(uint32_t track_id,
                                      uint64_t start_ts,
                                      uint64_t end_ts);
    ph_sample_list_t get_track_samples(uint32_t track_id,
                                       uint64_t start_ts,
                                       uint64_t end_ts);

    profiler_hub::common::thread_pool& get_thread_pool() { return m_thread_pool; }

    /** @brief Registers a future issued through this ctx, for cleanup on ~ph_ctx(). */
    void register_future(ph_future* future);
    /** @brief Unregisters a future previously passed to register_future(). */
    void unregister_future(ph_future* future);
    /** @brief Checks a future was issued through this ctx (owned by it). */
    [[nodiscard]] bool owns_future(ph_future* future) const;

private:
    void initialize_track_list();
    void initilaize_node_info();
    void initialize_node_agents();

    ph_event_list_t  core_get_track_events(profiler_hub::common::connection& conn,
                                           uint32_t                          track_id,
                                           uint64_t                          start_ts,
                                           uint64_t                          end_ts);
    ph_sample_list_t core_get_track_samples(profiler_hub::common::connection& conn,
                                            uint32_t                          track_id,
                                            uint64_t                          start_ts,
                                            uint64_t                          end_ts);

    static size_t default_thread_pool_size();

    std::string         m_file_path;
    ph_schema_version_t m_schema_version;

    static constexpr size_t               k_connection_count = 5;
    profiler_hub::common::connection_pool m_connection_pool{ m_file_path,
                                                             k_connection_count };

    profiler_hub::reader_types::track_info_list_t m_tracks;
    std::vector<ph_track_t>                       m_c_tracks;
    std::unordered_map<uint32_t, profiler_hub::reader_types::track_info_ptr_t>
        m_track_by_id;

    profiler_hub::reader_types::agent_info_list_t m_agents;
    std::vector<ph_agent_t>                       m_c_agents;

    std::vector<std::shared_ptr<profiler_hub::reader_types::node_info_t>> m_nodes;
    std::shared_ptr<ph_node_t>                                            m_c_node;

    struct track_events_result_t
    {
        profiler_hub::reader_types::timeline_event_list_t events;
        std::vector<ph_event_t>                           c_events;
    };
    std::mutex                           m_track_results_mutex;
    std::deque<track_events_result_t>    m_track_events_results;
    std::deque<std::vector<ph_sample_t>> m_track_samples_results;

    profiler_hub::common::thread_pool m_thread_pool{ default_thread_pool_size() };

    mutable std::mutex             m_futures_mutex;
    std::unordered_set<ph_future*> m_live_futures;
};
