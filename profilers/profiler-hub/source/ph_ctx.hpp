#pragma once

#include "cached_track_reader.hpp"
#include "common/connection_pool.hpp"
#include "common/thread_pool.hpp"
#include "pooled_connection_source.hpp"
#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader.hpp"
#include "track_read_options.hpp"
#include <cstdint>
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

    [[nodiscard]] ph_schema_version_t get_schema_version();
    [[nodiscard]] ph_track_list_t     get_track_list();
    [[nodiscard]] ph_node_t           get_node();
    bool has_track(uint32_t track_id) const { return m_track_by_id.contains(track_id); }
    [[nodiscard]] ph_event_list_t  get_track_events(uint32_t track_id,
                                                    uint64_t start_ts,
                                                    uint64_t end_ts);
    [[nodiscard]] uint32_t         get_track_nesting_depth(uint32_t track_id);
    [[nodiscard]] ph_sample_list_t get_track_samples(uint32_t track_id,
                                                     uint64_t start_ts,
                                                     uint64_t end_ts);

    profiler_hub::common::thread_pool& get_thread_pool() { return m_thread_pool; }

    /**
     * @brief Registers a future issued through this ctx, for cleanup on ~ph_ctx().
     * @return false if the ctx is already shutting down and the future was not
     *         registered.
     */
    [[nodiscard]] bool register_future(ph_future* future);
    /**
     * @brief Unregisters a future previously passed to register_future().
     * @return false if the future was not registered (never issued, already
     *         freed, or taken over by ~ph_ctx()).
     */
    [[nodiscard]] bool unregister_future(ph_future* future);
    /** @brief Checks a future was issued through this ctx (owned by it). */
    [[nodiscard]] bool owns_future(ph_future* future) const;

private:
    void initialize_track_list();
    void initialize_node_info();
    void initialize_node_agents();
    void initialize_node_processes();

    static size_t default_thread_pool_size();
    static size_t default_connection_count();

    std::string                      m_file_path;
    profiler_hub::track_read_options m_read_options;
    ph_schema_version_t              m_schema_version;

    std::shared_ptr<profiler_hub::reader_catalog_t> m_catalog;
    profiler_hub::common::connection_pool           m_connection_pool{ m_file_path,
                                                             default_connection_count(),
                                                             m_catalog };
    profiler_hub::pooled_connection_source m_connection_source{ m_connection_pool };

    std::vector<ph_track_t> m_c_tracks;
    std::unordered_map<uint32_t, profiler_hub::reader_types::track_info_ptr_t>
        m_track_by_id;

    profiler_hub::reader_types::agent_info_list_t m_agents;
    std::vector<ph_agent_t>                       m_c_agents;

    profiler_hub::reader_types::process_info_list_t m_processes;
    std::vector<ph_process_t>                       m_c_processes;

    std::vector<std::shared_ptr<profiler_hub::reader_types::node_info_t>> m_nodes;
    std::shared_ptr<ph_node_t>                                            m_c_node;

    mutable std::mutex             m_futures_mutex;
    std::unordered_set<ph_future*> m_live_futures;
    bool                           m_closing{ false };

    profiler_hub::common::thread_pool m_thread_pool{ default_thread_pool_size() };
    profiler_hub::cached_track_reader m_track_reader{ m_connection_source,
                                                      m_thread_pool,
                                                      m_read_options };
};
