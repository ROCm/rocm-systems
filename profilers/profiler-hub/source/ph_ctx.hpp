#pragma once

#include "cached_track_reader.hpp"
#include "common/connection_pool.hpp"
#include "common/thread_pool.hpp"
#include "pending_futures.hpp"
#include "pooled_connection_source.hpp"
#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader.hpp"
#include "track_read_options.hpp"
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string_view>
#include <unordered_map>
#include <vector>

struct ph_future;

struct ph_ctx
{
    /** Opens nothing yet: the trace is read by load() or load_async(). */
    explicit ph_ctx(std::string_view trace_path);
    ~ph_ctx();

    ph_ctx(const ph_ctx&)            = delete;
    ph_ctx& operator=(const ph_ctx&) = delete;
    ph_ctx(ph_ctx&&)                 = delete;
    ph_ctx& operator=(ph_ctx&&)      = delete;

    /** Reads the trace on the calling thread. Throws if it cannot be read. */
    void load();

    /**
     * Reads the trace on a worker of the ctx pool and reports to @p future, which must
     * have been attached by the caller. The future ends with PH_FUTURE_CANCELLED if it is
     * cancelled or the ctx is destroyed meanwhile.
     */
    void load_async(std::shared_ptr<ph_future> future);

    /**
     * Runs @p operation on a worker of the ctx pool and ends @p future with its
     * ph_result_t. @p future must have been attached by the caller. It ends with
     * PH_FUTURE_CANCELLED if it is cancelled before the operation starts or the ctx is
     * destroyed first.
     */
    void run_async(std::shared_ptr<ph_future>   future,
                   std::function<ph_result_t()> operation);

    /**
     * Blocks until the trace is read.
     * @return PH_RESULT_SUCCESS once it is, PH_RESULT_CANCELLED if the load was
     *         cancelled, PH_RESULT_INVALID_CONTEXT if it failed.
     */
    [[nodiscard]] ph_result_t wait_until_ready();

    [[nodiscard]] ph_schema_version_t get_schema_version();
    [[nodiscard]] ph_track_list_t     get_track_list();
    [[nodiscard]] ph_node_t           get_node();
    bool has_track(uint32_t track_id) const { return m_track_by_id.contains(track_id); }
    [[nodiscard]] ph_event_list_t  get_track_events(uint32_t track_id,
                                                    uint64_t start_ts,
                                                    uint64_t end_ts);
    [[nodiscard]] ph_sample_list_t get_track_samples(uint32_t track_id,
                                                     uint64_t start_ts,
                                                     uint64_t end_ts);

    profiler_hub::common::thread_pool& get_thread_pool() { return m_thread_pool; }

private:
    enum class load_state_t : uint8_t
    {
        loading,
        ready,
        failed,
        cancelled,
    };

    using stop_requested_fn = std::function<bool()>;
    using progress_fn       = std::function<void(double)>;

    /** @return false if @p stop_requested became true before the trace was read. */
    [[nodiscard]] bool read_trace(const stop_requested_fn& stop_requested,
                                  const progress_fn&       progress);
    void               set_load_state(load_state_t state);
    void               set_load_state_if_loading(load_state_t state);

    void               initialize_track_list();
    [[nodiscard]] bool load_all_tracks(const stop_requested_fn& stop_requested,
                                       const progress_fn&       progress);
    void               initialize_node_info();
    void               initialize_node_agents();
    void               initialize_node_processes();

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

    profiler_hub::pending_futures     m_operations;
    profiler_hub::common::thread_pool m_thread_pool{ default_thread_pool_size() };
    profiler_hub::cached_track_reader m_track_reader{ m_connection_source,
                                                      m_thread_pool,
                                                      m_read_options };

    std::mutex              m_state_mutex;
    std::condition_variable m_state_cv;
    load_state_t            m_state{ load_state_t::loading };

    std::optional<profiler_hub::common::thread_pool::task_handle> m_load_task;
    std::shared_ptr<ph_future>                                    m_load_future;
};
