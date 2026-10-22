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
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct ph_future;

struct ph_ctx
{
    /** Rejects an unsupported database before any reader is opened. The trace
     *  catalog is still read by load() or load_async(). */
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
    [[nodiscard]] const std::shared_ptr<profiler_hub::reader_catalog_t>& catalog() const
    {
        return m_catalog;
    }

    [[nodiscard]] ph_schema_version_t get_schema_version();
    [[nodiscard]] ph_track_list_t     get_track_list();
    [[nodiscard]] ph_node_t           get_node();
    bool has_track(uint32_t track_id) const { return m_track_by_id.contains(track_id); }
    [[nodiscard]] ph_event_list_t                get_track_events(uint32_t track_id,
                                                                  uint64_t start_ts,
                                                                  uint64_t end_ts);
    [[nodiscard]] std::optional<ph_event_list_t> get_correlated_events(
        uint64_t        event_id,
        ph_event_type_t event_type);
    [[nodiscard]] ph_call_stack_t              get_call_stack(uint64_t        event_id,
                                                              ph_event_type_t event_type);
    [[nodiscard]] ph_region_details_t          get_region_details(uint64_t event_id);
    [[nodiscard]] ph_kernel_dispatch_details_t get_kernel_dispatch_details(
        uint64_t event_id);
    [[nodiscard]] ph_memory_copy_details_t  get_memory_copy_details(uint64_t event_id);
    [[nodiscard]] ph_memory_alloc_details_t get_memory_alloc_details(uint64_t event_id);
    [[nodiscard]] std::optional<uint32_t>  track_id_of(uint64_t        event_id,
                                                       ph_event_type_t event_type);
    [[nodiscard]] ph_sample_list_t          get_track_samples(uint32_t track_id,
                                                              uint64_t start_ts,
                                                              uint64_t end_ts);
    void build_histogram(uint32_t max_buckets, ph_histogram_t& histogram);

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
    using progress_fn       = std::function<void(double, ph_progress_description_t)>;

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
    void               initialize_node_threads();
    void               initialize_node_streams();
    void               initialize_node_queues();
    void               initialize_node_pmc_infos();
    void               initialize_stream_hardware_links();

    struct accepted_trace_t
    {
        std::string         path;
        ph_schema_version_t version;
    };

    explicit ph_ctx(accepted_trace_t accepted);
    [[nodiscard]] static accepted_trace_t accept_trace(std::string_view trace_path);

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

    profiler_hub::reader_types::thread_info_list_t m_threads;
    std::vector<ph_thread_t>                       m_c_threads;

    profiler_hub::reader_types::stream_info_list_t m_streams;
    std::vector<ph_stream_t>                       m_c_streams;

    profiler_hub::reader_types::queue_info_list_t m_queues;
    std::vector<ph_queue_t>                       m_c_queues;

    profiler_hub::reader_types::pmc_info_list_t m_pmc_infos;
    std::vector<ph_pmc_info_t>                  m_c_pmc_infos;

    std::vector<ph_stream_hardware_link_t> m_c_stream_hardware_links;

    struct c_histogram_storage_t
    {
        ph_histogram_t                                  histogram{};
        std::vector<ph_track_histogram_t>               tracks;
        std::vector<std::vector<ph_histogram_bucket_t>> buckets;
    };

    std::mutex                                          m_histogram_mutex;
    std::vector<std::unique_ptr<c_histogram_storage_t>> m_histograms;

    std::mutex                          m_correlated_events_mutex;
    std::deque<std::vector<ph_event_t>> m_correlated_events;

    struct call_stack_hold
    {
        profiler_hub::reader_types::call_stack_t data;
        std::vector<ph_stack_frame_t>            frames;
    };

    std::mutex                                   m_call_stack_mutex;
    std::deque<std::unique_ptr<call_stack_hold>> m_call_stacks;

    struct region_detail_hold
    {
        profiler_hub::reader_types::region_data_t data;
        std::vector<ph_arg_data_t>                args;
        ph_region_details_t                       view{};
    };
    struct kernel_detail_hold
    {
        profiler_hub::reader_types::kernel_dispatch_data_t data;
        ph_kernel_dispatch_details_t                       view{};
    };
    struct copy_detail_hold
    {
        profiler_hub::reader_types::memory_copy_data_t data;
        ph_memory_copy_details_t                       view{};
    };
    struct alloc_detail_hold
    {
        profiler_hub::reader_types::memory_alloc_data_t data;
        ph_memory_alloc_details_t                       view{};
    };

    std::mutex                                      m_detail_mutex;
    std::deque<std::unique_ptr<region_detail_hold>> m_region_details;
    std::deque<std::unique_ptr<kernel_detail_hold>> m_kernel_details;
    std::deque<std::unique_ptr<copy_detail_hold>>   m_copy_details;
    std::deque<std::unique_ptr<alloc_detail_hold>>  m_alloc_details;

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
