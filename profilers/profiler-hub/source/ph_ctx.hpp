#pragma once

#include "cached_track_reader.hpp"
#include "event_query.hpp"
#include "common/connection_pool.hpp"
#include "histogram_grid.hpp"
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

namespace profiler_hub
{
class track_file_store;
}

struct ph_ctx
{
    /** Rejects an unsupported database before any reader is opened. The trace
     *  catalog is still read by load() or load_async(). */
    explicit ph_ctx(std::string_view trace_path);
    explicit ph_ctx(std::vector<std::string> trace_paths);
    ph_ctx(std::vector<std::string> trace_paths, std::string cache_directory);
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
     * ph_result_t. @p future must have been attached by the caller.
     * PH_RESULT_SUCCESS ends it as PH_FUTURE_FINISHED. PH_RESULT_CANCELLED ends
     * it as PH_FUTURE_CANCELLED, as does cancellation before the operation
     * starts or destruction of the ctx. Any other result ends it as
     * PH_FUTURE_ERROR. An exception ends it as PH_FUTURE_ERROR with
     * PH_RESULT_INTERNAL_ERROR, so the future does not stay unfinished.
     */
    void run_async(std::shared_ptr<ph_future>   future,
                   std::function<ph_result_t()> operation);

    /**
     * Blocks until the trace is read.
     * @return PH_RESULT_SUCCESS once it is, PH_RESULT_CANCELLED if the load was
     *         cancelled, PH_RESULT_INVALID_CONTEXT if it failed.
     */
    [[nodiscard]] ph_result_t wait_until_ready();
    [[nodiscard]] uint32_t    trace_source_count() const;
    [[nodiscard]] const char* trace_source_path(uint32_t index) const;
    [[nodiscard]] const char* trace_source_guid(uint32_t index) const;
    [[nodiscard]] const std::shared_ptr<profiler_hub::reader_catalog_t>& catalog() const;
    [[nodiscard]] const std::shared_ptr<profiler_hub::reader_catalog_t>& catalog_at(
        uint32_t index) const;

    [[nodiscard]] uint32_t read_table_events(
        const profiler_hub::resolved_event_query& query,
        uint32_t                                  event_type,
        uint64_t                                  after_id,
        uint64_t                                  last_id,
        ph_table_event*                           rows,
        uint32_t                                  capacity);
    [[nodiscard]] bool summarize_name_durations(
        const profiler_hub::resolved_event_query&           query,
        uint32_t                                            event_type,
        std::vector<profiler_hub::name_duration_summary_t>& rows);
    [[nodiscard]] bool table_event_id_span(const profiler_hub::resolved_event_query& query,
                                           uint32_t event_type,
                                           uint64_t& first_id,
                                           uint64_t& last_id);

    [[nodiscard]] ph_schema_version_t get_schema_version();
    [[nodiscard]] ph_track_list_t     get_track_list();
    [[nodiscard]] ph_node_t           get_node();
    [[nodiscard]] ph_node_t           source_node(uint32_t index) const;
    bool has_track(uint32_t track_id) const { return m_track_by_id.contains(track_id); }
    [[nodiscard]] ph_event_list_t get_track_events(uint32_t track_id,
                                                   uint64_t start_ts,
                                                   uint64_t end_ts);
    void release_track_events(ph_event_list_t* events);
    void release_track_samples(ph_sample_list_t* samples);
    [[nodiscard]] std::optional<ph_event_list_t> get_correlated_events(
        uint32_t        track_id,
        uint64_t        event_id,
        ph_event_type_t event_type);
    [[nodiscard]] ph_call_stack_t              get_call_stack(uint32_t        track_id,
                                                              uint64_t        event_id,
                                                              ph_event_type_t event_type);
    [[nodiscard]] ph_region_details_t          get_region_details(uint32_t track_id,
                                                                  uint64_t event_id);
    [[nodiscard]] ph_kernel_dispatch_details_t get_kernel_dispatch_details(
        uint32_t track_id,
        uint64_t event_id);
    [[nodiscard]] ph_memory_copy_details_t  get_memory_copy_details(uint32_t track_id,
                                                                    uint64_t event_id);
    [[nodiscard]] ph_memory_alloc_details_t get_memory_alloc_details(uint32_t track_id,
                                                                     uint64_t event_id);
    [[nodiscard]] std::optional<uint32_t>  track_id_of(uint32_t        track_id,
                                                       uint64_t        event_id,
                                                       ph_event_type_t event_type);
    [[nodiscard]] ph_sample_list_t          get_track_samples(uint32_t track_id,
                                                              uint64_t start_ts,
                                                              uint64_t end_ts);
    void build_histogram(uint32_t max_buckets, ph_histogram_t& histogram);

    /** Writes every loaded source into one trimmed database. Times are absolute
     *  on source 0's clock; each other source keeps the same displayed range. */
    [[nodiscard]] ph_result_t trim_trace(uint64_t    start_ts,
                                         uint64_t    end_ts,
                                         const char* output_path,
                                         ph_future*  future);

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
    void               prepare_track_store();
    void               attach_track_store();
    [[nodiscard]] profiler_hub::histogram_grid_t histogram_grid(size_t max_buckets) const;
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
    [[nodiscard]] const char* keep_event_name(std::string_view text);

    struct accepted_source_t
    {
        std::string                                path;
        std::string                                uuid;
        ph_schema_version_t                        version{};
        profiler_hub::reader_types::trace_schema_t schema{
            profiler_hub::reader_types::trace_schema_t::unknown };
    };

    struct placed_track_t
    {
        uint32_t                                         source_index{};
        profiler_hub::reader_types::track_info_ptr_t     track;
    };

    struct trace_source_t
    {
        std::string                                         path;
        std::string                                         uuid;
        std::shared_ptr<profiler_hub::reader_catalog_t>     catalog;
        std::unique_ptr<profiler_hub::common::connection_pool> pool;
        std::unique_ptr<profiler_hub::pooled_connection_source> connections;
        std::unique_ptr<profiler_hub::cached_track_reader>  reader;
        uint32_t                                            track_offset{};
        uint32_t                                            track_count{};
        std::unordered_map<uint32_t, uint32_t>              global_of_local;
        profiler_hub::reader_types::agent_info_list_t       agents;
        std::vector<ph_agent_t>                             c_agents;
        profiler_hub::reader_types::process_info_list_t     processes;
        std::vector<ph_process_t>                           c_processes;
        profiler_hub::reader_types::thread_info_list_t      threads;
        std::vector<ph_thread_t>                            c_threads;
        profiler_hub::reader_types::stream_info_list_t      streams;
        std::vector<ph_stream_t>                            c_streams;
        profiler_hub::reader_types::queue_info_list_t       queues;
        std::vector<ph_queue_t>                             c_queues;
        profiler_hub::reader_types::pmc_info_list_t         pmc_infos;
        std::vector<ph_pmc_info_t>                          c_pmc_infos;
        std::vector<ph_stream_hardware_link_t>              c_stream_hardware_links;
        std::vector<std::shared_ptr<profiler_hub::reader_types::node_info_t>> nodes;
        ph_node_t                                           node{};
    };

    ph_ctx(std::vector<accepted_source_t> accepted, std::string cache_directory);
    [[nodiscard]] static std::vector<accepted_source_t> accept_traces(
        const std::vector<std::string>& trace_paths);

    static size_t default_thread_pool_size();
    static size_t default_connection_count();

    [[nodiscard]] trace_source_t& source_at(uint32_t index);
    [[nodiscard]] const trace_source_t& source_at(uint32_t index) const;
    [[nodiscard]] const placed_track_t* place_of(uint32_t track_id) const;
    // The file that holds this event. A known track_id whose file contains
    // the id selects that file. Otherwise the files are searched by id.
    [[nodiscard]] trace_source_t* source_for_event(uint32_t        track_id,
                                                   uint64_t        event_id,
                                                   ph_event_type_t event_type);
    void open_sources();
    [[nodiscard]] ph_optional_uint64_t global_track_id(
        const trace_source_t&                            source,
        const profiler_hub::reader_types::track_info_ptr_t& track) const;

    std::vector<accepted_source_t>   m_accepted;
    std::string                      m_cache_directory;
    std::unique_ptr<profiler_hub::track_file_store> m_track_store;
    std::vector<std::unique_ptr<trace_source_t>> m_sources;
    std::string                      m_file_path;
    profiler_hub::track_read_options m_read_options;
    ph_schema_version_t              m_schema_version;
    profiler_hub::reader_types::trace_schema_t m_trace_schema{
        profiler_hub::reader_types::trace_schema_t::unknown };

    std::vector<ph_track_t> m_c_tracks;
    std::unordered_map<uint32_t, placed_track_t> m_track_by_id;

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
    std::mutex                          m_event_name_mutex;
    std::deque<std::string>             m_event_names;

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

    profiler_hub::pending_futures     m_operations;
    profiler_hub::common::thread_pool m_thread_pool{ default_thread_pool_size() };

    std::mutex              m_state_mutex;
    std::condition_variable m_state_cv;
    load_state_t            m_state{ load_state_t::loading };

    std::optional<profiler_hub::common::thread_pool::task_handle> m_load_task;
    std::shared_ptr<ph_future>                                    m_load_future;
};
