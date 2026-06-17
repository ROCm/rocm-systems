// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "agent_manager.hpp"
#include "config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"
#include <cstdint>

#include <functional>
#include <memory>
#include <perfetto.h>
#include <unordered_map>

namespace rocprofsys
{
namespace trace_cache
{
using char_vec_t = std::vector<char>;

struct pmc_track_info
{
    const char*                        default_units;
    std::function<bool(std::uint64_t)> exists_fn;
    std::function<void(std::uint64_t, const std::string&, const std::string&)> emplace_fn;
    std::function<void(std::uint64_t, std::uint64_t, std::uint64_t, double)>   trace_fn;
};

class perfetto_processor_t : public processor_t<perfetto_processor_t>
{
public:
    perfetto_processor_t(const std::shared_ptr<metadata_registry>& metadata,
                         const std::shared_ptr<agent_manager>& agent_mngr, int pid,
                         int ppid, output_file_registry& output_registry);

    void prepare_for_processing();
    void finalize_processing();

    void handle(const kernel_dispatch_sample& sample);
    void handle(const scratch_memory_sample& sample);
    void handle(const memory_copy_sample& sample);
    void handle(const memory_allocate_sample& sample);
    void handle(const region_sample& sample);
    void handle(const in_time_sample& sample);
    void handle(const pmc_event_with_sample& sample);
    void handle(const gpu_pmc_sample& sample);
    void handle(const ainic_pmc_sample& sample);
    void handle(const cpu_pmc_sample& sample);
    void handle(const gpu_perf_counter_sample& sample);
    void handle(const backtrace_region_sample& sample);
    void handle(const kfd_sample& sample);

private:
    void       initialize_perfetto();
    void       setup_perfetto();
    void       start_session();
    void       stop_session();
    void       initialize_pmc_track_map();
    void       flush(bool& perfetto_output_error);
    char_vec_t get_session_data();

    // Returns a cached ::perfetto::Track for the given (category, args...) key,
    // calling get_perfetto_track only on the first encounter to avoid the global
    // mutex on every event in high-frequency handle() paths.
    template <typename CategoryT, typename FuncT, typename... Args>
    ::perfetto::Track get_or_create_track(CategoryT, FuncT&& desc_gen, Args&&... args);

    // Returns a cached per-thread ::perfetto::ThreadTrack for a system thread id,
    // created via ThreadTrack::ForThread on first use. Replayed region events are
    // emitted on the originating thread's track instead of the single replay thread.
    // Worker threads (sequent_value > 0) get a "Thread N" descriptor name.
    // The main thread keeps its default descriptor.
    ::perfetto::ThreadTrack get_thread_track(std::uint64_t thread_id);

    template <typename CategoryT>
    void emit_kfd_event(const kfd_sample& sample);
    void handle_kfd_page_fault(const kfd_sample& sample);
    void handle_kfd_page_migrate(const kfd_sample& sample);

    metadata_registry&                          m_metadata;
    std::uint64_t                               m_process_id;
    std::uint64_t                               m_parrent_pid;
    agent_manager&                              m_agent_manager;
    ::perfetto::TraceConfig                     m_session_config;
    std::shared_ptr<tmp_file>                   m_tmp_file{ nullptr };
    std::unique_ptr<::perfetto::TracingSession> m_tracing_session{ nullptr };
    bool                                        m_use_annotations{ false };
    bool                                        m_default_group_by_queue{ true };

    std::unordered_map<size_t, pmc_track_info> m_pmc_track_map;
    // Each perfetto_processor_t instance is owned by a single consumer thread
    // for its entire lifetime (see process_buffered_storage in cache_manager.cpp).
    // No synchronization is required for instance-local state below.
    // Note: m_output_registry is shared across threads; it must be internally
    // thread-safe.
    std::unordered_map<std::uint64_t, ::perfetto::Track>       m_track_cache;
    std::unordered_map<std::uint64_t, ::perfetto::ThreadTrack> m_thread_track_cache;
    std::unordered_map<std::uint32_t, agent_type>              m_kfd_node_type_cache;
    // KFD node_id -> per-type GPU index matching kfd_sample.device_id.
    std::unordered_map<std::uint32_t, std::uint32_t> m_kfd_node_to_gpu_index_cache;
    std::map<std::uint32_t, std::uint64_t>           m_unified_memory_fault_counts;
    output_file_registry&                            m_output_registry;
};
}  // namespace trace_cache
}  // namespace rocprofsys
