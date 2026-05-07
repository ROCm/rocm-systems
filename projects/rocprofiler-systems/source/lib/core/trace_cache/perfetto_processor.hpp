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
    void handle(const backtrace_region_sample& sample);
    void handle(const kfd_sample& sample);

private:
    void       initialize_perfetto();
    void       setup_perfetto();
    void       start_session();
    void       stop_session();
    void       flush(bool& perfetto_output_error);
    char_vec_t get_session_data();

    metadata_registry&                          m_metadata;
    std::uint64_t                               m_process_id;
    std::uint64_t                               m_parrent_pid;
    agent_manager&                              m_agent_manager;
    ::perfetto::TraceConfig                     m_session_config;
    std::shared_ptr<tmp_file>                   m_tmp_file{ nullptr };
    std::unique_ptr<::perfetto::TracingSession> m_tracing_session{ nullptr };
    bool                                        m_use_annotations{ false };

    std::unordered_map<size_t, pmc_track_info> m_pmc_track_map;
    output_file_registry&                      m_output_registry;
};
}  // namespace trace_cache
}  // namespace rocprofsys
