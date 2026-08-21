// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "agent_manager.hpp"
#include "core/node_info.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"

#include "trace_cache/sample_type.hpp"

#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

namespace rocprofsys
{
namespace trace_cache
{

class rocpd_processor_t : public processor_t<rocpd_processor_t>
{
public:
    rocpd_processor_t(const std::shared_ptr<metadata_registry>& metadata,
                      const std::shared_ptr<agent_manager>& agent_mngr, int pid, int ppid,
                      output_file_registry& output_registry);

    void prepare_for_processing();
    void finalize_processing();

    void handle(const kernel_dispatch_sample& sample);
    void handle(const scratch_memory_sample& sample);
    void handle(const memory_copy_sample& sample);
    void handle(const memory_allocate_sample& sample);
    void handle(const region_sample& sample);
    void handle(const in_time_sample& sample);
    void handle(const pmc_event_with_sample& sample);
    void handle(const backtrace_region_sample& sample);
    void handle(const gpu_pmc_sample& sample);
    void handle(const ainic_pmc_sample& sample);
    void handle(const cpu_pmc_sample& sample);
    void handle(const gpu_perf_counter_sample& sample);
    void handle(const kfd_sample& sample);

private:
    void post_process_metadata();

    std::shared_ptr<metadata_registry>      m_metadata;
    std::shared_ptr<agent_manager>          m_agent_manager;
    std::unique_ptr<profiler_hub::writer_t> m_writer;
    output_file_registry&                   m_output_registry;
    std::string                             m_db_output_path;
};

}  // namespace trace_cache
}  // namespace rocprofsys
