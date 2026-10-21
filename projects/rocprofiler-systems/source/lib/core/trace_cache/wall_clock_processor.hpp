// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_processor_interface.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/wall_clock_report.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/gpu_perf_counter/sample.hpp"
#include "library/pmc/collectors/hipfile/sample.hpp"
#include "library/pmc/collectors/nic/sample.hpp"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
struct agent_manager;
}

namespace rocprofsys::trace_cache
{
struct metadata_registry;

struct wall_clock_options
{
    wall_clock_layout layout;
    bool              text     = true;
    bool              json     = true;
    bool              cout     = false;
    std::int32_t      rank     = 0;
    std::int32_t      mpi_size = 1;
    std::string       suffix;
};

class wall_clock_processor_t : public sample_processor_interface
{
public:
    /**
     * @param metadata Source of kernel symbols used to name kernel dispatch rows.
     * @param agents   Names the devices of memory copies.
     * @param pid      Process the report belongs to; part of the output file name.
     * @param options  Layout and files of the report.
     */
    wall_clock_processor_t(std::shared_ptr<metadata_registry> metadata,
                           std::shared_ptr<agent_manager> agents, pid_t pid,
                           wall_clock_options options);

    void prepare_for_processing() override {}

    void finalize_processing() override;

    void handle(const region_sample& sample) override;

    void handle(const kernel_dispatch_sample& sample) override;

    void handle(const in_time_sample&) override {}
    void handle(const pmc_event_with_sample&) override {}
    void handle(const memory_copy_sample& sample) override;
    void handle(const memory_allocate_sample&) override {}
    void handle(const scratch_memory_sample& sample) override;
    void handle(const gpu_pmc_sample&) override {}
    void handle(const ainic_pmc_sample&) override {}
    void handle(const cpu_pmc_sample&) override {}
    void handle(const gpu_perf_counter_sample&) override {}
    void handle(const hipfile_pmc_sample&) override {}
    void handle(const backtrace_region_sample&) override {}
    void handle(const kfd_sample&) override {}

private:
    void add_thread_lifetimes();
    void write_file(std::string_view type, const auto& write);

    std::shared_ptr<metadata_registry> m_metadata;
    pid_t                              m_pid;
    std::shared_ptr<agent_manager>     m_agents;
    wall_clock_options                 m_options;
    std::vector<wall_clock_interval>   m_regions;
    std::vector<wall_clock_interval>   m_gpu_activity;
};

}  // namespace rocprofsys::trace_cache
