// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_processor_interface.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/wall_clock_report.hpp"

#include <sys/types.h>

#include <memory>
#include <optional>
#include <string>

namespace rocprofsys
{
struct agent_manager;
}

namespace rocprofsys::trace_cache
{
struct metadata_registry;

struct wall_clock_options
{
    wall_clock_settings        settings;
    std::optional<std::string> text_path;
    std::optional<std::string> json_path;
    bool                       cout{};
};

class wall_clock_processor_t : public sample_processor_interface
{
public:
    /**
     * @param metadata Source of kernel symbols used to name kernel dispatch rows.
     * @param agents   Names the devices of memory copies.
     * @param pid      Process the report belongs to.
     * @param options  Settings and outputs of the report.
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

    std::shared_ptr<metadata_registry> m_metadata;
    pid_t                              m_pid;
    std::shared_ptr<agent_manager>     m_agents;
    wall_clock_options                 m_options;
    wall_clock_report                  m_report;
};

}  // namespace rocprofsys::trace_cache
