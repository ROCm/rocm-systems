// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_processor_interface.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/wall_clock_report.hpp"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
struct agent_manager;
}

namespace rocprofsys::trace_cache
{
struct metadata_registry;

namespace data
{
struct processor_config_t;
}

/// Where the report goes; an output without a path is off.
struct wall_clock_options
{
    wall_clock_settings        settings;
    std::optional<std::string> text_path;
    std::optional<std::string> json_path;
    bool                       should_print_to_stdout{};

    [[nodiscard]] bool has_output() const noexcept
    {
        return text_path.has_value() || json_path.has_value() || should_print_to_stdout;
    }
};

/// Reads the wall_clock settings of this run and names the files of `process`.
[[nodiscard]] wall_clock_options
make_wall_clock_options(const data::processor_config_t& process);

/**
 * Builds the wall_clock report of one process from its cached samples and writes it when
 * the process is finalized.
 *
 * @throws std::runtime_error From handle() when a kernel dispatch names a kernel that
 * has no registered symbol, and from handle(memory_copy_sample) when an agent is
 * unknown.
 */
class wall_clock_processor_t : public sample_processor_interface
{
public:
    /**
     * @param metadata Source of kernel symbols used to name kernel dispatch rows.
     * @param agents   Names the devices of memory copies; may be null when no memory
     *                 copy is handled.
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
    template <typename Sample>
    void add_region(const Sample& sample, std::string name);

    template <typename Sample>
    void add_gpu_activity(const Sample& sample, std::string name);

    const std::string& kernel_name(std::uint64_t kernel_id);

    void add_thread_lifetimes();

    std::shared_ptr<metadata_registry>             m_metadata;
    pid_t                                          m_pid;
    std::shared_ptr<agent_manager>                 m_agents;
    wall_clock_options                             m_options;
    wall_clock_report                              m_report;
    std::unordered_map<std::uint64_t, std::string> m_kernel_names;
};

}  // namespace rocprofsys::trace_cache
