// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

/// Collects \ref wall_clock_scope_event_sample during trace-cache replay and writes
/// \c wall_clock_evt.{json,txt} (offline reconstruction of the instrumented wall-clock
/// tree).
class wall_clock_scope_event_processor_t
: public processor_t<wall_clock_scope_event_processor_t>
{
public:
    wall_clock_scope_event_processor_t(int pid, int ppid, output_file_registry& registry);

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
    void handle(const wall_clock_scope_event_sample& sample);

private:
    int                                        m_pid;
    int                                        m_ppid;
    output_file_registry&                      m_registry;
    std::vector<wall_clock_scope_event_sample> m_events;
};

}  // namespace trace_cache
}  // namespace rocprofsys
