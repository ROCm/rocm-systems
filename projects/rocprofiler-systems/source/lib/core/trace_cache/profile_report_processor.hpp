// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"

#include <rocprofiler-sdk/version.h>

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>

namespace rocprofsys
{
namespace trace_cache
{

class profile_report_processor_t : public processor_t<profile_report_processor_t>
{
public:
    profile_report_processor_t(int pid, int ppid, output_file_registry& output_registry);

    void prepare_for_processing();
    void finalize_processing();

    void handle(const kernel_dispatch_sample& sample);
    void handle(const scratch_memory_sample& sample);
    void handle(const memory_copy_sample& sample);
#if(ROCPROFILER_VERSION >= 600)
    void handle(const memory_allocate_sample& sample);
#endif
    void handle(const region_sample& sample);
    void handle(const in_time_sample& sample);
    void handle(const pmc_event_with_sample& sample);
    void handle(const gpu_pmc_sample& sample);
    void handle(const ainic_pmc_sample& sample);
    void handle(const cpu_pmc_sample& sample);
    void handle(const backtrace_region_sample& sample);
    void handle(const kfd_sample& sample);
    void handle(const wall_clock_event_sample& sample);

private:
    struct aggregate_row
    {
        uint64_t count              = 0;
        uint64_t total_nsec         = 0;
        uint64_t sum_exclusive_nsec = 0;
        uint64_t min_nsec           = std::numeric_limits<uint64_t>::max();
        uint64_t max_nsec           = 0;
        double   sum_sq_sec         = 0.0;
    };

    using aggregate_key_t =
        std::tuple<std::string, wall_clock_event_source, uint64_t, uint64_t, uint32_t>;

    int                                      m_pid;
    int                                      m_ppid;
    output_file_registry&                    m_output_registry;
    std::map<aggregate_key_t, aggregate_row> m_aggregates;
};

}  // namespace trace_cache
}  // namespace rocprofsys
