// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"
#include "core/trace_cache/sample_processor.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

/// Pass B: reconstruct hierarchical wall-clock statistics from span begin/end samples
/// (Pass A / Option B) and emit a timemory-style text report.
class wall_clock_report_processor_t : public processor_t<wall_clock_report_processor_t>
{
public:
    wall_clock_report_processor_t(int pid, int ppid,
                                  output_file_registry& output_registry);

    void prepare_for_processing();
    void finalize_processing();

    void handle(const kernel_dispatch_sample&) {}
    void handle(const scratch_memory_sample&) {}
    void handle(const memory_copy_sample&) {}
#if(ROCPROFILER_VERSION >= 600)
    void handle(const memory_allocate_sample&) {}
#endif
    void handle(const region_sample&) {}
    void handle(const in_time_sample&) {}
    void handle(const pmc_event_with_sample&) {}
    void handle(const gpu_pmc_sample&) {}
    void handle(const ainic_pmc_sample&) {}
    void handle(const cpu_pmc_sample&) {}
    void handle(const backtrace_region_sample&) {}
    void handle(const kfd_sample&) {}

    void handle(const wall_clock_span_begin_sample& sample);
    void handle(const wall_clock_span_end_sample& sample);

private:
    struct span_begin_state
    {
        std::uint64_t parent_span_id{};
        std::uint64_t thread_seq_id{};
        std::uint64_t path_hash{};
        std::uint64_t t_start_ns{};
        std::string   name;
        std::string   category;
    };

    struct completed_span
    {
        std::uint64_t span_id{};
        std::uint64_t parent_span_id{};
        std::uint64_t thread_seq_id{};
        std::uint64_t path_hash{};
        std::uint64_t t_start_ns{};
        std::uint64_t t_end_ns{};
        std::string   name;
        std::string   category;
    };

    int                                                 m_pid;
    int                                                 m_ppid;
    output_file_registry&                               m_output_registry;
    std::mutex                                          m_mutex;
    std::unordered_map<std::uint64_t, std::uint64_t>    m_path_hash_for_span;
    std::unordered_map<std::uint64_t, span_begin_state> m_open_begin;
    std::vector<completed_span>                         m_completed;

    static std::uint64_t path_hash_combine(std::uint64_t      parent_path_hash,
                                           const std::string& name,
                                           const std::string& category);

    /// Preorder index per span_id (per thread_seq): siblings ordered by begin timestamp.
    static void compute_span_preorder_indices(
        const std::vector<completed_span>&                completed,
        std::unordered_map<std::uint64_t, std::uint64_t>& span_preorder_out);

    void write_report(const std::string& path) const;
};

}  // namespace trace_cache
}  // namespace rocprofsys
