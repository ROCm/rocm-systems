// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// tsv_processor_adapter_t — processor_t<>-derived wrapper around tsv_processor_t for
// registration with sample_processor_t via processor_view_t::add_handler().
//
// Kept in a separate header from tsv_processor.hpp to isolate AMD-SMI / PMC collector
// includes (sample_processor.hpp transitively requires amd_smi/amdsmi.h). Only include
// this header from translation units that already pay that include cost (e.g.
// cache_manager.cpp).
//
// All non-backtrace sample types are no-ops here; tsv_processor_t only handles
// backtrace_region_sample (timer_sampling category).

#include "core/trace_cache/sample_processor.hpp"
#include "core/trace_cache/tsv_processor.hpp"

namespace rocprofsys
{
namespace trace_cache
{

class tsv_processor_adapter_t : public processor_t<tsv_processor_adapter_t>
{
public:
    explicit tsv_processor_adapter_t(std::string output_dir)
    : m_proc(std::move(output_dir))
    {}

    void prepare_for_processing() { m_proc.prepare_for_processing(); }
    void finalize_processing() { m_proc.finalize_processing(); }

    void handle(const backtrace_region_sample& s) { m_proc.handle(s); }

    void handle(const kernel_dispatch_sample&) noexcept {}
    void handle(const scratch_memory_sample&) noexcept {}
    void handle(const memory_copy_sample&) noexcept {}
    void handle(const memory_allocate_sample&) noexcept {}
    void handle(const region_sample&) noexcept {}
    void handle(const in_time_sample&) noexcept {}
    void handle(const pmc_event_with_sample&) noexcept {}
    void handle(const gpu_pmc_sample&) noexcept {}
    void handle(const ainic_pmc_sample&) noexcept {}
    void handle(const cpu_pmc_sample&) noexcept {}
    void handle(const kfd_sample&) noexcept {}

private:
    tsv_processor_t m_proc;
};

}  // namespace trace_cache
}  // namespace rocprofsys
