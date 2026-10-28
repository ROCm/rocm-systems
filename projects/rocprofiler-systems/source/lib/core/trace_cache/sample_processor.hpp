// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/defines.h"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_processor_interface.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <rocprofiler-sdk/version.h>

#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{

struct sample_processor_t
{
    void clear_handlers() { m_processors.clear(); }

    void add_handler(std::shared_ptr<sample_processor_interface> handler)
    {
        m_processors.push_back(std::move(handler));
    }

    template <typename SampleType>
    ROCPROFSYS_INLINE void handle_sample(const SampleType& sample) const
    {
        for(const auto& processor : m_processors)
        {
            processor->handle(sample);
        }
    }

    ROCPROFSYS_INLINE void prepare_for_processing() const noexcept
    {
        for(const auto& processor : m_processors)
        {
            processor->prepare_for_processing();
        }
    }

    ROCPROFSYS_INLINE void finalize_processing() const noexcept
    {
        for(const auto& processor : m_processors)
        {
            processor->finalize_processing();
        }
    }

    [[nodiscard]] ROCPROFSYS_INLINE bool is_empty() const noexcept
    {
        return m_processors.empty();
    }

    ROCPROFSYS_INLINE void execute_sample_processing(
        type_identifier_t type_identifier, const trace_cache::cacheable_t& sample) const
    {
        switch(type_identifier)
        {
            case type_identifier_t::region:
                handle_sample(static_cast<const region_sample&>(sample));
                break;
            case type_identifier_t::kernel_dispatch:
                handle_sample(static_cast<const kernel_dispatch_sample&>(sample));
                break;
            case type_identifier_t::scratch_memory:
                handle_sample(static_cast<const scratch_memory_sample&>(sample));
                break;
            case type_identifier_t::memory_copy:
                handle_sample(static_cast<const memory_copy_sample&>(sample));
                break;
#if ROCPROFILER_VERSION >= 600
            case type_identifier_t::memory_alloc:
                handle_sample(static_cast<const memory_allocate_sample&>(sample));
                break;
#endif
            case type_identifier_t::in_time_sample:
                handle_sample(static_cast<const in_time_sample&>(sample));
                break;
            case type_identifier_t::pmc_event_with_sample:
                handle_sample(static_cast<const pmc_event_with_sample&>(sample));
                break;
            case type_identifier_t::gpu_pmc_sample:
                handle_sample(static_cast<const gpu_pmc_sample&>(sample));
                break;
            case type_identifier_t::ainic_pmc_sample:
                handle_sample(static_cast<const ainic_pmc_sample&>(sample));
                break;
            case type_identifier_t::cpu_pmc_sample:
                handle_sample(static_cast<const cpu_pmc_sample&>(sample));
                break;
            case type_identifier_t::hipfile_pmc_sample:
                handle_sample(static_cast<const hipfile_pmc_sample&>(sample));
                break;
            case type_identifier_t::gpu_perf_counter_sample:
                handle_sample(static_cast<const gpu_perf_counter_sample&>(sample));
                break;
            case type_identifier_t::backtrace_region_sample:
                handle_sample(static_cast<const backtrace_region_sample&>(sample));
                break;
            case type_identifier_t::kfd_sample:
                handle_sample(static_cast<const kfd_sample&>(sample));
                break;
            default: throw std::runtime_error("Unsupported sample type");
        }
    }

private:
    std::vector<std::shared_ptr<sample_processor_interface>> m_processors;
};

}  // namespace rocprofsys::trace_cache
