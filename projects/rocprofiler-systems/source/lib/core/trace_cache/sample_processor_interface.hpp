// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "core/trace_cache/sample_type.hpp"

#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/gpu/sample.hpp"
#include "library/pmc/collectors/gpu_perf_counter/sample.hpp"
#include "library/pmc/collectors/hipfile/sample.hpp"
#include "library/pmc/collectors/nic/sample.hpp"

namespace rocprofsys::trace_cache
{

/**
 * Consumer of cached samples. One handle() overload per sample type, plus
 * prepare/finalize hooks bracketing a parse pass.
 */
class sample_processor_interface
{
public:
    virtual ~sample_processor_interface() = default;

    virtual void handle(const kernel_dispatch_sample& sample)  = 0;
    virtual void handle(const scratch_memory_sample& sample)   = 0;
    virtual void handle(const memory_copy_sample& sample)      = 0;
    virtual void handle(const memory_allocate_sample& sample)  = 0;
    virtual void handle(const region_sample& sample)           = 0;
    virtual void handle(const in_time_sample& sample)          = 0;
    virtual void handle(const pmc_event_with_sample& sample)   = 0;
    virtual void handle(const gpu_pmc_sample& sample)          = 0;
    virtual void handle(const ainic_pmc_sample& sample)        = 0;
    virtual void handle(const cpu_pmc_sample& sample)          = 0;
    virtual void handle(const gpu_perf_counter_sample& sample) = 0;
    virtual void handle(const hipfile_pmc_sample& sample)      = 0;
    virtual void handle(const backtrace_region_sample& sample) = 0;
    virtual void handle(const kfd_sample& sample)              = 0;

    virtual void prepare_for_processing() = 0;
    virtual void finalize_processing()    = 0;

protected:
    sample_processor_interface()                                             = default;
    sample_processor_interface(const sample_processor_interface&)            = default;
    sample_processor_interface& operator=(const sample_processor_interface&) = default;
};

}  // namespace rocprofsys::trace_cache
