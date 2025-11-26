// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/gpu.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <timemory/components/timing/backends.hpp>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

#include <atomic>
#include <cstdint>

namespace rocprofsys
{
namespace amd_smi
{
#if ROCPROFSYS_USE_ROCM > 0

//==============================================================================
// Production Policy Classes - No virtual functions, fully inlineable
//==============================================================================

/// @brief Production AMD SMI driver policy - direct API calls
struct AmdSmiDriverPolicy
{
    static amdsmi_status_t get_gpu_activity(amdsmi_processor_handle handle,
                                            amdsmi_engine_usage_t*  usage)
    {
        return amdsmi_get_gpu_activity(handle, usage);
    }

    static amdsmi_status_t get_temp_metric(amdsmi_processor_handle     handle,
                                           amdsmi_temperature_type_t   type,
                                           amdsmi_temperature_metric_t metric,
                                           int64_t*                    temp)
    {
        return amdsmi_get_temp_metric(handle, type, metric, temp);
    }

    static amdsmi_status_t get_power_info(amdsmi_processor_handle handle,
                                          amdsmi_power_info_t*    power)
    {
#    if(AMDSMI_LIB_VERSION_MAJOR == 2 && AMDSMI_LIB_VERSION_MINOR == 0) ||               \
        (AMDSMI_LIB_VERSION_MAJOR == 25 && AMDSMI_LIB_VERSION_MINOR == 2)
        return amdsmi_get_power_info(handle, 0, power);
#    else
        return amdsmi_get_power_info(handle, power);
#    endif
    }

    static amdsmi_status_t get_gpu_memory_usage(amdsmi_processor_handle handle,
                                                amdsmi_memory_type_t    type,
                                                uint64_t*               usage)
    {
        return amdsmi_get_gpu_memory_usage(handle, type, usage);
    }

    static amdsmi_status_t get_gpu_metrics_info(amdsmi_processor_handle handle,
                                                amdsmi_gpu_metrics_t*   metrics)
    {
        return amdsmi_get_gpu_metrics_info(handle, metrics);
    }

    static amdsmi_status_t status_to_string(amdsmi_status_t status, const char** msg)
    {
        return amdsmi_status_code_to_string(status, msg);
    }

    static amdsmi_processor_handle get_handle(uint32_t device_id)
    {
        return gpu::get_handle_from_id(device_id);
    }
};

/// @brief Production clock policy - real time
struct ClockPolicy
{
    static size_t now_ns() { return tim::get_clock_real_now<size_t, std::nano>(); }
};

/// @brief Production storage policy - trace cache
struct StoragePolicy
{
    static void store(const trace_cache::amd_smi_sample& sample)
    {
        trace_cache::get_buffer_storage().store(sample);
    }
};

/// @brief Production state policy - uses external state reference
struct StatePolicy
{
    std::atomic<State>& m_state_ref;

    explicit StatePolicy(std::atomic<State>& state_ref)
    : m_state_ref(state_ref)
    {}

    State       get_state() const { return m_state_ref.load(); }
    void        set_state(State s) { m_state_ref.store(s); }
    static bool is_child_process() { return rocprofsys::is_child_process(); }
};

/// @brief Production GPU capabilities policy - queries real GPU
struct GpuCapabilitiesPolicy
{
    static bool vcn_is_device_level_only(uint32_t dev_id)
    {
        return gpu::vcn_is_device_level_only(dev_id);
    }

    static bool jpeg_is_device_level_only(uint32_t dev_id)
    {
        return gpu::jpeg_is_device_level_only(dev_id);
    }
};

//==============================================================================
// Production Configuration - bundles all policies together
//==============================================================================

/// @brief Production configuration with all real implementations
struct ProductionConfig
{
    using Driver          = AmdSmiDriverPolicy;
    using Clock           = ClockPolicy;
    using Storage         = StoragePolicy;
    using State           = StatePolicy;
    using GpuCapabilities = GpuCapabilitiesPolicy;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys
