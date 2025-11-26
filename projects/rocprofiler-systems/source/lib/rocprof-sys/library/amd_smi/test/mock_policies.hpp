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

#include "core/state.hpp"
#include "core/trace_cache/sample_type.hpp"

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

#include <atomic>
#include <cstdint>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{
#if ROCPROFSYS_USE_ROCM > 0

//==============================================================================
// Mock Policy Classes for Testing
// These use static members to allow configuration from tests
//==============================================================================

/// @brief Mock AMD SMI driver policy with configurable behavior
struct MockDriverPolicy
{
    // Configurable mock data (static for policy-based design)
    static inline amdsmi_engine_usage_t mock_usage{};
    static inline int64_t               mock_temp        = 50;
    static inline amdsmi_power_info_t   mock_power{};
    static inline uint64_t              mock_mem_usage   = 1024ULL * 1024ULL * 100ULL;
    static inline amdsmi_gpu_metrics_t  mock_gpu_metrics{};

    // Configurable return status
    static inline amdsmi_status_t activity_status    = AMDSMI_STATUS_SUCCESS;
    static inline amdsmi_status_t temp_status        = AMDSMI_STATUS_SUCCESS;
    static inline amdsmi_status_t power_status       = AMDSMI_STATUS_SUCCESS;
    static inline amdsmi_status_t mem_usage_status   = AMDSMI_STATUS_SUCCESS;
    static inline amdsmi_status_t gpu_metrics_status = AMDSMI_STATUS_SUCCESS;

    // Call tracking
    static inline int get_gpu_activity_calls     = 0;
    static inline int get_temp_metric_calls      = 0;
    static inline int get_power_info_calls       = 0;
    static inline int get_gpu_memory_usage_calls = 0;
    static inline int get_gpu_metrics_info_calls = 0;

    static amdsmi_status_t get_gpu_activity(amdsmi_processor_handle /*handle*/,
                                            amdsmi_engine_usage_t* usage)
    {
        ++get_gpu_activity_calls;
        if(usage && activity_status == AMDSMI_STATUS_SUCCESS) *usage = mock_usage;
        return activity_status;
    }

    static amdsmi_status_t get_temp_metric(amdsmi_processor_handle /*handle*/,
                                           amdsmi_temperature_type_t /*type*/,
                                           amdsmi_temperature_metric_t /*metric*/,
                                           int64_t* temp)
    {
        ++get_temp_metric_calls;
        if(temp && temp_status == AMDSMI_STATUS_SUCCESS) *temp = mock_temp;
        return temp_status;
    }

    static amdsmi_status_t get_power_info(amdsmi_processor_handle /*handle*/,
                                          amdsmi_power_info_t* power)
    {
        ++get_power_info_calls;
        if(power && power_status == AMDSMI_STATUS_SUCCESS) *power = mock_power;
        return power_status;
    }

    static amdsmi_status_t get_gpu_memory_usage(amdsmi_processor_handle /*handle*/,
                                                amdsmi_memory_type_t /*type*/,
                                                uint64_t* usage)
    {
        ++get_gpu_memory_usage_calls;
        if(usage && mem_usage_status == AMDSMI_STATUS_SUCCESS) *usage = mock_mem_usage;
        return mem_usage_status;
    }

    static amdsmi_status_t get_gpu_metrics_info(amdsmi_processor_handle /*handle*/,
                                                amdsmi_gpu_metrics_t* metrics)
    {
        ++get_gpu_metrics_info_calls;
        if(metrics && gpu_metrics_status == AMDSMI_STATUS_SUCCESS)
            *metrics = mock_gpu_metrics;
        return gpu_metrics_status;
    }

    static amdsmi_status_t status_to_string(amdsmi_status_t /*status*/, const char** msg)
    {
        static const char* mock_msg = "mock_error";
        if(msg) *msg = mock_msg;
        return AMDSMI_STATUS_SUCCESS;
    }

    static amdsmi_processor_handle get_handle(uint32_t /*device_id*/)
    {
        static int mock_handle = 0;
        return reinterpret_cast<amdsmi_processor_handle>(&mock_handle);
    }

    /// @brief Reset all mock data to defaults
    static void reset()
    {
        mock_usage       = {};
        mock_temp        = 50;
        mock_power       = {};
        mock_mem_usage   = 1024ULL * 1024ULL * 100ULL;
        mock_gpu_metrics = {};

        activity_status    = AMDSMI_STATUS_SUCCESS;
        temp_status        = AMDSMI_STATUS_SUCCESS;
        power_status       = AMDSMI_STATUS_SUCCESS;
        mem_usage_status   = AMDSMI_STATUS_SUCCESS;
        gpu_metrics_status = AMDSMI_STATUS_SUCCESS;

        reset_call_counts();
    }

    static void reset_call_counts()
    {
        get_gpu_activity_calls     = 0;
        get_temp_metric_calls      = 0;
        get_power_info_calls       = 0;
        get_gpu_memory_usage_calls = 0;
        get_gpu_metrics_info_calls = 0;
    }
};

/// @brief Mock clock policy with configurable time
struct MockClockPolicy
{
    static inline size_t mock_time = 1000000000;  // 1 second in ns

    static size_t now_ns() { return mock_time; }

    static void advance(size_t ns) { mock_time += ns; }
    static void set_time(size_t ns) { mock_time = ns; }
    static void reset() { mock_time = 1000000000; }
};

/// @brief Mock storage policy that captures samples
struct MockStoragePolicy
{
    static inline std::vector<trace_cache::amd_smi_sample> stored_samples;

    static void store(const trace_cache::amd_smi_sample& sample)
    {
        stored_samples.push_back(sample);
    }

    static bool   has_samples() { return !stored_samples.empty(); }
    static size_t sample_count() { return stored_samples.size(); }
    static const trace_cache::amd_smi_sample& last_sample() { return stored_samples.back(); }
    static void                               clear() { stored_samples.clear(); }
};

/// @brief Mock state policy
struct MockStatePolicy
{
    std::atomic<State>& m_state_ref;
    static inline bool  mock_is_child = false;

    explicit MockStatePolicy(std::atomic<State>& state_ref)
    : m_state_ref(state_ref)
    {}

    State get_state() const { return m_state_ref.load(); }
    void  set_state(State s) { m_state_ref.store(s); }
    bool  is_child_process() const { return mock_is_child; }

    static void reset() { mock_is_child = false; }
};

/// @brief Mock GPU capabilities policy
struct MockGpuCapabilitiesPolicy
{
    static inline bool mock_vcn_device_level  = true;
    static inline bool mock_jpeg_device_level = true;

    static bool vcn_is_device_level_only(uint32_t /*dev_id*/)
    {
        return mock_vcn_device_level;
    }

    static bool jpeg_is_device_level_only(uint32_t /*dev_id*/)
    {
        return mock_jpeg_device_level;
    }

    static void reset()
    {
        mock_vcn_device_level  = true;
        mock_jpeg_device_level = true;
    }
};

//==============================================================================
// Mock Configuration
//==============================================================================

/// @brief Mock configuration for testing
struct MockConfig
{
    using Driver          = MockDriverPolicy;
    using Clock           = MockClockPolicy;
    using Storage         = MockStoragePolicy;
    using State           = MockStatePolicy;
    using GpuCapabilities = MockGpuCapabilitiesPolicy;
};

/// @brief Reset all mock policies to defaults
inline void
reset_all_mocks()
{
    MockDriverPolicy::reset();
    MockClockPolicy::reset();
    MockStoragePolicy::clear();
    MockStatePolicy::reset();
    MockGpuCapabilitiesPolicy::reset();
}

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
