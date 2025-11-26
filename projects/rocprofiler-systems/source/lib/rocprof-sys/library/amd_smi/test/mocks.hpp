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

#include "library/amd_smi/interfaces.hpp"

#include <cstring>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{
#if ROCPROFSYS_USE_ROCM > 0

/// @brief Mock implementation of IAmdSmiDriver for unit testing
/// Allows configuring return values and simulating various scenarios
class MockAmdSmiDriver : public IAmdSmiDriver
{
public:
    // Configurable mock data
    amdsmi_engine_usage_t mock_usage{};
    int64_t               mock_temp        = 50;
    amdsmi_power_info_t   mock_power{};
    uint64_t              mock_mem_usage   = 1024ULL * 1024ULL * 100ULL;  // 100 MB
    amdsmi_gpu_metrics_t  mock_gpu_metrics{};
    amdsmi_version_t      mock_version{ 1, 0, 0, "mock" };
    uint32_t              mock_device_count = 1;

    // Configurable return status (for simulating errors)
    amdsmi_status_t activity_status    = AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t temp_status        = AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t power_status       = AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t mem_usage_status   = AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t gpu_metrics_status = AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t version_status     = AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t shutdown_status    = AMDSMI_STATUS_SUCCESS;

    // Call tracking for verification
    int get_gpu_activity_calls    = 0;
    int get_temp_metric_calls     = 0;
    int get_power_info_calls      = 0;
    int get_gpu_memory_usage_calls = 0;
    int get_gpu_metrics_info_calls = 0;

    amdsmi_status_t get_gpu_activity(amdsmi_processor_handle /*handle*/,
                                     amdsmi_engine_usage_t* usage) override
    {
        ++get_gpu_activity_calls;
        if(usage && activity_status == AMDSMI_STATUS_SUCCESS)
            *usage = mock_usage;
        return activity_status;
    }

    amdsmi_status_t get_temp_metric(amdsmi_processor_handle /*handle*/,
                                    amdsmi_temperature_type_t /*type*/,
                                    amdsmi_temperature_metric_t /*metric*/,
                                    int64_t* temp) override
    {
        ++get_temp_metric_calls;
        if(temp && temp_status == AMDSMI_STATUS_SUCCESS)
            *temp = mock_temp;
        return temp_status;
    }

    amdsmi_status_t get_power_info(amdsmi_processor_handle /*handle*/,
                                   amdsmi_power_info_t* power) override
    {
        ++get_power_info_calls;
        if(power && power_status == AMDSMI_STATUS_SUCCESS)
            *power = mock_power;
        return power_status;
    }

    amdsmi_status_t get_gpu_memory_usage(amdsmi_processor_handle /*handle*/,
                                         amdsmi_memory_type_t /*type*/,
                                         uint64_t* usage) override
    {
        ++get_gpu_memory_usage_calls;
        if(usage && mem_usage_status == AMDSMI_STATUS_SUCCESS)
            *usage = mock_mem_usage;
        return mem_usage_status;
    }

    amdsmi_status_t get_gpu_metrics_info(amdsmi_processor_handle /*handle*/,
                                         amdsmi_gpu_metrics_t* metrics) override
    {
        ++get_gpu_metrics_info_calls;
        if(metrics && gpu_metrics_status == AMDSMI_STATUS_SUCCESS)
            *metrics = mock_gpu_metrics;
        return gpu_metrics_status;
    }

    amdsmi_status_t get_lib_version(amdsmi_version_t* version) override
    {
        if(version && version_status == AMDSMI_STATUS_SUCCESS)
            *version = mock_version;
        return version_status;
    }

    amdsmi_status_t shut_down() override { return shutdown_status; }

    bool initialize() override { return true; }

    uint32_t device_count() override { return mock_device_count; }

    amdsmi_processor_handle get_handle(uint32_t /*device_id*/) override
    {
        // Return a mock handle (just a non-null pointer for testing)
        static int mock_handle = 0;
        return reinterpret_cast<amdsmi_processor_handle>(&mock_handle);
    }

    amdsmi_status_t status_to_string(amdsmi_status_t status, const char** msg) override
    {
        static const char* mock_msg = "mock_error";
        if(msg)
            *msg = mock_msg;
        (void) status;
        return AMDSMI_STATUS_SUCCESS;
    }

    /// @brief Reset all call counters
    void reset_call_counts()
    {
        get_gpu_activity_calls     = 0;
        get_temp_metric_calls      = 0;
        get_power_info_calls       = 0;
        get_gpu_memory_usage_calls = 0;
        get_gpu_metrics_info_calls = 0;
    }

    /// @brief Reset all mock data to defaults
    void reset()
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
};

#endif  // ROCPROFSYS_USE_ROCM > 0

/// @brief Mock implementation of IClock for deterministic testing
class MockClock : public IClock
{
public:
    size_t mock_time = 1000000000;  // 1 second in ns

    size_t now_ns() override { return mock_time; }

    /// @brief Advance time by specified nanoseconds
    void advance(size_t ns) { mock_time += ns; }

    /// @brief Set time to specific value
    void set_time(size_t ns) { mock_time = ns; }

    /// @brief Reset to default
    void reset() { mock_time = 1000000000; }
};

/// @brief Mock implementation of ISampleStorage for testing
/// Captures all stored samples for verification
class MockSampleStorage : public ISampleStorage
{
public:
    std::vector<trace_cache::amd_smi_sample> stored_samples;

    void store(const trace_cache::amd_smi_sample& sample) override
    {
        stored_samples.push_back(sample);
    }

    /// @brief Check if any samples were stored
    bool has_samples() const { return !stored_samples.empty(); }

    /// @brief Get number of stored samples
    size_t sample_count() const { return stored_samples.size(); }

    /// @brief Get last stored sample (throws if empty)
    const trace_cache::amd_smi_sample& last_sample() const { return stored_samples.back(); }

    /// @brief Clear all stored samples
    void clear() { stored_samples.clear(); }
};

/// @brief Mock implementation of IStateManager for testing
class MockStateManager : public IStateManager
{
public:
    State mock_state    = State::Active;
    bool  mock_is_child = false;

    State get_state() const override { return mock_state; }

    void set_state(State s) override { mock_state = s; }

    bool is_child_process() const override { return mock_is_child; }

    /// @brief Reset to defaults
    void reset()
    {
        mock_state    = State::Active;
        mock_is_child = false;
    }
};

/// @brief Mock implementation of IGpuCapabilities for testing
class MockGpuCapabilities : public IGpuCapabilities
{
public:
    bool mock_vcn_device_level  = true;
    bool mock_jpeg_device_level = true;

    bool vcn_is_device_level_only(uint32_t /*dev_id*/) override
    {
        return mock_vcn_device_level;
    }

    bool jpeg_is_device_level_only(uint32_t /*dev_id*/) override
    {
        return mock_jpeg_device_level;
    }

    /// @brief Reset to defaults
    void reset()
    {
        mock_vcn_device_level  = true;
        mock_jpeg_device_level = true;
    }
};

/// @brief Test fixture helper that creates all mocks
struct MockFixture
{
#if ROCPROFSYS_USE_ROCM > 0
    std::shared_ptr<MockAmdSmiDriver>   driver   = std::make_shared<MockAmdSmiDriver>();
#endif
    std::shared_ptr<MockClock>          clock    = std::make_shared<MockClock>();
    std::shared_ptr<MockSampleStorage>  storage  = std::make_shared<MockSampleStorage>();
    std::shared_ptr<MockStateManager>   state    = std::make_shared<MockStateManager>();
    std::shared_ptr<MockGpuCapabilities> gpu_caps = std::make_shared<MockGpuCapabilities>();

    /// @brief Reset all mocks to default state
    void reset_all()
    {
#if ROCPROFSYS_USE_ROCM > 0
        driver->reset();
#endif
        clock->reset();
        storage->clear();
        state->reset();
        gpu_caps->reset();
    }
};

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
