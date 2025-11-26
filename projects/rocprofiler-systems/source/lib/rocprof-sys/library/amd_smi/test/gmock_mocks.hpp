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

#include "library/amd_smi/test/gmock_interfaces.hpp"

#include <gmock/gmock.h>

#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
{
#if ROCPROFSYS_USE_ROCM > 0

/// @brief GMock mock for AMD SMI driver
class MockDriver : public IDriver
{
public:
    MOCK_METHOD(amdsmi_status_t, get_gpu_activity,
                (amdsmi_processor_handle handle, amdsmi_engine_usage_t* usage), (override));

    MOCK_METHOD(amdsmi_status_t, get_temp_metric,
                (amdsmi_processor_handle handle, amdsmi_temperature_type_t type,
                 amdsmi_temperature_metric_t metric, int64_t* temp),
                (override));

    MOCK_METHOD(amdsmi_status_t, get_power_info,
                (amdsmi_processor_handle handle, amdsmi_power_info_t* power), (override));

    MOCK_METHOD(amdsmi_status_t, get_gpu_memory_usage,
                (amdsmi_processor_handle handle, amdsmi_memory_type_t type,
                 uint64_t* usage),
                (override));

    MOCK_METHOD(amdsmi_status_t, get_gpu_metrics_info,
                (amdsmi_processor_handle handle, amdsmi_gpu_metrics_t* metrics),
                (override));

    MOCK_METHOD(amdsmi_status_t, status_to_string,
                (amdsmi_status_t status, const char** msg), (override));

    MOCK_METHOD(amdsmi_processor_handle, get_handle, (uint32_t device_id), (override));

    /// @brief Set up default success behavior for all methods
    void SetUpDefaults()
    {
        using ::testing::_;
        using ::testing::DoAll;
        using ::testing::Return;
        using ::testing::SetArgPointee;

        ON_CALL(*this, get_gpu_activity(_, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_temp_metric(_, _, _, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_power_info(_, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_gpu_memory_usage(_, _, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_gpu_metrics_info(_, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, status_to_string(_, _))
            .WillByDefault(Return(AMDSMI_STATUS_SUCCESS));
        ON_CALL(*this, get_handle(_))
            .WillByDefault(Return(reinterpret_cast<amdsmi_processor_handle>(&mock_handle)));
    }

private:
    int mock_handle = 0;
};

/// @brief GMock mock for clock
class MockClock : public IClock
{
public:
    MOCK_METHOD(size_t, now_ns, (), (override));

    void SetUpDefaults(size_t default_time = 1000000000)
    {
        ON_CALL(*this, now_ns()).WillByDefault(::testing::Return(default_time));
    }
};

/// @brief GMock mock for storage - also captures samples for verification
class MockStorage : public IStorage
{
public:
    MOCK_METHOD(void, store, (const trace_cache::amd_smi_sample& sample), (override));

    /// @brief Set up to capture stored samples
    void SetUpCapture()
    {
        using ::testing::_;

        // Actions can be implicitly constructed from callables (no Invoke() needed)
        ON_CALL(*this, store(_)).WillByDefault([this](const auto& sample) {
            stored_samples.push_back(sample);
        });
    }

    bool has_samples() const { return !stored_samples.empty(); }
    size_t sample_count() const { return stored_samples.size(); }
    const trace_cache::amd_smi_sample& last_sample() const { return stored_samples.back(); }
    void clear() { stored_samples.clear(); }

    std::vector<trace_cache::amd_smi_sample> stored_samples;
};

/// @brief GMock mock for state management
class MockState : public IState
{
public:
    MOCK_METHOD(State, get_state, (), (const, override));
    MOCK_METHOD(void, set_state, (State s), (override));
    MOCK_METHOD(bool, is_child_process, (), (const, override));

    void SetUpDefaults(State default_state = State::Active)
    {
        using ::testing::Return;

        ON_CALL(*this, get_state()).WillByDefault(Return(default_state));
        ON_CALL(*this, is_child_process()).WillByDefault(Return(false));
    }
};

/// @brief GMock mock for GPU capabilities
class MockGpuCapabilities : public IGpuCapabilities
{
public:
    MOCK_METHOD(bool, vcn_is_device_level_only, (uint32_t dev_id), (override));
    MOCK_METHOD(bool, jpeg_is_device_level_only, (uint32_t dev_id), (override));

    void SetUpDefaults()
    {
        using ::testing::_;
        using ::testing::Return;

        ON_CALL(*this, vcn_is_device_level_only(_)).WillByDefault(Return(true));
        ON_CALL(*this, jpeg_is_device_level_only(_)).WillByDefault(Return(true));
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
