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
#include "core/gpu.hpp"
#include "core/common.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"

#include <timemory/components/timing/backends.hpp>

#include <atomic>
#include <memory>

namespace rocprofsys
{
namespace amd_smi
{
#if ROCPROFSYS_USE_ROCM > 0

/// @brief Production implementation of IAmdSmiDriver
/// Wraps actual AMD SMI library calls
class AmdSmiDriverImpl : public IAmdSmiDriver
{
public:
    amdsmi_status_t get_gpu_activity(amdsmi_processor_handle handle,
                                     amdsmi_engine_usage_t*  usage) override
    {
        return amdsmi_get_gpu_activity(handle, usage);
    }

    amdsmi_status_t get_temp_metric(amdsmi_processor_handle     handle,
                                    amdsmi_temperature_type_t   type,
                                    amdsmi_temperature_metric_t metric,
                                    int64_t*                    temp) override
    {
        return amdsmi_get_temp_metric(handle, type, metric, temp);
    }

    amdsmi_status_t get_power_info(amdsmi_processor_handle handle,
                                   amdsmi_power_info_t*    power) override
    {
#if(AMDSMI_LIB_VERSION_MAJOR == 2 && AMDSMI_LIB_VERSION_MINOR == 0) ||                   \
    (AMDSMI_LIB_VERSION_MAJOR == 25 && AMDSMI_LIB_VERSION_MINOR == 2)
        return amdsmi_get_power_info(handle, 0, power);
#else
        return amdsmi_get_power_info(handle, power);
#endif
    }

    amdsmi_status_t get_gpu_memory_usage(amdsmi_processor_handle handle,
                                         amdsmi_memory_type_t    type,
                                         uint64_t*               usage) override
    {
        return amdsmi_get_gpu_memory_usage(handle, type, usage);
    }

    amdsmi_status_t get_gpu_metrics_info(amdsmi_processor_handle handle,
                                         amdsmi_gpu_metrics_t*   metrics) override
    {
        return amdsmi_get_gpu_metrics_info(handle, metrics);
    }

    amdsmi_status_t get_lib_version(amdsmi_version_t* version) override
    {
        return amdsmi_get_lib_version(version);
    }

    amdsmi_status_t shut_down() override { return amdsmi_shut_down(); }

    bool initialize() override { return gpu::initialize_amdsmi(); }

    uint32_t device_count() override { return gpu::device_count(); }

    amdsmi_processor_handle get_handle(uint32_t device_id) override
    {
        return gpu::get_handle_from_id(device_id);
    }

    amdsmi_status_t status_to_string(amdsmi_status_t status, const char** msg) override
    {
        return amdsmi_status_code_to_string(status, msg);
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

/// @brief Production implementation of IClock
/// Uses timemory for real clock values
class SystemClock : public IClock
{
public:
    size_t now_ns() override { return tim::get_clock_real_now<size_t, std::nano>(); }
};

/// @brief Production implementation of ISampleStorage
/// Stores samples in trace cache
class TraceCacheStorage : public ISampleStorage
{
public:
    void store(const trace_cache::amd_smi_sample& sample) override
    {
        trace_cache::get_buffer_storage().store(sample);
    }
};

/// @brief Production implementation of IStateManager
/// Manages actual application state using a reference to external state
class StateManagerImpl : public IStateManager
{
public:
    /// @brief Construct with reference to external state
    explicit StateManagerImpl(std::atomic<State>& state_ref)
    : m_state_ref(state_ref)
    {}

    State get_state() const override { return m_state_ref.load(); }

    void set_state(State s) override { m_state_ref.store(s); }

    bool is_child_process() const override { return rocprofsys::is_child_process(); }

private:
    std::atomic<State>& m_state_ref;
};

#if ROCPROFSYS_USE_ROCM > 0

/// @brief Production implementation of IGpuCapabilities
/// Queries actual GPU capabilities
class GpuCapabilitiesImpl : public IGpuCapabilities
{
public:
    bool vcn_is_device_level_only(uint32_t dev_id) override
    {
        return gpu::vcn_is_device_level_only(dev_id);
    }

    bool jpeg_is_device_level_only(uint32_t dev_id) override
    {
        return gpu::jpeg_is_device_level_only(dev_id);
    }
};

#endif  // ROCPROFSYS_USE_ROCM > 0

/// @brief Get singleton instance of production driver
inline std::shared_ptr<IAmdSmiDriver>
get_default_driver()
{
#if ROCPROFSYS_USE_ROCM > 0
    static auto instance = std::make_shared<AmdSmiDriverImpl>();
    return instance;
#else
    return nullptr;
#endif
}

/// @brief Get singleton instance of production clock
inline std::shared_ptr<IClock>
get_default_clock()
{
    static auto instance = std::make_shared<SystemClock>();
    return instance;
}

/// @brief Get singleton instance of production storage
inline std::shared_ptr<ISampleStorage>
get_default_storage()
{
    static auto instance = std::make_shared<TraceCacheStorage>();
    return instance;
}

/// @brief Get singleton instance of production state manager
/// NOTE: This creates a state manager with its own internal state.
/// For production use in amd_smi.cpp, create StateManagerImpl with
/// the global get_state() reference instead.
inline std::shared_ptr<IStateManager>
get_default_state_manager()
{
    // Default state for testing - in production, pass the global state reference
    static std::atomic<State> default_state{ State::PreInit };
    static auto instance = std::make_shared<StateManagerImpl>(default_state);
    return instance;
}

/// @brief Get singleton instance of production GPU capabilities
inline std::shared_ptr<IGpuCapabilities>
get_default_gpu_capabilities()
{
#if ROCPROFSYS_USE_ROCM > 0
    static auto instance = std::make_shared<GpuCapabilitiesImpl>();
    return instance;
#else
    return nullptr;
#endif
}

}  // namespace amd_smi
}  // namespace rocprofsys
