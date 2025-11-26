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

#include <cstdint>
#include <memory>

namespace rocprofsys
{
namespace amd_smi
{
#if ROCPROFSYS_USE_ROCM > 0

/// @brief Interface for AMD SMI hardware operations
/// Abstracts all direct AMD SMI API calls to enable mocking in tests
struct IAmdSmiDriver
{
    virtual ~IAmdSmiDriver() = default;

    /// @brief Get GPU activity/utilization metrics
    virtual amdsmi_status_t get_gpu_activity(amdsmi_processor_handle handle,
                                             amdsmi_engine_usage_t*  usage) = 0;

    /// @brief Get temperature metric
    virtual amdsmi_status_t get_temp_metric(amdsmi_processor_handle      handle,
                                            amdsmi_temperature_type_t    type,
                                            amdsmi_temperature_metric_t  metric,
                                            int64_t*                     temp) = 0;

    /// @brief Get power information
    virtual amdsmi_status_t get_power_info(amdsmi_processor_handle handle,
                                           amdsmi_power_info_t*    power) = 0;

    /// @brief Get GPU memory usage
    virtual amdsmi_status_t get_gpu_memory_usage(amdsmi_processor_handle handle,
                                                 amdsmi_memory_type_t    type,
                                                 uint64_t*               usage) = 0;

    /// @brief Get comprehensive GPU metrics
    virtual amdsmi_status_t get_gpu_metrics_info(amdsmi_processor_handle handle,
                                                 amdsmi_gpu_metrics_t*   metrics) = 0;

    /// @brief Get AMD SMI library version
    virtual amdsmi_status_t get_lib_version(amdsmi_version_t* version) = 0;

    /// @brief Shutdown AMD SMI library
    virtual amdsmi_status_t shut_down() = 0;

    /// @brief Initialize AMD SMI library
    virtual bool initialize() = 0;

    /// @brief Get total device count
    virtual uint32_t device_count() = 0;

    /// @brief Get processor handle for a device
    virtual amdsmi_processor_handle get_handle(uint32_t device_id) = 0;

    /// @brief Convert status code to string
    virtual amdsmi_status_t status_to_string(amdsmi_status_t status,
                                             const char**    msg) = 0;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

/// @brief Interface for clock/time operations
/// Enables deterministic time values in tests
struct IClock
{
    virtual ~IClock() = default;

    /// @brief Get current time in nanoseconds
    virtual size_t now_ns() = 0;
};

/// @brief Interface for sample storage operations
/// Abstracts trace cache storage to enable testing without file I/O
struct ISampleStorage
{
    virtual ~ISampleStorage() = default;

    /// @brief Store an AMD SMI sample
    virtual void store(const trace_cache::amd_smi_sample& sample) = 0;
};

/// @brief Interface for state management
/// Enables testing different state scenarios
struct IStateManager
{
    virtual ~IStateManager() = default;

    /// @brief Get current state
    virtual State get_state() const = 0;

    /// @brief Set current state
    virtual void set_state(State s) = 0;

    /// @brief Check if running in child process after fork
    virtual bool is_child_process() const = 0;
};

/// @brief Interface for GPU capabilities queries
/// Abstracts GPU feature detection for testing
struct IGpuCapabilities
{
    virtual ~IGpuCapabilities() = default;

    /// @brief Check if VCN activity is device-level only (vs per-XCP)
    virtual bool vcn_is_device_level_only(uint32_t dev_id) = 0;

    /// @brief Check if JPEG activity is device-level only (vs per-XCP)
    virtual bool jpeg_is_device_level_only(uint32_t dev_id) = 0;
};

/// @brief Factory function type for creating interfaces
/// Enables swapping implementations at runtime for testing
template <typename T>
using interface_factory_t = std::function<std::shared_ptr<T>()>;

}  // namespace amd_smi
}  // namespace rocprofsys

