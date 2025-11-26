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

#include "core/gpu_metrics.hpp"
#include "library/amd_smi.hpp"
#include "library/amd_smi/interfaces.hpp"
#include "library/amd_smi/metrics.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <set>

namespace rocprofsys
{
namespace amd_smi
{
#if ROCPROFSYS_USE_ROCM > 0

/// @brief Result of collecting basic GPU metrics
struct BasicMetricsResult
{
    amdsmi_engine_usage_t busy_perc{};
    int64_t               temp = 0;
    amdsmi_power_info_t   power{};
    uint64_t              mem_usage = 0;
    bool                  success   = false;
};

/// @brief Sampler class with dependency injection for testability
/// Encapsulates all GPU sampling logic with injectable dependencies
class Sampler
{
public:
    /// @brief Construct sampler with all dependencies
    /// @param driver AMD SMI driver interface
    /// @param clock Clock interface for timestamps
    /// @param storage Sample storage interface
    /// @param state_mgr State manager interface
    /// @param gpu_caps GPU capabilities interface
    Sampler(std::shared_ptr<IAmdSmiDriver> driver, std::shared_ptr<IClock> clock,
            std::shared_ptr<ISampleStorage>   storage,
            std::shared_ptr<IStateManager>    state_mgr,
            std::shared_ptr<IGpuCapabilities> gpu_caps);

    /// @brief Default constructor using production implementations
    Sampler();

    /// @brief Sample metrics from a specific device
    /// @param device_id Device ID to sample
    /// @param device_settings Settings for this device
    /// @return Collected data if successful, nullopt if sampling was skipped
    std::optional<data> sample(uint32_t device_id, const settings& device_settings);

    /// @brief Collect basic metrics only (for unit testing individual components)
    /// @param device_id Device ID
    /// @param handle Processor handle
    /// @param s Device settings
    /// @return Basic metrics result
    BasicMetricsResult collect_basic_metrics(uint32_t                device_id,
                                             amdsmi_processor_handle handle,
                                             const settings&         s);

    /// @brief Check AMD SMI call result and optionally disable feature or device
    /// @param device_id Device ID for per-device error tracking
    /// @param status AMD SMI status code
    /// @param option Pointer to option flag to disable on NOT_SUPPORTED
    /// @param function_name Name of function for error messages
    /// @return true if successful, false if error (device will be disabled)
    bool check_status(uint32_t device_id, amdsmi_status_t status, bool* option,
                      const char* function_name);

    /// @brief Check if a specific device is disabled due to errors
    /// @param device_id Device ID to check
    /// @return true if device is disabled
    bool is_device_disabled(uint32_t device_id) const;

    /// @brief Disable sampling for a specific device
    /// @param device_id Device ID to disable
    void disable_device(uint32_t device_id);

    /// @brief Get the set of disabled device IDs (for testing)
    const std::set<uint32_t>& disabled_devices() const { return m_disabled_devices; }

    // Accessors for testing
    std::shared_ptr<IAmdSmiDriver>    driver() const { return m_driver; }
    std::shared_ptr<IClock>           clock() const { return m_clock; }
    std::shared_ptr<ISampleStorage>   storage() const { return m_storage; }
    std::shared_ptr<IStateManager>    state_manager() const { return m_state_mgr; }
    std::shared_ptr<IGpuCapabilities> gpu_capabilities() const { return m_gpu_caps; }

private:
    std::shared_ptr<IAmdSmiDriver>    m_driver;
    std::shared_ptr<IClock>           m_clock;
    std::shared_ptr<ISampleStorage>   m_storage;
    std::shared_ptr<IStateManager>    m_state_mgr;
    std::shared_ptr<IGpuCapabilities> m_gpu_caps;
    std::set<uint32_t>                m_disabled_devices;  ///< Devices with errors
    mutable std::mutex                m_disabled_mutex;    ///< Protects m_disabled_devices
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys
