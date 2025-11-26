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

#include "core/debug.hpp"
#include "core/gpu_metrics.hpp"
#include "library/amd_smi.hpp"
#include "library/amd_smi/metrics.hpp"
#include "library/amd_smi/policies.hpp"

#include <cassert>
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
    int64_t               temp      = 0;
    amdsmi_power_info_t   power{};
    uint64_t              mem_usage = 0;
    bool                  success   = false;
};

/// @brief Template-based Sampler with compile-time policy injection
/// No virtual functions - all calls are resolved at compile time and can be inlined
/// @tparam Config Configuration struct containing policy type aliases
template <typename Config>
class SamplerImpl
{
public:
    using Driver          = typename Config::Driver;
    using Clock           = typename Config::Clock;
    using Storage         = typename Config::Storage;
    using StateType       = typename Config::State;
    using GpuCapabilities = typename Config::GpuCapabilities;

    /// @brief Construct with state reference
    explicit SamplerImpl(std::atomic<State>& state_ref)
    : m_state(state_ref)
    {}

    /// @brief Sample metrics from a specific device
    /// @param device_id Device ID to sample
    /// @param device_settings Settings for this device
    /// @return Collected data if successful, nullopt if sampling was skipped
    std::optional<data> sample(uint32_t device_id, const settings& device_settings)
    {
        // Early exit checks
        if(m_state.is_child_process()) return std::nullopt;
        if(m_state.get_state() != State::Active) return std::nullopt;

        // Skip devices that have been disabled due to errors
        if(is_device_disabled(device_id)) return std::nullopt;

        const auto timestamp = Clock::now_ns();
        assert(timestamp < static_cast<size_t>(std::numeric_limits<int64_t>::max()));

        // Initialize result data
        data result{};
        result.m_dev_id = device_id;
        result.m_ts     = static_cast<int64_t>(timestamp);

        auto* handle = Driver::get_handle(device_id);

        // Collect basic metrics
        auto basic = collect_basic_metrics(device_id, handle, device_settings);
        if(!basic.success) return std::nullopt;

        result.m_busy_perc = basic.busy_perc;
        result.m_temp      = basic.temp;
        result.m_power     = basic.power;
        result.m_mem_usage = basic.mem_usage;

        const bool basic_metrics_enabled = device_settings.busy || device_settings.temp ||
                                           device_settings.power || device_settings.mem_usage;

        const bool advanced_metrics_needed =
            device_settings.vcn_activity || device_settings.jpeg_activity ||
            device_settings.xgmi || device_settings.pcie;

        // Fetch raw GPU metrics if any advanced metrics are enabled
        amdsmi_gpu_metrics_t raw_gpu_metrics = {};
        if(advanced_metrics_needed)
        {
            auto status = Driver::get_gpu_metrics_info(handle, &raw_gpu_metrics);
            bool needed = advanced_metrics_needed;
            if(!check_status(device_id, status, &needed, "amdsmi_get_gpu_metrics_info"))
                return std::nullopt;
        }

        // Process advanced GPU metrics
        gpu::gpu_metrics_capabilities_t capabilities      = {};
        bool                            has_advanced_data = false;
        metrics::ProcessedMetrics vcn_result{}, jpeg_result{}, xgmi_result{}, pcie_result{};

        if(advanced_metrics_needed)
        {
            capabilities.flags.vcn_is_device_level_only =
                GpuCapabilities::vcn_is_device_level_only(device_id);
            capabilities.flags.jpeg_is_device_level_only =
                GpuCapabilities::jpeg_is_device_level_only(device_id);

            if(device_settings.vcn_activity)
            {
                vcn_result = metrics::process_vcn_metrics(
                    raw_gpu_metrics, capabilities.flags.vcn_is_device_level_only);
                has_advanced_data |= vcn_result.has_data;
            }

            if(device_settings.jpeg_activity)
            {
                jpeg_result = metrics::process_jpeg_metrics(
                    raw_gpu_metrics, capabilities.flags.jpeg_is_device_level_only);
                has_advanced_data |= jpeg_result.has_data;
            }

            if(device_settings.xgmi)
            {
                xgmi_result = metrics::process_xgmi_metrics(raw_gpu_metrics);
                has_advanced_data |= xgmi_result.has_data;
            }

            if(device_settings.pcie)
            {
                pcie_result = metrics::process_pcie_metrics(raw_gpu_metrics);
                has_advanced_data |= pcie_result.has_data;
            }
        }

        // Store sample if we have any data worth recording
        if(basic_metrics_enabled || has_advanced_data)
        {
            auto merged = metrics::merge_processed_metrics(vcn_result, jpeg_result,
                                                           xgmi_result, pcie_result);

            Storage::store(trace_cache::amd_smi_sample{
                metrics::serialize_settings(device_settings), device_id, timestamp,
                result.m_busy_perc.gfx_activity, result.m_busy_perc.umc_activity,
                result.m_busy_perc.mm_activity, result.m_power.current_socket_power,
                result.m_temp, result.m_mem_usage,
                metrics::serialize_gpu_metrics(merged.metrics, capabilities,
                                               device_settings) });

            if(has_advanced_data) result.m_gpu_metrics.push_back(merged.metrics);
        }

        return result;
    }

    /// @brief Collect basic metrics only
    BasicMetricsResult collect_basic_metrics(uint32_t                device_id,
                                             amdsmi_processor_handle handle,
                                             const settings&         s)
    {
        BasicMetricsResult result{};
        settings           local_settings = s;

        if(local_settings.busy)
        {
            auto status = Driver::get_gpu_activity(handle, &result.busy_perc);
            if(!check_status(device_id, status, &local_settings.busy,
                             "amdsmi_get_gpu_activity"))
                return result;
        }

        if(local_settings.temp)
        {
            auto status = Driver::get_temp_metric(handle, AMDSMI_TEMPERATURE_TYPE_JUNCTION,
                                                  AMDSMI_TEMP_CURRENT, &result.temp);
            if(!check_status(device_id, status, &local_settings.temp,
                             "amdsmi_get_temp_metric"))
                return result;
        }

        if(local_settings.power)
        {
            auto status = Driver::get_power_info(handle, &result.power);
            if(!check_status(device_id, status, &local_settings.power,
                             "amdsmi_get_power_info"))
                return result;
        }

        if(local_settings.mem_usage)
        {
            auto status = Driver::get_gpu_memory_usage(handle, AMDSMI_MEM_TYPE_VRAM,
                                                       &result.mem_usage);
            if(!check_status(device_id, status, &local_settings.mem_usage,
                             "amdsmi_get_gpu_memory_usage"))
                return result;
        }

        result.success = true;
        return result;
    }

    /// @brief Check AMD SMI call result and optionally disable feature or device
    bool check_status(uint32_t device_id, amdsmi_status_t status, bool* option,
                      const char* function_name)
    {
        if(status == AMDSMI_STATUS_SUCCESS) return true;

        if(status == AMDSMI_STATUS_NOT_SUPPORTED && option)
        {
            *option = false;
            return true;
        }

        const char* msg = nullptr;
        Driver::status_to_string(status, &msg);
        ROCPROFSYS_VERBOSE_F(
            0,
            "[%s] Device %u: Error code %i :: %s. Disabling future samples for this GPU.\n",
            function_name, device_id, static_cast<int>(status), msg ? msg : "unknown");
        disable_device(device_id);
        return false;
    }

    /// @brief Check if a specific device is disabled due to errors
    bool is_device_disabled(uint32_t device_id) const
    {
        std::lock_guard<std::mutex> lock(m_disabled_mutex);
        return m_disabled_devices.count(device_id) > 0;
    }

    /// @brief Disable sampling for a specific device
    void disable_device(uint32_t device_id)
    {
        std::lock_guard<std::mutex> lock(m_disabled_mutex);
        m_disabled_devices.insert(device_id);
    }

    /// @brief Get the set of disabled device IDs (for testing)
    const std::set<uint32_t>& disabled_devices() const { return m_disabled_devices; }

private:
    StateType          m_state;
    std::set<uint32_t> m_disabled_devices;
    mutable std::mutex m_disabled_mutex;
};

//==============================================================================
// Type alias for production use
//==============================================================================

/// @brief Production sampler with all real implementations
using Sampler = SamplerImpl<ProductionConfig>;

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys
