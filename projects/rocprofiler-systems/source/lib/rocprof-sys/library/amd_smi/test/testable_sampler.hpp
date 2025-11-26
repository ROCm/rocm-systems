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

/// @file testable_sampler.hpp
/// @brief A sampler variant that uses GMock interfaces for testing.
/// This provides full GMock capabilities (EXPECT_CALL, verification, etc.)
/// while the production Sampler uses zero-overhead static policies.

#include "core/gpu_metrics.hpp"
#include "library/amd_smi.hpp"
#include "library/amd_smi/metrics.hpp"
#include "library/amd_smi/test/gmock_interfaces.hpp"

#include <iostream>

#include <cassert>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>

namespace rocprofsys
{
namespace amd_smi
{
namespace testing
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

/// @brief Testable sampler using GMock interfaces
/// This class uses virtual interfaces for full GMock support in tests.
/// Production code uses SamplerImpl<ProductionConfig> with zero overhead.
class TestableSampler
{
public:
    TestableSampler(std::shared_ptr<IDriver>          driver,
                    std::shared_ptr<IClock>           clock,
                    std::shared_ptr<IStorage>         storage,
                    std::shared_ptr<IState>           state,
                    std::shared_ptr<IGpuCapabilities> gpu_caps)
    : m_driver(std::move(driver))
    , m_clock(std::move(clock))
    , m_storage(std::move(storage))
    , m_state(std::move(state))
    , m_gpu_caps(std::move(gpu_caps))
    {}

    std::optional<data> sample(uint32_t device_id, const settings& device_settings)
    {
        // Early exit checks
        if(m_state->is_child_process()) return std::nullopt;
        if(m_state->get_state() != State::Active) return std::nullopt;

        // Skip devices that have been disabled due to errors
        if(is_device_disabled(device_id)) return std::nullopt;

        const auto timestamp = m_clock->now_ns();
        assert(timestamp < static_cast<size_t>(std::numeric_limits<int64_t>::max()));

        // Initialize result data
        data result{};
        result.m_dev_id = device_id;
        result.m_ts     = static_cast<int64_t>(timestamp);

        auto* handle = m_driver->get_handle(device_id);

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
            auto status = m_driver->get_gpu_metrics_info(handle, &raw_gpu_metrics);
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
                m_gpu_caps->vcn_is_device_level_only(device_id);
            capabilities.flags.jpeg_is_device_level_only =
                m_gpu_caps->jpeg_is_device_level_only(device_id);

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

            m_storage->store(trace_cache::amd_smi_sample{
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

    BasicMetricsResult collect_basic_metrics(uint32_t                device_id,
                                             amdsmi_processor_handle handle,
                                             const settings&         s)
    {
        BasicMetricsResult result{};
        settings           local_settings = s;

        if(local_settings.busy)
        {
            auto status = m_driver->get_gpu_activity(handle, &result.busy_perc);
            if(!check_status(device_id, status, &local_settings.busy,
                             "amdsmi_get_gpu_activity"))
                return result;
        }

        if(local_settings.temp)
        {
            auto status = m_driver->get_temp_metric(handle, AMDSMI_TEMPERATURE_TYPE_JUNCTION,
                                                    AMDSMI_TEMP_CURRENT, &result.temp);
            if(!check_status(device_id, status, &local_settings.temp,
                             "amdsmi_get_temp_metric"))
                return result;
        }

        if(local_settings.power)
        {
            auto status = m_driver->get_power_info(handle, &result.power);
            if(!check_status(device_id, status, &local_settings.power,
                             "amdsmi_get_power_info"))
                return result;
        }

        if(local_settings.mem_usage)
        {
            auto status = m_driver->get_gpu_memory_usage(handle, AMDSMI_MEM_TYPE_VRAM,
                                                         &result.mem_usage);
            if(!check_status(device_id, status, &local_settings.mem_usage,
                             "amdsmi_get_gpu_memory_usage"))
                return result;
        }

        result.success = true;
        return result;
    }

    bool check_status(uint32_t device_id, amdsmi_status_t status, bool* option,
                      const char* /* function_name */)
    {
        if(status == AMDSMI_STATUS_SUCCESS) return true;

        if(status == AMDSMI_STATUS_NOT_SUPPORTED && option)
        {
            *option = false;
            return true;
        }

        // In test environment, we don't use production logging macros
        // to avoid initialization issues
        disable_device(device_id);
        return false;
    }

    bool is_device_disabled(uint32_t device_id) const
    {
        std::lock_guard<std::mutex> lock(m_disabled_mutex);
        return m_disabled_devices.count(device_id) > 0;
    }

    void disable_device(uint32_t device_id)
    {
        std::lock_guard<std::mutex> lock(m_disabled_mutex);
        m_disabled_devices.insert(device_id);
    }

    const std::set<uint32_t>& disabled_devices() const { return m_disabled_devices; }

private:
    std::shared_ptr<IDriver>          m_driver;
    std::shared_ptr<IClock>           m_clock;
    std::shared_ptr<IStorage>         m_storage;
    std::shared_ptr<IState>           m_state;
    std::shared_ptr<IGpuCapabilities> m_gpu_caps;
    std::set<uint32_t>                m_disabled_devices;
    mutable std::mutex                m_disabled_mutex;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace testing
}  // namespace amd_smi
}  // namespace rocprofsys
