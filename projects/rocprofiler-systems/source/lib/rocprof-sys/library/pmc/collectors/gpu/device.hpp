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

#include "library/pmc/gpu/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/fmt/ranges.h>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

#if ROCPROFSYS_USE_ROCM > 0

using ::rocprofsys::pmc::gpu::enabled_metrics;
using ::rocprofsys::pmc::gpu::metrics;

template <typename Driver>
class device
{
public:
    device(std::shared_ptr<Driver> driver, amdsmi_processor_handle handle,
           processor_type_t processor_type, size_t logical_index)
    : m_driver_api{ std::move(driver) }
    , m_device_handle{ handle }
    , m_device_type{ processor_type }
    , m_index{ logical_index }
    {
        m_is_supported = initialize_supported_metrics();
    }

    [[nodiscard]] bool is_supported() const noexcept { return m_is_supported; }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
    }

    [[nodiscard]] processor_type_t get_device_type() const noexcept
    {
        return m_device_type;
    }

    [[nodiscard]] size_t get_index() const noexcept { return m_index; }

    [[nodiscard]] amdsmi_processor_handle get_handle() const noexcept
    {
        return m_device_handle;
    }

    [[nodiscard]] metrics get_gpu_metrics(const enabled_metrics& user_enabled) const
    {
        metrics metrics{};

        amdsmi_gpu_metrics_t amd_smi_metrics{};
        auto status = m_driver_api->get_metrics_info(m_device_handle, &amd_smi_metrics);
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to read GPU metrics: AMD SMI error " +
                                     std::to_string(static_cast<int>(status)));
        }

        collect_power_metrics(amd_smi_metrics, metrics, user_enabled);
        collect_temperature_metrics(amd_smi_metrics, metrics, user_enabled);
        collect_activity_metrics(amd_smi_metrics, metrics, user_enabled);
        collect_memory_metrics(metrics, user_enabled);
        collect_xcp_metrics(amd_smi_metrics, metrics, user_enabled);
        collect_xgmi_metrics(amd_smi_metrics, metrics, user_enabled);
        collect_pcie_metrics(amd_smi_metrics, metrics, user_enabled);

        return metrics;
    }

private:
    void collect_power_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& metrics,
                               const enabled_metrics& user_enabled) const
    {
        if(m_supported_metrics.current_socket_power() &&
           user_enabled.current_socket_power())
        {
            metrics.current_socket_power = gpu_metrics.current_socket_power;
        }
        if(m_supported_metrics.average_socket_power() &&
           user_enabled.average_socket_power())
        {
            metrics.average_socket_power = gpu_metrics.average_socket_power;
        }
    }

    void collect_temperature_metrics(const amdsmi_gpu_metrics_t& gpu_metrics,
                                     metrics&                    metrics,
                                     const enabled_metrics&      user_enabled) const
    {
        if(m_supported_metrics.hotspot_temperature() &&
           user_enabled.hotspot_temperature())
        {
            metrics.hotspot_temperature = gpu_metrics.temperature_hotspot;
        }
        if(m_supported_metrics.edge_temperature() && user_enabled.edge_temperature())
        {
            metrics.edge_temperature = gpu_metrics.temperature_edge;
        }
    }

    void collect_activity_metrics(const amdsmi_gpu_metrics_t& gpu_metrics,
                                  metrics&                    metrics,
                                  const enabled_metrics&      user_enabled) const
    {
        metrics.gfx_activity =
            (m_supported_metrics.gfx_activity() && user_enabled.gfx_activity())
                ? gpu_metrics.average_gfx_activity
                : metrics.gfx_activity;
        metrics.umc_activity =
            (m_supported_metrics.umc_activity() && user_enabled.umc_activity())
                ? gpu_metrics.average_umc_activity
                : metrics.umc_activity;
        metrics.mm_activity =
            (m_supported_metrics.mm_activity() && user_enabled.mm_activity())
                ? gpu_metrics.average_mm_activity
                : metrics.mm_activity;
    }

    void collect_memory_metrics(metrics&               metrics,
                                const enabled_metrics& user_enabled) const
    {
        if(!m_supported_metrics.memory_usage() || !user_enabled.memory_usage())
        {
            return;
        }

        uint64_t mem_usage = 0;
        if(m_driver_api->get_memory_usage(m_device_handle, AMDSMI_MEM_TYPE_VRAM,
                                          &mem_usage) == AMDSMI_STATUS_SUCCESS)
        {
            metrics.memory_usage = mem_usage;
        }
    }

    void collect_xcp_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& metrics,
                             const enabled_metrics& user_enabled) const
    {
        if(m_supported_metrics.xcp_vcn_activity() && user_enabled.xcp_vcn_activity())
        {
            for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                std::copy(std::begin(gpu_metrics.xcp_stats[xcp].vcn_busy),
                          std::end(gpu_metrics.xcp_stats[xcp].vcn_busy),
                          metrics.xcp_stats[xcp].vcn_busy.begin());
            }
        }

        if(m_supported_metrics.vcn_activity() && user_enabled.vcn_activity())
        {
            std::copy(std::begin(gpu_metrics.vcn_activity),
                      std::end(gpu_metrics.vcn_activity), metrics.vcn_activity.begin());
        }

        if(m_supported_metrics.xcp_jpeg_activity() && user_enabled.xcp_jpeg_activity())
        {
            for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                std::copy(std::begin(gpu_metrics.xcp_stats[xcp].jpeg_busy),
                          std::end(gpu_metrics.xcp_stats[xcp].jpeg_busy),
                          metrics.xcp_stats[xcp].jpeg_busy.begin());
            }
        }

        if(m_supported_metrics.jpeg_activity() && user_enabled.jpeg_activity())
        {
            std::copy(std::begin(gpu_metrics.jpeg_activity),
                      std::end(gpu_metrics.jpeg_activity), metrics.jpeg_activity.begin());
        }
    }

    void collect_xgmi_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& metrics,
                              const enabled_metrics& user_enabled) const
    {
        if(!m_supported_metrics.xgmi() || !user_enabled.xgmi())
        {
            return;
        }

        populate_if_supported(metrics.xgmi.link.width, gpu_metrics.xgmi_link_width);
        populate_if_supported(metrics.xgmi.link.speed, gpu_metrics.xgmi_link_speed);

        for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
        {
            populate_if_supported(metrics.xgmi.data_acc.read[i],
                                  gpu_metrics.xgmi_read_data_acc[i]);
            populate_if_supported(metrics.xgmi.data_acc.write[i],
                                  gpu_metrics.xgmi_write_data_acc[i]);
        }
    }

    void collect_pcie_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& metrics,
                              const enabled_metrics& user_enabled) const
    {
        if(!m_supported_metrics.pcie() || !user_enabled.pcie())
        {
            return;
        }

        populate_if_supported(metrics.pcie.link.width, gpu_metrics.pcie_link_width);
        populate_if_supported(metrics.pcie.link.speed, gpu_metrics.pcie_link_speed);
        populate_if_supported(metrics.pcie.bandwidth.acc, gpu_metrics.pcie_bandwidth_acc);
        populate_if_supported(metrics.pcie.bandwidth.inst,
                              gpu_metrics.pcie_bandwidth_inst);
    }

    bool initialize_supported_metrics()
    {
        uint64_t mem_usage = 0;
        m_supported_metrics.set_memory_usage(
            m_driver_api->get_memory_usage(m_device_handle, AMDSMI_MEM_TYPE_VRAM,
                                           &mem_usage) == AMDSMI_STATUS_SUCCESS &&
            is_metric_supported(mem_usage));

        amdsmi_gpu_metrics_t gpu_metrics{};
        if(m_driver_api->get_metrics_info(m_device_handle, &gpu_metrics) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return m_supported_metrics.any();
        }

        m_supported_metrics.set_current_socket_power(
            is_metric_supported(gpu_metrics.current_socket_power));
        m_supported_metrics.set_average_socket_power(
            is_metric_supported(gpu_metrics.average_socket_power));

        m_supported_metrics.set_hotspot_temperature(
            is_metric_supported(gpu_metrics.temperature_hotspot));
        m_supported_metrics.set_edge_temperature(
            is_metric_supported(gpu_metrics.temperature_edge));

        m_supported_metrics.set_gfx_activity(
            is_metric_supported(gpu_metrics.average_gfx_activity));
        m_supported_metrics.set_umc_activity(
            is_metric_supported(gpu_metrics.average_umc_activity));
        m_supported_metrics.set_mm_activity(
            is_metric_supported(gpu_metrics.average_mm_activity));

        m_supported_metrics.set_xcp_vcn_activity(std::any_of(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [](const amdsmi_gpu_xcp_metrics_t& xcp_stats) {
                return std::any_of(std::begin(xcp_stats.vcn_busy),
                                   std::end(xcp_stats.vcn_busy),
                                   [](uint16_t v) { return is_metric_supported(v); });
            }));

        m_supported_metrics.set_xcp_jpeg_activity(std::any_of(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [](const amdsmi_gpu_xcp_metrics_t& xcp_stats) {
                return std::any_of(std::begin(xcp_stats.jpeg_busy),
                                   std::end(xcp_stats.jpeg_busy),
                                   [](uint16_t v) { return is_metric_supported(v); });
            }));

        m_supported_metrics.set_vcn_activity(
            !m_supported_metrics.xcp_vcn_activity() &&
            std::any_of(std::begin(gpu_metrics.vcn_activity),
                        std::end(gpu_metrics.vcn_activity),
                        [](uint16_t v) { return is_metric_supported(v); }));

        m_supported_metrics.set_jpeg_activity(
            !m_supported_metrics.xcp_jpeg_activity() &&
            std::any_of(std::begin(gpu_metrics.jpeg_activity),
                        std::end(gpu_metrics.jpeg_activity),
                        [](uint16_t v) { return is_metric_supported(v); }));

        m_supported_metrics.set_xgmi(
            is_metric_supported(gpu_metrics.xgmi_link_width) ||
            is_metric_supported(gpu_metrics.xgmi_link_speed) ||
            std::any_of(std::begin(gpu_metrics.xgmi_read_data_acc),
                        std::end(gpu_metrics.xgmi_read_data_acc),
                        [](uint64_t v) { return is_metric_supported(v); }));

        m_supported_metrics.set_pcie(
            is_metric_supported(gpu_metrics.pcie_link_width) ||
            is_metric_supported(gpu_metrics.pcie_link_speed) ||
            is_metric_supported(gpu_metrics.pcie_bandwidth_acc) ||
            is_metric_supported(gpu_metrics.pcie_bandwidth_inst));

        LOG_DEBUG("Device [{}] supported metrics: {}", m_index,
                  format_supported_metrics(m_supported_metrics));

        return m_supported_metrics.any();
    }

    static std::string format_supported_metrics(const enabled_metrics& metrics)
    {
        return fmt::format("Current power: {}, Average power: {}, Memory usage: {}, "
                           "Hotspot temp: {}, Edge temp: {}, GFX activity: {}, "
                           "UMC activity: {}, MM activity: {}, VCN busy: {}, "
                           "VCN activity: {}, JPEG busy: {}, JPEG activity: {}, "
                           "XGMI: {}, PCIe: {}",
                           metrics.current_socket_power(), metrics.average_socket_power(),
                           metrics.memory_usage(), metrics.hotspot_temperature(),
                           metrics.edge_temperature(), metrics.gfx_activity(),
                           metrics.umc_activity(), metrics.mm_activity(),
                           metrics.xcp_vcn_activity(), metrics.vcn_activity(),
                           metrics.xcp_jpeg_activity(), metrics.jpeg_activity(),
                           metrics.xgmi(), metrics.pcie());
    }

    template <typename T>
    static bool is_metric_supported(T value,
                                    T invalid_sentinel = std::numeric_limits<T>::max())
    {
        return value != invalid_sentinel;
    }

    template <typename T>
    static bool populate_if_supported(T& dest, T src,
                                      T invalid_sentinel = std::numeric_limits<T>::max())
    {
        const bool valid = is_metric_supported(src, invalid_sentinel);
        dest             = valid ? src : T{ 0 };
        return valid;
    }

    std::shared_ptr<Driver> m_driver_api;
    amdsmi_processor_handle m_device_handle;
    processor_type_t        m_device_type;
    enabled_metrics         m_supported_metrics;
    size_t                  m_index;
    bool                    m_is_supported = false;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
