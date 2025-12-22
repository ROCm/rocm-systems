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

#include "library/amd_smi/common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace amd_smi
{

#if ROCPROFSYS_USE_ROCM > 0

template <typename Driver>
class processor
{
public:
    processor(std::shared_ptr<Driver> driver, amdsmi_processor_handle handle,
              processor_type_t processor_type, size_t logical_index)
    : m_driver_api{ std::move(driver) }
    , m_processor_handle{ handle }
    , m_processor_type{ processor_type }
    , m_index{ logical_index }
    {
        initialize_supported_metrics();
    }

    enabled_metric get_supported_metrics() const { return m_supported_metrics; }

    processor_type_t get_processor_type() const { return m_processor_type; }

    size_t get_index() const { return m_index; }

    amdsmi_processor_handle get_handle() const { return m_processor_handle; }

    bool is_enabled() const { return m_enabled; }

    void set_enabled(bool enabled) { m_enabled = enabled; }

    bool is_disabled_due_to_error() const { return m_disabled_due_to_error; }

    smi_metrics get_smi_metrics() const
    {
        smi_metrics metrics{};

        if(m_disabled_due_to_error)
        {
            return metrics;
        }

        collect_activity_metrics(metrics);
        collect_power_metrics(metrics);
        collect_temperature_metrics(metrics);
        collect_memory_metrics(metrics);
        collect_gpu_metrics(metrics);

        return metrics;
    }

    void print_supported_metrics() const
    {
        std::cout << "=== SUPPORTED SMI METRICS (Processor " << m_index << ") ===\n";
        std::cout << std::left << std::boolalpha;

        auto print_metric = [](const char* name, bool supported) {
            std::cout << "  " << std::setw(25) << name << ": " << supported << '\n';
        };

        print_metric("current_socket_power",
                     m_supported_metrics.bits.current_socket_power);
        print_metric("average_socket_power",
                     m_supported_metrics.bits.average_socket_power);
        print_metric("memory_usage", m_supported_metrics.bits.memory_usage);
        print_metric("hotspot_temperature", m_supported_metrics.bits.hotspot_temperature);
        print_metric("edge_temperature", m_supported_metrics.bits.edge_temperature);
        print_metric("gfx_activity", m_supported_metrics.bits.gfx_activity);
        print_metric("umc_activity", m_supported_metrics.bits.umc_activity);
        print_metric("mm_activity", m_supported_metrics.bits.mm_activity);
        print_metric("vcn_activity", m_supported_metrics.bits.vcn_activity);
        print_metric("jpeg_activity", m_supported_metrics.bits.jpeg_activity);
        print_metric("xgmi", m_supported_metrics.bits.xgmi);
        print_metric("pcie", m_supported_metrics.bits.pcie);

        std::cout << "=========================\n" << std::flush;
    }

private:
    void collect_activity_metrics(smi_metrics& metrics) const
    {
        if(!m_supported_metrics.bits.gfx_activity &&
           !m_supported_metrics.bits.umc_activity &&
           !m_supported_metrics.bits.mm_activity)
        {
            return;
        }

        amdsmi_engine_usage_t activity{};
        if(m_driver_api->get_activity(m_processor_handle, &activity) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return;
        }

        if(m_supported_metrics.bits.gfx_activity)
        {
            metrics.gfx_activity = activity.gfx_activity;
        }
        if(m_supported_metrics.bits.umc_activity)
        {
            metrics.umc_activity = activity.umc_activity;
        }
        if(m_supported_metrics.bits.mm_activity)
        {
            metrics.mm_activity = activity.mm_activity;
        }
    }

    void collect_power_metrics(smi_metrics& metrics) const
    {
        if(!m_supported_metrics.bits.current_socket_power &&
           !m_supported_metrics.bits.average_socket_power)
        {
            return;
        }

        amdsmi_power_info_t power_info{};
        if(m_driver_api->get_power_info(m_processor_handle, &power_info) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return;
        }

        if(m_supported_metrics.bits.current_socket_power)
        {
            metrics.current_socket_power = power_info.current_socket_power;
        }
        if(m_supported_metrics.bits.average_socket_power)
        {
            metrics.average_socket_power = power_info.average_socket_power;
        }
    }

    void collect_temperature_metrics(smi_metrics& metrics) const
    {
        if(m_supported_metrics.bits.hotspot_temperature)
        {
            int64_t temp = 0;
            if(m_driver_api->get_temperature_metric(
                   m_processor_handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                   AMDSMI_TEMP_CURRENT, &temp) == AMDSMI_STATUS_SUCCESS)
            {
                metrics.hotspot_temperature = temp;
            }
        }

        if(m_supported_metrics.bits.edge_temperature)
        {
            int64_t temp = 0;
            if(m_driver_api->get_temperature_metric(
                   m_processor_handle, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT,
                   &temp) == AMDSMI_STATUS_SUCCESS)
            {
                metrics.edge_temperature = temp;
            }
        }
    }

    void collect_memory_metrics(smi_metrics& metrics) const
    {
        if(!m_supported_metrics.bits.memory_usage)
        {
            return;
        }

        uint64_t mem_usage = 0;
        if(m_driver_api->get_memory_usage(m_processor_handle, AMDSMI_MEM_TYPE_VRAM,
                                          &mem_usage) == AMDSMI_STATUS_SUCCESS)
        {
            metrics.memory_usage = mem_usage;
        }
    }

    void collect_gpu_metrics(smi_metrics& metrics) const
    {
        if(!m_supported_metrics.bits.vcn_activity &&
           !m_supported_metrics.bits.jpeg_activity && !m_supported_metrics.bits.xgmi &&
           !m_supported_metrics.bits.pcie)
        {
            return;
        }

        amdsmi_gpu_metrics_t gpu_metrics{};
        if(m_driver_api->get_metrics_info(m_processor_handle, &gpu_metrics) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return;
        }

        if(m_supported_metrics.bits.vcn_activity)
        {
            for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                std::memcpy(metrics.xcp_stats[xcp].vcn_busy.data(),
                            gpu_metrics.xcp_stats[xcp].vcn_busy,
                            sizeof(gpu_metrics.xcp_stats[xcp].vcn_busy));
            }
        }

        if(m_supported_metrics.bits.jpeg_activity)
        {
            for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                std::memcpy(metrics.xcp_stats[xcp].jpeg_busy.data(),
                            gpu_metrics.xcp_stats[xcp].jpeg_busy,
                            sizeof(gpu_metrics.xcp_stats[xcp].jpeg_busy));
            }
        }

        if(m_supported_metrics.bits.xgmi)
        {
            metrics.xgmi_link_width = (gpu_metrics.xgmi_link_width != UINT16_MAX)
                                          ? gpu_metrics.xgmi_link_width
                                          : 0;
            metrics.xgmi_link_speed = (gpu_metrics.xgmi_link_speed != UINT16_MAX)
                                          ? gpu_metrics.xgmi_link_speed
                                          : 0;

            for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
            {
                metrics.xgmi_read_data_acc[i] =
                    (gpu_metrics.xgmi_read_data_acc[i] != UINT64_MAX)
                        ? gpu_metrics.xgmi_read_data_acc[i]
                        : 0;
                metrics.xgmi_write_data_acc[i] =
                    (gpu_metrics.xgmi_write_data_acc[i] != UINT64_MAX)
                        ? gpu_metrics.xgmi_write_data_acc[i]
                        : 0;
            }
        }

        if(m_supported_metrics.bits.pcie)
        {
            metrics.pcie_link_width     = (gpu_metrics.pcie_link_width != UINT16_MAX)
                                              ? gpu_metrics.pcie_link_width
                                              : 0;
            metrics.pcie_link_speed     = (gpu_metrics.pcie_link_speed != UINT16_MAX)
                                              ? gpu_metrics.pcie_link_speed
                                              : 0;
            metrics.pcie_bandwidth_acc  = (gpu_metrics.pcie_bandwidth_acc != UINT64_MAX)
                                              ? gpu_metrics.pcie_bandwidth_acc
                                              : 0;
            metrics.pcie_bandwidth_inst = (gpu_metrics.pcie_bandwidth_inst != UINT64_MAX)
                                              ? gpu_metrics.pcie_bandwidth_inst
                                              : 0;
        }
    }

    void initialize_supported_metrics()
    {
        amdsmi_power_info_t power_info{};
        if(m_driver_api->get_power_info(m_processor_handle, &power_info) ==
           AMDSMI_STATUS_SUCCESS)
        {
            m_supported_metrics.bits.current_socket_power =
                (power_info.current_socket_power != METRIC_VALUE_NOT_SUPPORTED);
            m_supported_metrics.bits.average_socket_power =
                (power_info.average_socket_power != METRIC_VALUE_NOT_SUPPORTED);
        }

        amdsmi_engine_usage_t activity{};
        if(m_driver_api->get_activity(m_processor_handle, &activity) ==
           AMDSMI_STATUS_SUCCESS)
        {
            m_supported_metrics.bits.gfx_activity =
                (activity.gfx_activity != METRIC_VALUE_NOT_SUPPORTED);
            m_supported_metrics.bits.umc_activity = true;
            m_supported_metrics.bits.mm_activity  = true;
        }

        uint64_t mem_usage = 0;
        m_supported_metrics.bits.memory_usage =
            (m_driver_api->get_memory_usage(m_processor_handle, AMDSMI_MEM_TYPE_VRAM,
                                            &mem_usage) == AMDSMI_STATUS_SUCCESS);

        int64_t temp        = 0;
        auto    temp_result = m_driver_api->get_temperature_metric(
            m_processor_handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMP_CURRENT,
            &temp);
        m_supported_metrics.bits.hotspot_temperature =
            (temp_result == AMDSMI_STATUS_SUCCESS && temp != METRIC_VALUE_NOT_SUPPORTED);

        temp_result = m_driver_api->get_temperature_metric(
            m_processor_handle, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temp);
        m_supported_metrics.bits.edge_temperature =
            (temp_result == AMDSMI_STATUS_SUCCESS && temp != METRIC_VALUE_NOT_SUPPORTED);

        amdsmi_gpu_metrics_t gpu_metrics{};
        if(m_driver_api->get_metrics_info(m_processor_handle, &gpu_metrics) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return;
        }

        m_supported_metrics.bits.vcn_activity = std::any_of(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [](const amdsmi_gpu_xcp_metrics_t& xcp_stats) {
                return std::any_of(
                    std::begin(xcp_stats.vcn_busy), std::end(xcp_stats.vcn_busy),
                    [](uint16_t v) { return v != METRIC_VALUE_NOT_SUPPORTED; });
            });

        m_supported_metrics.bits.jpeg_activity = std::any_of(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [](const amdsmi_gpu_xcp_metrics_t& xcp_stats) {
                return std::any_of(
                    std::begin(xcp_stats.jpeg_busy), std::end(xcp_stats.jpeg_busy),
                    [](uint16_t v) { return v != METRIC_VALUE_NOT_SUPPORTED; });
            });

        m_supported_metrics.bits.xgmi =
            (gpu_metrics.xgmi_link_width != UINT16_MAX) ||
            (gpu_metrics.xgmi_link_speed != UINT16_MAX) ||
            std::any_of(std::begin(gpu_metrics.xgmi_read_data_acc),
                        std::end(gpu_metrics.xgmi_read_data_acc),
                        [](uint64_t v) { return v != UINT64_MAX; });

        m_supported_metrics.bits.pcie = (gpu_metrics.pcie_link_width != UINT16_MAX) ||
                                        (gpu_metrics.pcie_link_speed != UINT16_MAX) ||
                                        (gpu_metrics.pcie_bandwidth_acc != UINT64_MAX) ||
                                        (gpu_metrics.pcie_bandwidth_inst != UINT64_MAX);
    }

private:
    std::shared_ptr<Driver> m_driver_api;
    amdsmi_processor_handle m_processor_handle;
    processor_type_t        m_processor_type;
    enabled_metric          m_supported_metrics{};
    size_t                  m_index;
    bool                    m_enabled               = true;
    bool                    m_disabled_due_to_error = false;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys
