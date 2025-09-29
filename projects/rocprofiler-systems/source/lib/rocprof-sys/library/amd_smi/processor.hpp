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

#include "core/benchmark/benchmark.hpp"
#include "core/benchmark/category.hpp"
#include <algorithm>
#include <amd_smi/amdsmi.h>
#include <bitset>
#include <cstdint>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rocprofsys
{
namespace amd_smi
{

constexpr auto metric_value_not_supported = 0xffff;

#ifdef AMDSMI_MAX_NUM_JPEG_ENG_V1
#    define AMDSMI_MAX_NUM_JPEG_ENGINES AMDSMI_MAX_NUM_JPEG_ENG_V1
#else
#    define AMDSMI_MAX_NUM_JPEG_ENGINES 40
#endif

#ifndef AMDSMI_MAX_NUM_XCP
#    define AMDSMI_MAX_NUM_XCP 8
#endif

struct supported_metrics
{
    uint32_t current_socket_power : 1;
    uint32_t average_socket_power : 1;
    uint32_t memory_usage         : 1;
    uint32_t hotspot_temperature  : 1;
    uint32_t edge_temperature     : 1;
    uint32_t gfx_activity         : 1;
    uint32_t umc_activity         : 1;
    uint32_t mm_activity          : 1;
    struct xcp_metrics_t
    {
        std::bitset<AMDSMI_MAX_NUM_VCN>          vcn_busy;
        std::bitset<AMDSMI_MAX_NUM_JPEG_ENGINES> jpeg_busy;
    } xcp_metrics[AMDSMI_MAX_NUM_XCP];
};

struct smi_metrics
{
    uint32_t current_socket_power;
    uint32_t average_socket_power;
    uint32_t memory_usage;
    uint16_t hotspot_temperature;
    uint16_t edge_temperature;
    uint32_t gfx_activity;
    uint32_t umc_activity;
    uint32_t mm_activity;
    struct
    {
        uint16_t vcn_busy[AMDSMI_MAX_NUM_VCN];
        uint16_t jpeg_busy[AMDSMI_MAX_NUM_JPEG_ENGINES];
    } xcp_stats[AMDSMI_MAX_NUM_XCP];
};

/**
 * @class processor
 * @tparam driver The driver interface type used to communicate with the
 * processor.
 * @brief Encapsulates operations for a single AMD SMI processor.
 *
 * The processor class provides methods to query processor type, power
 * information, and temperature metrics. It is constructed with a shared pointer
 * to a driver, a processor handle, and the processor type.
 */
template <typename driver>
struct processor
{
    /**
     * @brief Constructs a processor object.
     * @param _driver Shared pointer to the driver interface.
     * @param handle The processor handle.
     * @param processor_type The type of the processor.
     */
    processor(std::shared_ptr<driver> _driver, amdsmi_processor_handle handle,
              processor_type_t processor_type, size_t logical_index)
    : m_driver_api{ _driver }
    , m_processor_handle{ handle }
    , m_processor_type{ processor_type }
    , m_index{ logical_index }
    {
        intialize_supported_metrics();
    }

    supported_metrics get_supported_metrics() { return m_supported_metrics; }

    /**
     * @brief Returns the type of the processor.
     * @return The processor type.
     */
    processor_type_t get_processor_type() { return m_processor_type; }

    smi_metrics get_smi_metrics()
    {
        amdsmi_gpu_metrics_t gpu_metrics;
        benchmark::start(benchmark::category::amd_smi_driver);
        auto driver_call_result =
            m_driver_api->get_metrics_info(m_processor_handle, &gpu_metrics);
        if(driver_call_result != AMDSMI_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to read SMI data! AMD SMI Error code: " +
                                     std::to_string(driver_call_result));
        }

        uint64_t memory_usage = std::numeric_limits<uint64_t>::max();
        driver_call_result    = m_driver_api->get_memory_usage(
            m_processor_handle, AMDSMI_MEM_TYPE_VRAM, &memory_usage);
        if(driver_call_result != AMDSMI_STATUS_SUCCESS)
        {
            std::cout << "Failed to read SMI data! AMD SMI Error code: "
                      << driver_call_result << std::endl;
        }
        benchmark::end(benchmark::category::amd_smi_driver);
        auto populate_metrics = [](auto flag, const auto& source, auto& destination) {
            if(flag) destination = source;
        };

        smi_metrics metrics;

        populate_metrics(m_supported_metrics.average_socket_power,
                         gpu_metrics.average_socket_power, metrics.average_socket_power);
        populate_metrics(m_supported_metrics.current_socket_power,
                         gpu_metrics.current_socket_power, metrics.current_socket_power);
        populate_metrics(m_supported_metrics.memory_usage, memory_usage,
                         metrics.memory_usage);
        populate_metrics(m_supported_metrics.gfx_activity,
                         gpu_metrics.average_gfx_activity, metrics.gfx_activity);
        populate_metrics(m_supported_metrics.umc_activity,
                         gpu_metrics.average_umc_activity, metrics.umc_activity);
        populate_metrics(m_supported_metrics.mm_activity, gpu_metrics.average_mm_activity,
                         metrics.mm_activity);
        populate_metrics(m_supported_metrics.edge_temperature,
                         gpu_metrics.temperature_edge, metrics.edge_temperature);
        populate_metrics(m_supported_metrics.hotspot_temperature,
                         gpu_metrics.temperature_hotspot, metrics.hotspot_temperature);

        std::for_each(
            std::begin(m_supported_metrics.xcp_metrics),
            std::end(m_supported_metrics.xcp_metrics),
            [&, xcp_index = 0](const supported_metrics::xcp_metrics_t& xcp) mutable {
                if(xcp.jpeg_busy.any())
                {
                    std::memcpy(metrics.xcp_stats[xcp_index].jpeg_busy,
                                gpu_metrics.xcp_stats[xcp_index].jpeg_busy,
                                sizeof(gpu_metrics.xcp_stats[xcp_index].jpeg_busy));
                }
                if(xcp.vcn_busy.any())
                {
                    std::memcpy(metrics.xcp_stats[xcp_index].vcn_busy,
                                gpu_metrics.xcp_stats[xcp_index].vcn_busy,
                                sizeof(gpu_metrics.xcp_stats[xcp_index].vcn_busy));
                }
                xcp_index++;
            });

        return metrics;
    }

    void print_supported_metrics()
    {
        auto metrics = get_supported_metrics();

        std::cout << "=== SUPPORTED SMI METRICS ===" << '\n';
        std::cout << std::left << std::boolalpha;

        // Power metrics
        std::cout << "  " << std::setw(25) << "current_socket_power"
                  << ": " << (bool) metrics.current_socket_power << '\n';
        std::cout << "  " << std::setw(25) << "average_socket_power"
                  << ": " << (bool) metrics.average_socket_power << '\n';

        // Memory and thermal metrics
        std::cout << "  " << std::setw(25) << "memory_usage"
                  << ": " << (bool) metrics.memory_usage << '\n';
        std::cout << "  " << std::setw(25) << "edge temperature"
                  << ": " << (bool) metrics.edge_temperature << '\n';
        std::cout << "  " << std::setw(25) << "hotspot temperature"
                  << ": " << (bool) metrics.hotspot_temperature << '\n';

        std::cout << "  " << std::setw(25) << "gfx_activity"
                  << ": " << (bool) metrics.gfx_activity << '\n';
        std::cout << "  " << std::setw(25) << "umc_activity"
                  << ": " << (bool) metrics.umc_activity << '\n';
        std::cout << "  " << std::setw(25) << "mm_activity"
                  << ": " << (bool) metrics.mm_activity << '\n';
        std::for_each(std::begin(metrics.xcp_metrics), std::end(metrics.xcp_metrics),
                      [index = 0](const auto& xcp_metircs) mutable {
                          std::cout
                              << "  " << std::setw(25)
                              << "[" + std::to_string(index) + "] vcn_activity" << ": "
                              << bitset_to_index_list(xcp_metircs.vcn_activity) << '\n';
                          std::cout
                              << "  " << std::setw(25)
                              << "[" + std::to_string(index) + "] jpeg_activity" << ": "
                              << bitset_to_index_list(xcp_metircs.jpeg_activity) << '\n';
                          index++;
                      });

        std::cout << "=========================" << std::endl;
    }

    size_t get_index() const { return m_index; }

private:
    template <typename BitsetT>
    static std::string bitset_to_index_list(const BitsetT& bs)
    {
        std::stringstream ss;
        ss << std::boolalpha << "[";
        for(std::size_t i = 0; i < bs.size(); ++i)
        {
            ss << (bool) bs[i];
            if(i != bs.size() - 1)
            {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    void intialize_supported_metrics()
    {
        amdsmi_power_info_t socker_power_info;
        auto                driver_call_result_success =
            m_driver_api->get_power_info(m_processor_handle, &socker_power_info) ==
            AMDSMI_STATUS_SUCCESS;
        m_supported_metrics.average_socket_power =
            driver_call_result_success &&
            socker_power_info.average_socket_power != metric_value_not_supported;
        m_supported_metrics.current_socket_power =
            driver_call_result_success &&
            socker_power_info.current_socket_power != metric_value_not_supported;

        amdsmi_engine_usage_t info;
        driver_call_result_success =
            m_driver_api->get_activity(m_processor_handle, &info) ==
            AMDSMI_STATUS_SUCCESS;
        m_supported_metrics.gfx_activity =
            driver_call_result_success && info.gfx_activity != metric_value_not_supported;
        m_supported_metrics.mm_activity  = driver_call_result_success;
        m_supported_metrics.umc_activity = driver_call_result_success;

        uint64_t memory_usage;
        driver_call_result_success =
            m_driver_api->get_memory_usage(m_processor_handle, AMDSMI_MEM_TYPE_VRAM,
                                           &memory_usage) == AMDSMI_STATUS_SUCCESS;
        m_supported_metrics.memory_usage = driver_call_result_success;

        int64_t temperature;
        driver_call_result_success =
            m_driver_api->get_temperature_metric(
                m_processor_handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMP_CURRENT,
                &temperature) == AMDSMI_STATUS_SUCCESS;
        m_supported_metrics.hotspot_temperature =
            driver_call_result_success && temperature != metric_value_not_supported;

        driver_call_result_success =
            m_driver_api->get_temperature_metric(
                m_processor_handle, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT,
                &temperature) == AMDSMI_STATUS_SUCCESS;
        m_supported_metrics.edge_temperature =
            driver_call_result_success && temperature != metric_value_not_supported;

        amdsmi_gpu_metrics_t gpu_metrics;
        driver_call_result_success =
            m_driver_api->get_metrics_info(m_processor_handle, &gpu_metrics) ==
            AMDSMI_STATUS_SUCCESS;
        std::for_each(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [&, xcp_index = 0](const amdsmi_gpu_xcp_metrics_t& xcp_stats) mutable {
                std::for_each(
                    std::begin(xcp_stats.jpeg_busy), std::end(xcp_stats.jpeg_busy),
                    [&, index = 0](const auto& engine_busy_value) mutable {
                        m_supported_metrics.xcp_metrics[xcp_index].jpeg_busy[index++] =
                            driver_call_result_success &&
                            engine_busy_value != metric_value_not_supported;
                    });
                std::for_each(
                    std::begin(xcp_stats.vcn_busy), std::end(xcp_stats.vcn_busy),
                    [&, index = 0](const auto& engine_busy_value) mutable {
                        m_supported_metrics.xcp_metrics[xcp_index].vcn_busy[index++] =
                            driver_call_result_success &&
                            engine_busy_value != metric_value_not_supported;
                    });
                xcp_index++;
            });
    }

private:
    supported_metrics       m_supported_metrics;
    std::shared_ptr<driver> m_driver_api;
    amdsmi_processor_handle m_processor_handle;
    processor_type_t        m_processor_type;
    size_t                  m_index;
};

}  // namespace amd_smi
}  // namespace rocprofsys
