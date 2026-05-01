// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <spdlog/fmt/fmt.h>
#include <stdexcept>
#include <string>

namespace rocprofsys::pmc::collectors::gpu
{

template <typename Driver>
class device
{
public:
    device(std::shared_ptr<Driver> driver, size_t logical_index)
    : m_driver{ std::move(driver) }
    , m_index{ logical_index }
    {
        initialize_device_info();
        m_is_supported = initialize_supported_metrics();
    }

    [[nodiscard]] bool is_supported() const noexcept { return m_is_supported; }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
    }

    [[nodiscard]] size_t get_index() const noexcept { return m_index; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_device_name; }

    [[nodiscard]] const std::string& get_product_name() const noexcept
    {
        return m_product_name;
    }

    [[nodiscard]] const std::string& get_vendor_name() const noexcept
    {
        return m_vendor_name;
    }

    [[nodiscard]] metrics get_gpu_metrics(
        [[maybe_unused]] const enabled_metrics& enabled_cfg,
        [[maybe_unused]] uint64_t               timestamp)
    {
        metrics gpu_metrics{};

        try
        {
            auto raw = m_driver->get_gpu_metrics();

            if(m_supported_metrics.bits.current_socket_power)
            {
                gpu_metrics.current_socket_power = raw.current_socket_power;
            }
            if(m_supported_metrics.bits.average_socket_power)
            {
                gpu_metrics.average_socket_power = raw.average_socket_power;
            }
            if(m_supported_metrics.bits.hotspot_temperature)
            {
                gpu_metrics.hotspot_temperature = raw.hotspot_temperature;
            }
            if(m_supported_metrics.bits.edge_temperature)
            {
                gpu_metrics.edge_temperature = raw.edge_temperature;
            }
            if(m_supported_metrics.bits.gfx_activity)
            {
                gpu_metrics.gfx_activity = raw.gfx_activity;
            }
            if(m_supported_metrics.bits.umc_activity)
            {
                gpu_metrics.umc_activity = raw.umc_activity;
            }
            if(m_supported_metrics.bits.mm_activity)
            {
                gpu_metrics.mm_activity = raw.mm_activity;
            }

            if(m_supported_metrics.bits.vcn_busy)
            {
                for(size_t i = 0; i < raw.xcp_stats.size(); ++i)
                {
                    gpu_metrics.xcp_stats[i].vcn_busy = raw.xcp_stats[i].vcn_busy;
                }
            }
            if(m_supported_metrics.bits.vcn_activity)
            {
                gpu_metrics.vcn_activity = raw.vcn_activity;
            }
            if(m_supported_metrics.bits.jpeg_busy)
            {
                for(size_t i = 0; i < raw.xcp_stats.size(); ++i)
                {
                    gpu_metrics.xcp_stats[i].jpeg_busy = raw.xcp_stats[i].jpeg_busy;
                }
            }
            if(m_supported_metrics.bits.jpeg_activity)
            {
                gpu_metrics.jpeg_activity = raw.jpeg_activity;
            }

            if(m_supported_metrics.bits.xgmi)
            {
                populate_if_supported(gpu_metrics.xgmi.link.width, raw.xgmi.link.width);
                populate_if_supported(gpu_metrics.xgmi.link.speed, raw.xgmi.link.speed);
                for(size_t i = 0; i < MAX_NUM_XGMI_LINKS; ++i)
                {
                    populate_if_supported(gpu_metrics.xgmi.data_acc.read[i],
                                          raw.xgmi.data_acc.read[i]);
                    populate_if_supported(gpu_metrics.xgmi.data_acc.write[i],
                                          raw.xgmi.data_acc.write[i]);
                }
            }
            if(m_supported_metrics.bits.pcie)
            {
                populate_if_supported(gpu_metrics.pcie.link.width, raw.pcie.link.width);
                populate_if_supported(gpu_metrics.pcie.link.speed, raw.pcie.link.speed);
                populate_if_supported(gpu_metrics.pcie.bandwidth.acc,
                                      raw.pcie.bandwidth.acc);
                populate_if_supported(gpu_metrics.pcie.bandwidth.inst,
                                      raw.pcie.bandwidth.inst);
            }
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("GPU device [{}] metrics query failed: {}", m_index, e.what());
            return gpu_metrics;
        }

        if(m_supported_metrics.bits.memory_usage)
        {
            try
            {
                gpu_metrics.memory_usage = m_driver->get_memory_usage();
            } catch(const std::runtime_error& e)
            {
                LOG_DEBUG("GPU device [{}] memory query failed: {}", m_index, e.what());
            }
        }

        collect_sdma_metrics(enabled_cfg, timestamp, gpu_metrics);

        return gpu_metrics;
    }

private:
    void initialize_device_info()
    {
        m_device_name = "GPU" + std::to_string(m_index);

        try
        {
            auto info      = m_driver->get_gpu_asic_info();
            m_product_name = info.product_name;
            m_vendor_name  = info.vendor_name;
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("GPU device [{}]: {}", m_index, e.what());
            m_product_name = "Unknown GPU";
            m_vendor_name  = "AMD";
        }
    }

    bool initialize_supported_metrics()
    {
        try
        {
            const auto usage = m_driver->get_memory_usage();
            m_supported_metrics.bits.memory_usage =
                is_metric_supported(usage, METRIC_VALUE_NOT_SUPPORTED_64) ? 1 : 0;
        } catch(const std::runtime_error&)
        {
            m_supported_metrics.bits.memory_usage = 0;
        }

        metrics raw{};
        try
        {
            raw = m_driver->get_gpu_metrics();
        } catch(const std::runtime_error&)
        {
            return m_supported_metrics.value != 0;
        }

        m_supported_metrics.bits.current_socket_power =
            is_metric_supported(raw.current_socket_power, METRIC_VALUE_NOT_SUPPORTED_16);
        m_supported_metrics.bits.average_socket_power =
            is_metric_supported(raw.average_socket_power, METRIC_VALUE_NOT_SUPPORTED_16);
        m_supported_metrics.bits.hotspot_temperature =
            is_metric_supported(raw.hotspot_temperature, METRIC_VALUE_NOT_SUPPORTED_16);
        m_supported_metrics.bits.edge_temperature =
            is_metric_supported(raw.edge_temperature, METRIC_VALUE_NOT_SUPPORTED_16);
        m_supported_metrics.bits.gfx_activity =
            is_metric_supported(raw.gfx_activity, METRIC_VALUE_NOT_SUPPORTED_16);
        m_supported_metrics.bits.umc_activity =
            is_metric_supported(raw.umc_activity, METRIC_VALUE_NOT_SUPPORTED_16);
        m_supported_metrics.bits.mm_activity =
            is_metric_supported(raw.mm_activity, METRIC_VALUE_NOT_SUPPORTED_16);

        m_supported_metrics.bits.vcn_busy = std::any_of(
            raw.xcp_stats.begin(), raw.xcp_stats.end(),
            [](const metrics::xcp_metrics& xcp) {
                return std::any_of(xcp.vcn_busy.begin(), xcp.vcn_busy.end(),
                                   [](uint16_t val) { return is_metric_supported(val); });
            });

        m_supported_metrics.bits.jpeg_busy = std::any_of(
            raw.xcp_stats.begin(), raw.xcp_stats.end(),
            [](const metrics::xcp_metrics& xcp) {
                return std::any_of(xcp.jpeg_busy.begin(), xcp.jpeg_busy.end(),
                                   [](uint16_t val) { return is_metric_supported(val); });
            });

        m_supported_metrics.bits.vcn_activity =
            !m_supported_metrics.bits.vcn_busy &&
            std::any_of(raw.vcn_activity.begin(), raw.vcn_activity.end(),
                        [](uint16_t val) { return is_metric_supported(val); });

        m_supported_metrics.bits.jpeg_activity =
            !m_supported_metrics.bits.jpeg_busy &&
            std::any_of(raw.jpeg_activity.begin(), raw.jpeg_activity.end(),
                        [](uint16_t val) { return is_metric_supported(val); });

        m_supported_metrics.bits.xgmi =
            is_metric_supported(raw.xgmi.link.width) ||
            is_metric_supported(raw.xgmi.link.speed) ||
            std::any_of(raw.xgmi.data_acc.read.begin(), raw.xgmi.data_acc.read.end(),
                        [](uint64_t val) { return is_metric_supported(val); });

        m_supported_metrics.bits.pcie = is_metric_supported(raw.pcie.link.width) ||
                                        is_metric_supported(raw.pcie.link.speed) ||
                                        is_metric_supported(raw.pcie.bandwidth.acc) ||
                                        is_metric_supported(raw.pcie.bandwidth.inst);

        initialize_sdma_support();

        LOG_DEBUG("Device [{}] supported metrics: {}", m_index,
                  format_supported_metrics(m_supported_metrics));

        return m_supported_metrics.value != 0;
    }

    void initialize_sdma_support()
    {
        m_supported_metrics.bits.sdma_usage = m_driver->is_sdma_supported() ? 1 : 0;
    }

    void collect_sdma_metrics([[maybe_unused]] const enabled_metrics& enabled_cfg,
                              [[maybe_unused]] uint64_t               timestamp,
                              [[maybe_unused]] metrics&               out)
    {
        if(!enabled_cfg.bits.sdma_usage || !m_supported_metrics.bits.sdma_usage)
        {
            return;
        }

        try
        {
            uint64_t current_cumulative = m_driver->get_raw_sdma_usage();

            if(m_sdma_state.has_prev && timestamp > m_sdma_state.prev_timestamp)
            {
                uint64_t delta_usage = current_cumulative - m_sdma_state.prev_cumulative;
                uint64_t delta_time  = timestamp - m_sdma_state.prev_timestamp;
                uint32_t pct =
                    static_cast<uint32_t>((delta_usage * 100000ULL) / delta_time);
                out.sdma_usage = (pct > 100) ? 100 : pct;
            }

            m_sdma_state.prev_cumulative = current_cumulative;
            m_sdma_state.prev_timestamp  = timestamp;
            m_sdma_state.has_prev        = true;
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("GPU device [{}] SDMA query failed: {}", m_index, e.what());
        }
    }

    static std::string format_supported_metrics(const enabled_metrics& met)
    {
        const auto bstr = [](bool val) { return val ? "true" : "false"; };

        return fmt::format(
            "Current power: {}, Average power: {}, Memory usage: {}, Hotspot temp: {}, "
            "Edge temp: {}, GFX activity: {}, UMC activity: {}, MM activity: {}, "
            "VCN activity: {}, JPEG activity: {}, VCN busy: {}, JPEG busy: {}, "
            "XGMI: {}, PCIe: {}, SDMA: {}",
            bstr(met.bits.current_socket_power), bstr(met.bits.average_socket_power),
            bstr(met.bits.memory_usage), bstr(met.bits.hotspot_temperature),
            bstr(met.bits.edge_temperature), bstr(met.bits.gfx_activity),
            bstr(met.bits.umc_activity), bstr(met.bits.mm_activity),
            bstr(met.bits.vcn_activity), bstr(met.bits.jpeg_activity),
            bstr(met.bits.vcn_busy), bstr(met.bits.jpeg_busy), bstr(met.bits.xgmi),
            bstr(met.bits.pcie), bstr(met.bits.sdma_usage));
    }

    struct sdma_state
    {
        uint64_t prev_cumulative = 0;
        uint64_t prev_timestamp  = 0;
        bool     has_prev        = false;
    };

    std::shared_ptr<Driver> m_driver;
    enabled_metrics         m_supported_metrics;
    size_t                  m_index;
    std::string             m_device_name;
    std::string             m_product_name;
    std::string             m_vendor_name;
    bool                    m_is_supported = false;
    sdma_state              m_sdma_state;
};

}  // namespace rocprofsys::pmc::collectors::gpu
