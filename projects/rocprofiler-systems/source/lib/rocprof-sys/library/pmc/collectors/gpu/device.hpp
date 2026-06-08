// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/amd_smi_probe.hpp"
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
        [[maybe_unused]] std::uint64_t          timestamp)
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

            if(m_supported_metrics.bits.gfx_clock)
            {
                gpu_metrics.gfx_clock_mhz = raw.gfx_clock_mhz;
            }
            if(m_supported_metrics.bits.mem_clock)
            {
                gpu_metrics.mem_clock_mhz = raw.mem_clock_mhz;
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
        static_assert(sizeof(enabled_metrics) ==
                          sizeof(::rocprofsys::amd_smi::metric_availability),
                      "enabled_metrics and metric_availability must match");

        bool memory_supported = false;
        try
        {
            const auto usage = m_driver->get_memory_usage();
            memory_supported = is_metric_supported(usage, METRIC_VALUE_NOT_SUPPORTED_64);
        } catch(const std::runtime_error&)
        {
            memory_supported = false;
        }

        ::rocprofsys::amd_smi::probe_input input{};
        try
        {
            const auto raw             = m_driver->get_gpu_metrics();
            input.current_socket_power = raw.current_socket_power;
            input.average_socket_power = raw.average_socket_power;
            input.hotspot_temperature  = raw.hotspot_temperature;
            input.edge_temperature     = raw.edge_temperature;
            input.gfx_activity         = raw.gfx_activity;
            input.umc_activity         = raw.umc_activity;
            input.mm_activity          = raw.mm_activity;
            input.gfx_clock_mhz        = raw.gfx_clock_mhz;
            input.mem_clock_mhz        = raw.mem_clock_mhz;
            input.vcn_activity         = raw.vcn_activity;
            input.jpeg_activity        = raw.jpeg_activity;
            for(std::size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
            {
                input.xcp_stats[xcp].vcn_busy  = raw.xcp_stats[xcp].vcn_busy;
                input.xcp_stats[xcp].jpeg_busy = raw.xcp_stats[xcp].jpeg_busy;
            }
            input.xgmi.width          = raw.xgmi.link.width;
            input.xgmi.speed          = raw.xgmi.link.speed;
            input.xgmi.data_acc.read  = raw.xgmi.data_acc.read;
            input.xgmi.data_acc.write = raw.xgmi.data_acc.write;
            input.pcie.width          = raw.pcie.link.width;
            input.pcie.speed          = raw.pcie.link.speed;
            input.pcie.bandwidth.acc  = raw.pcie.bandwidth.acc;
            input.pcie.bandwidth.inst = raw.pcie.bandwidth.inst;
        } catch(const std::runtime_error&)
        {
            m_supported_metrics.value = memory_supported ? (1u << 2) : 0;
            return m_supported_metrics.value != 0;
        }

        const bool sdma_supported = m_driver->is_sdma_supported();
        m_supported_metrics.value = ::rocprofsys::amd_smi::compute_availability(
                                        input, memory_supported, sdma_supported)
                                        .value;

        LOG_DEBUG("Device [{}] supported metrics: {}", m_index,
                  format_supported_metrics(m_supported_metrics));

        return m_supported_metrics.value != 0;
    }

    void collect_sdma_metrics([[maybe_unused]] const enabled_metrics& enabled_cfg,
                              [[maybe_unused]] std::uint64_t          timestamp,
                              [[maybe_unused]] metrics&               out)
    {
        if(!enabled_cfg.bits.sdma_usage || !m_supported_metrics.bits.sdma_usage)
        {
            return;
        }

        try
        {
            std::uint64_t current_cumulative = m_driver->get_raw_sdma_usage();

            if(m_sdma_state.has_prev && timestamp > m_sdma_state.prev_timestamp)
            {
                std::uint64_t delta_usage =
                    current_cumulative - m_sdma_state.prev_cumulative;
                std::uint64_t delta_time = timestamp - m_sdma_state.prev_timestamp;
                std::uint32_t pct =
                    static_cast<std::uint32_t>((delta_usage * 100000ULL) / delta_time);
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
            "XGMI: {}, PCIe: {}, SDMA: {}, GFX clock: {}, Mem clock: {}",
            bstr(met.bits.current_socket_power), bstr(met.bits.average_socket_power),
            bstr(met.bits.memory_usage), bstr(met.bits.hotspot_temperature),
            bstr(met.bits.edge_temperature), bstr(met.bits.gfx_activity),
            bstr(met.bits.umc_activity), bstr(met.bits.mm_activity),
            bstr(met.bits.vcn_activity), bstr(met.bits.jpeg_activity),
            bstr(met.bits.vcn_busy), bstr(met.bits.jpeg_busy), bstr(met.bits.xgmi),
            bstr(met.bits.pcie), bstr(met.bits.sdma_usage), bstr(met.bits.gfx_clock),
            bstr(met.bits.mem_clock));
    }

    struct sdma_state
    {
        std::uint64_t prev_cumulative = 0;
        std::uint64_t prev_timestamp  = 0;
        bool          has_prev        = false;
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
