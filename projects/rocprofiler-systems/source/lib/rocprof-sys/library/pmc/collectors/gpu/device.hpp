// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/amd_smi.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <spdlog/fmt/fmt.h>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu
{

template <typename Driver>
class device
{
public:
    device(std::shared_ptr<Driver> driver, amdsmi_processor_handle handle,
           processor_type_t /*processor_type*/, size_t             logical_index)
    : m_driver_api{ std::move(driver) }
    , m_device_handle{ handle }
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
        metrics metrics{};

        amdsmi_gpu_metrics_t amd_smi_metrics{};
        if(m_driver_api->get_metrics_info(m_device_handle, &amd_smi_metrics) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return metrics;
        }

        collect_power_metrics(amd_smi_metrics, metrics);
        collect_temperature_metrics(amd_smi_metrics, metrics);
        collect_activity_metrics(amd_smi_metrics, metrics);
        collect_memory_metrics(metrics);
        collect_xcp_metrics(amd_smi_metrics, metrics);
        collect_xgmi_metrics(amd_smi_metrics, metrics);
        collect_pcie_metrics(amd_smi_metrics, metrics);

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        if(enabled_cfg.bits.sdma_usage && m_supported_metrics.bits.sdma_usage)
        {
            uint64_t current_cumulative = get_raw_sdma_usage();

            if(m_sdma_state.has_prev && timestamp > m_sdma_state.prev_timestamp)
            {
                uint64_t delta_usage = current_cumulative - m_sdma_state.prev_cumulative;
                uint64_t delta_time  = timestamp - m_sdma_state.prev_timestamp;
                uint32_t pct =
                    static_cast<uint32_t>((delta_usage * 100000ULL) / delta_time);
                metrics.sdma_usage = (pct > 100) ? 100 : pct;
            }

            m_sdma_state.prev_cumulative = current_cumulative;
            m_sdma_state.prev_timestamp  = timestamp;
            m_sdma_state.has_prev        = true;
        }
#endif

        return metrics;
    }

    /**
     * @brief Get raw cumulative SDMA usage from all processes on this GPU.
     *
     * Queries the process list and sums sdma_usage (in microseconds) across
     * all processes. Returns 0 if the query fails or SDMA is not supported.
     * The caller (collector) is responsible for delta computation.
     *
     * @return Cumulative SDMA usage in microseconds.
     */
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    [[nodiscard]] uint64_t get_raw_sdma_usage() const
    {
        uint32_t num_processes = 0;
        auto     status =
            m_driver_api->get_gpu_process_list(m_device_handle, &num_processes, nullptr);
        if(status != AMDSMI_STATUS_SUCCESS || num_processes == 0)
        {
            return 0;
        }

        std::vector<amdsmi_proc_info_t> proc_list(num_processes);
        status = m_driver_api->get_gpu_process_list(m_device_handle, &num_processes,
                                                    proc_list.data());
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            return 0;
        }

        uint64_t cumulative = 0;
        for(const auto& proc : proc_list)
        {
            cumulative += proc.sdma_usage;
        }
        return cumulative;
    }
#endif

private:
    /**
     * @brief Initialize device info (name, product_name, vendor_name).
     *
     * Queries GPU ASIC information from AMD SMI to populate device identification.
     */
    void initialize_device_info()
    {
        // Generate a simple device name based on index
        m_device_name = "GPU" + std::to_string(m_index);

        // Get ASIC info for vendor and product names
        amdsmi_asic_info_t asic_info{};
        if(m_driver_api->get_gpu_asic_info(m_device_handle, &asic_info) ==
           AMDSMI_STATUS_SUCCESS)
        {
            m_product_name = asic_info.market_name;
            m_vendor_name  = asic_info.vendor_name;
        }
        else
        {
            m_product_name = "Unknown GPU";
            m_vendor_name  = "AMD";
        }
    }

    void collect_power_metrics(const amdsmi_gpu_metrics_t& gpu_metrics,
                               metrics&                    out) const
    {
        if(m_supported_metrics.bits.current_socket_power)
        {
            out.current_socket_power = gpu_metrics.current_socket_power;
        }
        if(m_supported_metrics.bits.average_socket_power)
        {
            out.average_socket_power = gpu_metrics.average_socket_power;
        }
    }

    void collect_temperature_metrics(const amdsmi_gpu_metrics_t& gpu_metrics,
                                     metrics&                    out) const
    {
        if(m_supported_metrics.bits.hotspot_temperature)
        {
            out.hotspot_temperature = gpu_metrics.temperature_hotspot;
        }
        if(m_supported_metrics.bits.edge_temperature)
        {
            out.edge_temperature = gpu_metrics.temperature_edge;
        }
    }

    void collect_activity_metrics(const amdsmi_gpu_metrics_t& gpu_metrics,
                                  metrics&                    out) const
    {
        if(m_supported_metrics.bits.gfx_activity)
        {
            out.gfx_activity = gpu_metrics.average_gfx_activity;
        }
        if(m_supported_metrics.bits.umc_activity)
        {
            out.umc_activity = gpu_metrics.average_umc_activity;
        }
        if(m_supported_metrics.bits.mm_activity)
        {
            out.mm_activity = gpu_metrics.average_mm_activity;
        }
    }

    void collect_memory_metrics(metrics& out) const
    {
        if(!m_supported_metrics.bits.memory_usage)
        {
            return;
        }

        uint64_t mem_usage = 0;
        if(m_driver_api->get_memory_usage(m_device_handle, AMDSMI_MEM_TYPE_VRAM,
                                          &mem_usage) == AMDSMI_STATUS_SUCCESS)
        {
            out.memory_usage = mem_usage;
        }
    }

    void collect_xcp_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& out) const
    {
        // Per-XCP VCN busy metrics (MI300)
        if(m_supported_metrics.bits.vcn_busy)
        {
            for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                std::copy(std::begin(gpu_metrics.xcp_stats[xcp].vcn_busy),
                          std::end(gpu_metrics.xcp_stats[xcp].vcn_busy),
                          out.xcp_stats[xcp].vcn_busy.begin());
            }
        }

        // Device-level VCN activity (Radeon)
        if(m_supported_metrics.bits.vcn_activity)
        {
            std::copy(std::begin(gpu_metrics.vcn_activity),
                      std::end(gpu_metrics.vcn_activity), out.vcn_activity.begin());
        }

        // Per-XCP JPEG busy metrics (MI300)
        if(m_supported_metrics.bits.jpeg_busy)
        {
            for(size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
            {
                std::copy(std::begin(gpu_metrics.xcp_stats[xcp].jpeg_busy),
                          std::end(gpu_metrics.xcp_stats[xcp].jpeg_busy),
                          out.xcp_stats[xcp].jpeg_busy.begin());
            }
        }

        // Device-level JPEG activity (Radeon)
        if(m_supported_metrics.bits.jpeg_activity)
        {
            std::copy(std::begin(gpu_metrics.jpeg_activity),
                      std::end(gpu_metrics.jpeg_activity), out.jpeg_activity.begin());
        }
    }

    void collect_xgmi_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& out) const
    {
        if(!m_supported_metrics.bits.xgmi)
        {
            return;
        }

        populate_if_supported(out.xgmi.link.width, gpu_metrics.xgmi_link_width);
        populate_if_supported(out.xgmi.link.speed, gpu_metrics.xgmi_link_speed);

        for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
        {
            populate_if_supported(out.xgmi.data_acc.read[i],
                                  gpu_metrics.xgmi_read_data_acc[i]);
            populate_if_supported(out.xgmi.data_acc.write[i],
                                  gpu_metrics.xgmi_write_data_acc[i]);
        }
    }

    void collect_pcie_metrics(const amdsmi_gpu_metrics_t& gpu_metrics, metrics& out) const
    {
        if(!m_supported_metrics.bits.pcie)
        {
            return;
        }

        populate_if_supported(out.pcie.link.width, gpu_metrics.pcie_link_width);
        populate_if_supported(out.pcie.link.speed, gpu_metrics.pcie_link_speed);
        populate_if_supported(out.pcie.bandwidth.acc, gpu_metrics.pcie_bandwidth_acc);
        populate_if_supported(out.pcie.bandwidth.inst, gpu_metrics.pcie_bandwidth_inst);
    }

    bool initialize_supported_metrics()
    {
        uint64_t mem_usage = 0;
        m_supported_metrics.bits.memory_usage =
            m_driver_api->get_memory_usage(m_device_handle, AMDSMI_MEM_TYPE_VRAM,
                                           &mem_usage) == AMDSMI_STATUS_SUCCESS &&
            is_metric_supported(mem_usage);

        amdsmi_gpu_metrics_t gpu_metrics{};
        if(m_driver_api->get_metrics_info(m_device_handle, &gpu_metrics) !=
           AMDSMI_STATUS_SUCCESS)
        {
            return m_supported_metrics.value != 0;
        }

        m_supported_metrics.bits.current_socket_power =
            is_metric_supported(gpu_metrics.current_socket_power);
        m_supported_metrics.bits.average_socket_power =
            is_metric_supported(gpu_metrics.average_socket_power);

        m_supported_metrics.bits.hotspot_temperature =
            is_metric_supported(gpu_metrics.temperature_hotspot);
        m_supported_metrics.bits.edge_temperature =
            is_metric_supported(gpu_metrics.temperature_edge);

        m_supported_metrics.bits.gfx_activity =
            is_metric_supported(gpu_metrics.average_gfx_activity);
        m_supported_metrics.bits.umc_activity =
            is_metric_supported(gpu_metrics.average_umc_activity);
        m_supported_metrics.bits.mm_activity =
            is_metric_supported(gpu_metrics.average_mm_activity);

        // Check per-XCP VCN/JPEG busy metrics (MI300)
        m_supported_metrics.bits.vcn_busy = std::any_of(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [](const amdsmi_gpu_xcp_metrics_t& xcp_stats) {
                return std::any_of(std::begin(xcp_stats.vcn_busy),
                                   std::end(xcp_stats.vcn_busy),
                                   [](uint16_t v) { return is_metric_supported(v); });
            });

        m_supported_metrics.bits.jpeg_busy = std::any_of(
            std::begin(gpu_metrics.xcp_stats), std::end(gpu_metrics.xcp_stats),
            [](const amdsmi_gpu_xcp_metrics_t& xcp_stats) {
                return std::any_of(std::begin(xcp_stats.jpeg_busy),
                                   std::end(xcp_stats.jpeg_busy),
                                   [](uint16_t v) { return is_metric_supported(v); });
            });

        // Check device-level VCN/JPEG activity metrics (Radeon)
        // Only enable device-level if per-XCP is not available (priority to per-XCP)
        m_supported_metrics.bits.vcn_activity =
            !m_supported_metrics.bits.vcn_busy &&
            std::any_of(std::begin(gpu_metrics.vcn_activity),
                        std::end(gpu_metrics.vcn_activity),
                        [](uint16_t v) { return is_metric_supported(v); });

        m_supported_metrics.bits.jpeg_activity =
            !m_supported_metrics.bits.jpeg_busy &&
            std::any_of(std::begin(gpu_metrics.jpeg_activity),
                        std::end(gpu_metrics.jpeg_activity),
                        [](uint16_t v) { return is_metric_supported(v); });

        m_supported_metrics.bits.xgmi =
            is_metric_supported(gpu_metrics.xgmi_link_width) ||
            is_metric_supported(gpu_metrics.xgmi_link_speed) ||
            std::any_of(std::begin(gpu_metrics.xgmi_read_data_acc),
                        std::end(gpu_metrics.xgmi_read_data_acc),
                        [](uint64_t v) { return is_metric_supported(v); });

        m_supported_metrics.bits.pcie =
            is_metric_supported(gpu_metrics.pcie_link_width) ||
            is_metric_supported(gpu_metrics.pcie_link_speed) ||
            is_metric_supported(gpu_metrics.pcie_bandwidth_acc) ||
            is_metric_supported(gpu_metrics.pcie_bandwidth_inst);

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        {
            uint32_t num_processes = 0;
            m_supported_metrics.bits.sdma_usage =
                m_driver_api->get_gpu_process_list(m_device_handle, &num_processes,
                                                   nullptr) == AMDSMI_STATUS_SUCCESS;
        }
#endif

        LOG_DEBUG("Device [{}] supported metrics: {}", m_index,
                  format_supported_metrics(m_supported_metrics));

        return m_supported_metrics.value != 0;
    }

    static std::string format_supported_metrics(const enabled_metrics& metrics)
    {
        const auto bool_string = [](bool value) { return value ? "true" : "false"; };

        return fmt::format(
            "Current power: {}, Average power: {}, Memory usage: {}, Hotspot temp: {}, "
            "Edge temp: {}, GFX activity: {}, UMC activity: {}, MM activity: {}, "
            "VCN activity: {}, JPEG activity: {}, XGMI: {}, PCIe: {}, SDMA: {}",
            bool_string(metrics.bits.current_socket_power),
            bool_string(metrics.bits.average_socket_power),
            bool_string(metrics.bits.memory_usage),
            bool_string(metrics.bits.hotspot_temperature),
            bool_string(metrics.bits.edge_temperature),
            bool_string(metrics.bits.gfx_activity),
            bool_string(metrics.bits.umc_activity), bool_string(metrics.bits.mm_activity),
            bool_string(metrics.bits.vcn_activity),
            bool_string(metrics.bits.jpeg_activity), bool_string(metrics.bits.xgmi),
            bool_string(metrics.bits.pcie), bool_string(metrics.bits.sdma_usage));
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

    struct sdma_state
    {
        uint64_t prev_cumulative = 0;
        uint64_t prev_timestamp  = 0;
        bool     has_prev        = false;
    };

    std::shared_ptr<Driver> m_driver_api;
    amdsmi_processor_handle m_device_handle;
    enabled_metrics         m_supported_metrics;
    size_t                  m_index;
    std::string             m_device_name;
    std::string             m_product_name;
    std::string             m_vendor_name;
    bool                    m_is_supported = false;
    sdma_state              m_sdma_state;
};

}  // namespace rocprofsys::pmc::collectors::gpu
