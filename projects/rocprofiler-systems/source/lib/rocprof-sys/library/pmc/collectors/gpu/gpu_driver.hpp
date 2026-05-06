// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/amd_smi.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <amd_smi/amdsmi.h>

namespace rocprofsys::pmc::collectors::gpu
{

class gpu_driver
{
public:
    explicit gpu_driver(amdsmi_processor_handle handle) noexcept
    : m_handle{ handle }
    {}

    [[nodiscard]] asic_info get_gpu_asic_info() const
    {
        amdsmi_asic_info_t raw{};
        check(amdsmi_get_gpu_asic_info(m_handle, &raw), "get_gpu_asic_info");
        return { raw.market_name, raw.vendor_name };
    }

    [[nodiscard]] metrics get_gpu_metrics() const
    {
        amdsmi_gpu_metrics_t raw{};
        check(amdsmi_get_gpu_metrics_info(m_handle, &raw), "get_gpu_metrics_info");

        metrics out{};
        convert_power(raw, out);
        convert_temperature(raw, out);
        convert_activity(raw, out);
        convert_xcp(raw, out);
        convert_xgmi(raw, out);
        convert_pcie(raw, out);
        return out;
    }

    [[nodiscard]] std::uint64_t get_memory_usage() const
    {
        std::uint64_t usage = 0;
        check(amdsmi_get_gpu_memory_usage(m_handle, AMDSMI_MEM_TYPE_VRAM, &usage),
              "get_memory_usage");
        return usage;
    }

    [[nodiscard]] std::uint64_t get_raw_sdma_usage() const
    {
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        std::uint32_t num_processes = 0;
        check(amdsmi_get_gpu_process_list(m_handle, &num_processes, nullptr),
              "get_gpu_process_list (count)");

        if(num_processes == 0)
        {
            return 0;
        }

        std::vector<amdsmi_proc_info_t> proc_list(num_processes);
        check(amdsmi_get_gpu_process_list(m_handle, &num_processes, proc_list.data()),
              "get_gpu_process_list (data)");

        std::uint64_t cumulative = 0;
        for(const auto& proc : proc_list)
        {
            cumulative += proc.sdma_usage;
        }
        return cumulative;
#else
        return 0;
#endif
    }

    [[nodiscard]] bool is_sdma_supported() const noexcept
    {
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        std::uint32_t num_processes = 0;
        return amdsmi_get_gpu_process_list(m_handle, &num_processes, nullptr) ==
               AMDSMI_STATUS_SUCCESS;
#else
        return false;
#endif
    }

private:
    static void check(amdsmi_status_t status, const char* func)
    {
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            const char* status_msg = nullptr;
            if(amdsmi_status_code_to_string(status, &status_msg) ==
                   AMDSMI_STATUS_SUCCESS &&
               status_msg != nullptr)
            {
                throw std::runtime_error(std::string(func) + " failed: " + status_msg);
            }
            throw std::runtime_error(std::string(func) + " failed with status " +
                                     std::to_string(static_cast<int>(status)));
        }
    }

    static void convert_power(const amdsmi_gpu_metrics_t& raw, metrics& out)
    {
        out.current_socket_power = raw.current_socket_power;
        out.average_socket_power = raw.average_socket_power;
    }

    static void convert_temperature(const amdsmi_gpu_metrics_t& raw, metrics& out)
    {
        out.hotspot_temperature = raw.temperature_hotspot;
        out.edge_temperature    = raw.temperature_edge;
    }

    static void convert_activity(const amdsmi_gpu_metrics_t& raw, metrics& out)
    {
        out.gfx_activity = raw.average_gfx_activity;
        out.umc_activity = raw.average_umc_activity;
        out.mm_activity  = raw.average_mm_activity;
    }

    static void convert_xcp(const amdsmi_gpu_metrics_t& raw, metrics& out)
    {
        for(size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            std::copy(std::begin(raw.xcp_stats[xcp].vcn_busy),
                      std::end(raw.xcp_stats[xcp].vcn_busy),
                      out.xcp_stats[xcp].vcn_busy.begin());

            constexpr size_t copy_count =
                std::min(static_cast<size_t>(sizeof(raw.xcp_stats[0].jpeg_busy) /
                                             sizeof(std::uint16_t)),
                         MAX_NUM_JPEG_V1);
            std::copy_n(std::begin(raw.xcp_stats[xcp].jpeg_busy), copy_count,
                        out.xcp_stats[xcp].jpeg_busy.begin());
        }

        std::copy(std::begin(raw.vcn_activity), std::end(raw.vcn_activity),
                  out.vcn_activity.begin());
        std::copy(std::begin(raw.jpeg_activity), std::end(raw.jpeg_activity),
                  out.jpeg_activity.begin());
    }

    static void convert_xgmi(const amdsmi_gpu_metrics_t& raw, metrics& out)
    {
        populate_if_supported(out.xgmi.link.width, raw.xgmi_link_width);
        populate_if_supported(out.xgmi.link.speed, raw.xgmi_link_speed);

        for(size_t idx = 0; idx < MAX_NUM_XGMI_LINKS; ++idx)
        {
            populate_if_supported(out.xgmi.data_acc.read[idx],
                                  raw.xgmi_read_data_acc[idx]);
            populate_if_supported(out.xgmi.data_acc.write[idx],
                                  raw.xgmi_write_data_acc[idx]);
        }
    }

    static void convert_pcie(const amdsmi_gpu_metrics_t& raw, metrics& out)
    {
        populate_if_supported(out.pcie.link.width, raw.pcie_link_width);
        populate_if_supported(out.pcie.link.speed, raw.pcie_link_speed);
        populate_if_supported(out.pcie.bandwidth.acc, raw.pcie_bandwidth_acc);
        populate_if_supported(out.pcie.bandwidth.inst, raw.pcie_bandwidth_inst);
    }

    amdsmi_processor_handle m_handle;
};

}  // namespace rocprofsys::pmc::collectors::gpu
