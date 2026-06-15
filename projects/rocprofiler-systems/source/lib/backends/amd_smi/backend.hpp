// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/amdsmi_backend.hpp"
#include "backends/amd_smi/gpu_types.hpp"
#include "backends/amd_smi/nic_types.hpp"
#include "backends/amd_smi/sdma_feature.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::backends::amd_smi
{

using gpu::asic_info;
using gpu::MAX_NUM_JPEG_V1;
using gpu::MAX_NUM_XCP;
using gpu::MAX_NUM_XGMI_LINKS;
using gpu::metrics;
using gpu::populate_if_supported;

/**
 * @brief Smart AMD SMI backend — owns an AmdsmiBackend policy and one processor handle.
 *
 * @c backend<AmdsmiBackend> is the single layer that:
 *  - Calls raw AmdsmiBackend methods (1:1 AMD SMI forwarding)
 *  - Checks every return status via check_call() — throws std::runtime_error on failure
 *  - Converts raw AMD SMI structs to domain types (metrics, asic_info, …)
 *
 * Static members handle global lifecycle (initialize / shutdown / enumeration)
 * so the provider layer does not need to know about AMD SMI types at all.
 *
 * @tparam AmdsmiBackend  Policy providing raw AMD SMI call wrappers and type aliases.
 *                        Defaults to the real @c amdsmi_backend; swap for a mock in
 * tests.
 */
template <typename AmdsmiBackend = amdsmi_backend>
class backend
{
public:
    explicit backend(typename AmdsmiBackend::processor_handle handle) noexcept
    : m_handle{ handle }
    {}

    // ── Static lifecycle ──────────────────────────────────────────────────────

    /**
     * @brief Initialize the AMD SMI library.
     * @throws std::runtime_error on failure.
     */
    static void initialize() { check_call(AmdsmiBackend::init(), "amdsmi_init"); }

    /**
     * @brief Shutdown the AMD SMI library (best-effort, never throws).
     */
    static void shutdown() noexcept { AmdsmiBackend::shutdown(); }

    /**
     * @brief Get AMD SMI library version.
     * @throws std::runtime_error on failure.
     */
    [[nodiscard]] static typename AmdsmiBackend::version_t get_lib_version()
    {
        typename AmdsmiBackend::version_t ver{};
        check_call(AmdsmiBackend::get_version(&ver), "amdsmi_get_lib_version");
        return ver;
    }

    // ── Static enumeration ────────────────────────────────────────────────────

    /**
     * @brief Enumerate all GPU processor handles across all sockets.
     *
     * Sockets that report no GPU processors are skipped silently.
     * @throws std::runtime_error if socket enumeration or any data fetch fails.
     */
    [[nodiscard]] static std::vector<typename AmdsmiBackend::processor_handle>
    enumerate_gpu_handles()
    {
        return enumerate_handles([](typename AmdsmiBackend::socket_handle     socket,
                                    std::uint32_t*                            count,
                                    typename AmdsmiBackend::processor_handle* procs) {
            return AmdsmiBackend::get_processor_handles(socket, count, procs);
        });
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    /**
     * @brief Enumerate all NIC processor handles across all sockets.
     *
     * Sockets that report no NIC processors are skipped silently.
     * @throws std::runtime_error if socket enumeration or any data fetch fails.
     */
    [[nodiscard]] static std::vector<typename AmdsmiBackend::processor_handle>
    enumerate_nic_handles()
    {
        return enumerate_handles([](typename AmdsmiBackend::socket_handle     socket,
                                    std::uint32_t*                            count,
                                    typename AmdsmiBackend::processor_handle* procs) {
            return AmdsmiBackend::get_processor_handles_by_type(
                socket, AMDSMI_PROCESSOR_TYPE_AMD_NIC, procs, count);
        });
    }
#endif

    // ── Per-device GPU queries ────────────────────────────────────────────────

    [[nodiscard]] asic_info get_gpu_asic_info() const
    {
        typename AmdsmiBackend::asic_info_t raw{};
        check_call(m_amdsmi.get_gpu_asic_info(m_handle, &raw),
                   "amdsmi_get_gpu_asic_info");
        return { raw.market_name, raw.vendor_name };
    }

    [[nodiscard]] metrics get_gpu_metrics() const
    {
        typename AmdsmiBackend::gpu_metrics_t raw{};
        check_call(m_amdsmi.get_metrics_info(m_handle, &raw),
                   "amdsmi_get_gpu_metrics_info");

        metrics out{};
        convert_power(raw, out);
        convert_temperature(raw, out);
        convert_activity(raw, out);
        convert_xcp(raw, out);
        convert_xgmi(raw, out);
        convert_pcie(raw, out);
        convert_clocks(raw, out);
        return out;
    }

    [[nodiscard]] std::uint64_t get_memory_usage() const
    {
        std::uint64_t usage = 0;
        check_call(
            m_amdsmi.get_memory_usage(m_handle, AmdsmiBackend::MEM_TYPE_VRAM, &usage),
            "amdsmi_get_gpu_memory_usage");
        return usage;
    }

    [[nodiscard]] std::uint64_t get_raw_sdma_usage() const
    {
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        std::uint32_t count = 0;
        if(m_amdsmi.get_gpu_process_list(m_handle, &count, nullptr) !=
               AmdsmiBackend::STATUS_SUCCESS ||
           count == 0)
            return 0;

        std::vector<typename AmdsmiBackend::proc_info_t> procs(count);
        check_call(m_amdsmi.get_gpu_process_list(m_handle, &count, procs.data()),
                   "amdsmi_get_gpu_process_list");

        std::uint64_t cumulative = 0;
        for(const auto& proc : procs)
            cumulative += proc.sdma_usage;
        return cumulative;
#else
        return 0;
#endif
    }

    [[nodiscard]] bool is_sdma_supported() const noexcept
    {
#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
        std::uint32_t count = 0;
        return m_amdsmi.get_gpu_process_list(m_handle, &count, nullptr) ==
               AmdsmiBackend::STATUS_SUCCESS;
#else
        return false;
#endif
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    // ── Per-device NIC queries ────────────────────────────────────────────────

    [[nodiscard]] nic::asic_info get_nic_asic_info() const
    {
        typename AmdsmiBackend::nic_asic_info_t raw{};
        check_call(m_amdsmi.get_nic_asic_info(m_handle, &raw),
                   "amdsmi_get_nic_asic_info");
        return { raw.product_name, raw.vendor_name };
    }

    [[nodiscard]] nic::port_info get_nic_port_info() const
    {
        typename AmdsmiBackend::nic_port_info_t raw{};
        check_call(m_amdsmi.get_nic_port_info(m_handle, &raw),
                   "amdsmi_get_nic_port_info");
        if(raw.num_ports == 0) return {};
        return { raw.ports[0].netdev };
    }

    [[nodiscard]] nic::rdma_info get_nic_rdma_info() const
    {
        // Heap-allocate: amdsmi_nic_rdma_devices_info_t is ~558 KiB.
        auto raw = std::make_unique<typename AmdsmiBackend::nic_rdma_devices_info_t>();
        check_call(m_amdsmi.get_nic_rdma_dev_info(m_handle, raw.get()),
                   "amdsmi_get_nic_rdma_dev_info");
        if(raw->num_rdma_dev == 0) return { 0 };
        return { raw->rdma_dev_info[0].num_rdma_ports };
    }

    [[nodiscard]] std::vector<nic::stat_entry> get_nic_rdma_port_statistics(
        std::uint8_t rdma_port_idx) const
    {
        std::uint32_t count = 0;
        check_call(m_amdsmi.get_nic_rdma_port_statistics(m_handle, rdma_port_idx, &count,
                                                         nullptr),
                   "amdsmi_get_nic_rdma_port_statistics (count)");

        if(count == 0) return {};

        std::vector<typename AmdsmiBackend::nic_stat_t> raw_stats(count);
        check_call(m_amdsmi.get_nic_rdma_port_statistics(m_handle, rdma_port_idx, &count,
                                                         raw_stats.data()),
                   "amdsmi_get_nic_rdma_port_statistics (data)");

        std::vector<nic::stat_entry> result;
        result.reserve(count);
        for(const auto& stat : raw_stats)
            result.emplace_back(nic::stat_entry{ stat.name, stat.value });
        return result;
    }
#endif

private:
    using gpu_metrics_t = typename AmdsmiBackend::gpu_metrics_t;

    // ── Error checking ────────────────────────────────────────────────────────

    static void check_call(typename AmdsmiBackend::status_t status, std::string_view func)
    {
        if(status == AmdsmiBackend::STATUS_SUCCESS) return;
        throw std::runtime_error(std::string(func) +
                                 " failed: " + AmdsmiBackend::status_to_string(status));
    }

    // ── Enumeration helper ────────────────────────────────────────────────────

    template <typename QueryFn>
    [[nodiscard]] static std::vector<typename AmdsmiBackend::processor_handle>
    enumerate_handles(QueryFn query)
    {
        std::vector<typename AmdsmiBackend::processor_handle> result;

        std::uint32_t socket_count = 0;
        check_call(AmdsmiBackend::get_socket_handles(&socket_count, nullptr),
                   "amdsmi_get_socket_handles (count)");

        if(socket_count == 0) return result;

        std::vector<typename AmdsmiBackend::socket_handle> sockets(socket_count);
        check_call(AmdsmiBackend::get_socket_handles(&socket_count, sockets.data()),
                   "amdsmi_get_socket_handles (data)");

        for(auto socket : sockets)
        {
            std::uint32_t count = 0;
            if(query(socket, &count, nullptr) != AmdsmiBackend::STATUS_SUCCESS ||
               count == 0)
                continue;

            std::vector<typename AmdsmiBackend::processor_handle> procs(count);
            check_call(query(socket, &count, procs.data()),
                       "amdsmi_get_processor_handles (data)");

            result.insert(result.end(), procs.begin(), procs.end());
        }

        return result;
    }

    // ── Metric conversion ─────────────────────────────────────────────────────

    static void convert_power(const gpu_metrics_t& raw, metrics& out)
    {
        out.current_socket_power = raw.current_socket_power;
        out.average_socket_power = raw.average_socket_power;
    }

    static void convert_temperature(const gpu_metrics_t& raw, metrics& out)
    {
        out.hotspot_temperature = raw.temperature_hotspot;
        out.edge_temperature    = raw.temperature_edge;
    }

    static void convert_activity(const gpu_metrics_t& raw, metrics& out)
    {
        out.gfx_activity = raw.average_gfx_activity;
        out.umc_activity = raw.average_umc_activity;
        out.mm_activity  = raw.average_mm_activity;
    }

    static void convert_xcp(const gpu_metrics_t& raw, metrics& out)
    {
        for(std::size_t xcp = 0; xcp < MAX_NUM_XCP; ++xcp)
        {
            std::copy(std::begin(raw.xcp_stats[xcp].vcn_busy),
                      std::end(raw.xcp_stats[xcp].vcn_busy),
                      out.xcp_stats[xcp].vcn_busy.begin());

            constexpr std::size_t copy_count =
                std::min(static_cast<std::size_t>(sizeof(raw.xcp_stats[0].jpeg_busy) /
                                                  sizeof(std::uint16_t)),
                         gpu::MAX_NUM_JPEG_V1);
            std::copy_n(std::begin(raw.xcp_stats[xcp].jpeg_busy), copy_count,
                        out.xcp_stats[xcp].jpeg_busy.begin());
        }

        std::copy(std::begin(raw.vcn_activity), std::end(raw.vcn_activity),
                  out.vcn_activity.begin());
        std::copy(std::begin(raw.jpeg_activity), std::end(raw.jpeg_activity),
                  out.jpeg_activity.begin());
    }

    static void convert_xgmi(const gpu_metrics_t& raw, metrics& out)
    {
        populate_if_supported(out.xgmi.link.width, raw.xgmi_link_width);
        populate_if_supported(out.xgmi.link.speed, raw.xgmi_link_speed);

        for(std::size_t idx = 0; idx < MAX_NUM_XGMI_LINKS; ++idx)
        {
            populate_if_supported(out.xgmi.data_acc.read[idx],
                                  raw.xgmi_read_data_acc[idx]);
            populate_if_supported(out.xgmi.data_acc.write[idx],
                                  raw.xgmi_write_data_acc[idx]);
        }
    }

    static void convert_pcie(const gpu_metrics_t& raw, metrics& out)
    {
        populate_if_supported(out.pcie.link.width, raw.pcie_link_width);
        populate_if_supported(out.pcie.link.speed, raw.pcie_link_speed);
        populate_if_supported(out.pcie.bandwidth.acc, raw.pcie_bandwidth_acc);
        populate_if_supported(out.pcie.bandwidth.inst, raw.pcie_bandwidth_inst);
    }

    static void convert_clocks(const gpu_metrics_t& raw, metrics& out)
    {
        constexpr auto clock_sentinel = static_cast<std::uint32_t>(0xFFFFU);
        populate_if_supported(out.gfx_clock_mhz,
                              static_cast<std::uint32_t>(raw.current_gfxclk),
                              clock_sentinel);
        populate_if_supported(out.mem_clock_mhz,
                              static_cast<std::uint32_t>(raw.current_uclk),
                              clock_sentinel);
    }

    // ── Members ───────────────────────────────────────────────────────────────

    AmdsmiBackend m_amdsmi{};  ///< Policy (stateless — zero overhead)
    typename AmdsmiBackend::processor_handle m_handle;
};

/**
 * @brief Factory exposing backend<amdsmi_backend> as the canonical backend type.
 *
 * The provider uses BackendFactory::backend_t for static lifecycle and enumeration.
 * backend<amdsmi_backend> provides initialize(), shutdown(), get_lib_version(),
 * and enumerate_*_handles() as statics — no instance needed.
 */
struct backend_factory
{
    using backend_t = backend<amdsmi_backend>;
};

}  // namespace rocprofsys::backends::amd_smi
