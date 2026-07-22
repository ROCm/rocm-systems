// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/gpu_types.hpp"
#include "backends/amd_smi/nic_types.hpp"
#include "backends/amd_smi/sdma_feature.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys::backends::amd_smi
{

using gpu::asic_info;
using gpu::MAX_NUM_JPEG_V1;
using gpu::MAX_NUM_XCP;
using gpu::MAX_NUM_XGMI_LINKS;
using gpu::metrics;
using gpu::populate_if_supported;

template <typename T>
concept session_types = requires {
    typename T::processor_handle;
    typename T::asic_info_t;
    typename T::gpu_metrics_t;
    typename T::memory_type_t;
    typename T::temperature_type_t;
    typename T::temperature_metric_t;
    { T::sdma_supported } -> std::convertible_to<bool>;
    { T::ainic_feature_gate } -> std::convertible_to<bool>;
};

template <typename T>
concept session_gpu_queries =
    requires(T sess, typename T::processor_handle ph, typename T::asic_info_t* aip,
             typename T::memory_type_t mt, std::uint64_t* u64p,
             typename T::temperature_type_t tt, typename T::temperature_metric_t tm) {
        { T::MEM_TYPE_VRAM } -> std::convertible_to<typename T::memory_type_t>;
        { T::TEMP_CURRENT } -> std::convertible_to<typename T::temperature_metric_t>;
        {
            T::TEMPERATURE_TYPE_HOTSPOT
        } -> std::convertible_to<typename T::temperature_type_t>;
        {
            T::TEMPERATURE_TYPE_EDGE
        } -> std::convertible_to<typename T::temperature_type_t>;
        { sess.get_gpu_asic_info(ph, aip) };
        { sess.get_metrics_info(ph) } -> std::convertible_to<typename T::gpu_metrics_t>;
        { sess.get_memory_usage(ph, mt, u64p) };
        { sess.get_temp_metric(ph, tt, tm) } -> std::convertible_to<std::int64_t>;
    };

template <typename T>
concept sdma_session_types = requires { typename T::proc_info_t; };

template <typename T>
concept sdma_session_queries = requires(T sess, typename T::processor_handle ph) {
    { sess.probe_sdma_support(ph) } -> std::convertible_to<bool>;
    {
        sess.get_gpu_process_list(ph)
    } -> std::same_as<std::vector<typename T::proc_info_t>>;
};

template <typename T>
concept sdma_session_contract = sdma_session_types<T> && sdma_session_queries<T>;

template <typename T>
concept nic_session_types = requires {
    typename T::nic_asic_info_t;
    typename T::nic_port_info_t;
    typename T::nic_rdma_devices_info_t;
    typename T::nic_stat_t;
};

template <typename T>
concept nic_session_queries =
    requires(T sess, typename T::processor_handle ph, typename T::nic_asic_info_t* nap,
             typename T::nic_port_info_t* npp, typename T::nic_rdma_devices_info_t* ndp,
             std::uint8_t port_idx, std::uint32_t* cp, typename T::nic_stat_t* nsp) {
        { sess.get_nic_asic_info(ph, nap) };
        { sess.get_nic_port_info(ph, npp) };
        { sess.get_nic_rdma_dev_info(ph, ndp) };
        { sess.get_nic_rdma_port_statistics(ph, port_idx, cp, nsp) };
    };

template <typename T>
concept nic_session_contract = nic_session_types<T> && nic_session_queries<T>;

/**
 * @brief Concept that a Backend session type passed to device must satisfy.
 *
 * Enumerates every expression device<T> evaluates on its session pointer,
 * so a missing method is caught at the template boundary rather than deep inside
 * the template body.
 */
template <typename T>
concept backend_session_contract = session_types<T> && session_gpu_queries<T> &&
                                   (!T::sdma_supported || sdma_session_contract<T>) &&
                                   (!T::ainic_feature_gate || nic_session_contract<T>);

/**
 * @brief Per-device proxy — bridges a shared backend session to one device handle.
 *
 * Holds a @c shared_ptr<Backend> session and one processor handle. Forwards
 * per-device calls through the session (which throws on failure) and converts
 * raw AMD SMI structs to domain types.
 *
 * Lives in the PMC layer; the concrete @c Backend type is named only at the
 * integration point (sampler.cpp).
 *
 * @tparam Backend  Session type satisfying @c backend_session_contract
 *                  (e.g. @c backends::amd_smi::backend<wrapper>).
 */
template <backend_session_contract Backend>
class device
{
public:
    static constexpr bool sdma_supported = Backend::sdma_supported;

    device(std::shared_ptr<Backend> session, Backend::processor_handle handle)
    : m_session{ std::move(session) }
    , m_handle{ handle }
    {}

    // ── Per-device GPU queries ────────────────────────────────────────────────

    [[nodiscard]] asic_info get_gpu_asic_info() const
    {
        typename Backend::asic_info_t raw{};
        m_session->get_gpu_asic_info(m_handle, &raw);
        return { raw.market_name, raw.vendor_name };
    }

    [[nodiscard]] metrics get_metrics() const
    {
        const auto raw = m_session->get_metrics_info(m_handle);

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

    [[nodiscard]] std::int64_t get_hotspot_temperature() const
    {
        return m_session->get_temp_metric(m_handle, Backend::TEMPERATURE_TYPE_HOTSPOT,
                                          Backend::TEMP_CURRENT);
    }

    [[nodiscard]] std::int64_t get_edge_temperature() const
    {
        return m_session->get_temp_metric(m_handle, Backend::TEMPERATURE_TYPE_EDGE,
                                          Backend::TEMP_CURRENT);
    }

    [[nodiscard]] std::uint64_t get_memory_usage() const
    {
        std::uint64_t usage = 0;
        m_session->get_memory_usage(m_handle, Backend::MEM_TYPE_VRAM, &usage);
        return usage;
    }

    // One-time GPU-level SDMA capability probe (available = library has the API,
    // supported = this GPU actually provides data).  Always returns false when
    // sdma_supported == false so callers need no #ifdef.
    [[nodiscard]] bool probe_sdma_gpu_support() const noexcept
    {
        if constexpr(sdma_supported) return m_session->probe_sdma_support(m_handle);
        return false;
    }

    [[nodiscard]] std::uint64_t get_raw_sdma_usage() const
    {
        if constexpr(Backend::sdma_supported)
        {
            std::uint64_t cumulative = 0;
            auto          procs      = m_session->get_gpu_process_list(m_handle);
            for(const auto& proc : procs)
                cumulative += proc.sdma_usage;
            return cumulative;
        }
        return 0;
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    // ── Per-device NIC queries ────────────────────────────────────────────────

    [[nodiscard]] nic::asic_info get_nic_asic_info() const
    {
        typename Backend::nic_asic_info_t raw{};
        m_session->get_nic_asic_info(m_handle, &raw);
        return { raw.product_name, raw.vendor_name };
    }

    [[nodiscard]] nic::port_info get_nic_port_info() const
    {
        typename Backend::nic_port_info_t raw{};
        m_session->get_nic_port_info(m_handle, &raw);
        if(raw.num_ports == 0) return {};
        return { raw.ports[0].netdev };
    }

    [[nodiscard]] nic::rdma_info get_nic_rdma_info() const
    {
        auto raw = std::make_unique<typename Backend::nic_rdma_devices_info_t>();
        m_session->get_nic_rdma_dev_info(m_handle, raw.get());
        if(raw->num_rdma_dev == 0) return { 0 };
        return { raw->rdma_dev_info[0].num_rdma_ports };
    }

    [[nodiscard]] std::vector<nic::stat_entry> get_nic_rdma_port_statistics(
        std::uint8_t rdma_port_idx) const
    {
        std::uint32_t count = 0;
        m_session->get_nic_rdma_port_statistics(m_handle, rdma_port_idx, &count, nullptr);

        if(count == 0) return {};

        std::vector<typename Backend::nic_stat_t> raw_stats(count);
        m_session->get_nic_rdma_port_statistics(m_handle, rdma_port_idx, &count,
                                                raw_stats.data());

        std::vector<nic::stat_entry> result;
        result.reserve(count);
        for(const auto& stat : raw_stats)
            result.emplace_back(nic::stat_entry{ stat.name, stat.value });
        return result;
    }
#endif

private:
    using gpu_metrics_t = typename Backend::gpu_metrics_t;

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
        constexpr auto sentinel = static_cast<std::uint32_t>(0xFFFFU);
        populate_if_supported(out.gfx_clock_mhz,
                              static_cast<std::uint32_t>(raw.current_gfxclk), sentinel);
        populate_if_supported(out.mem_clock_mhz,
                              static_cast<std::uint32_t>(raw.current_uclk), sentinel);
    }

    // ── Members ───────────────────────────────────────────────────────────────

    std::shared_ptr<Backend>           m_session;
    typename Backend::processor_handle m_handle;
};

}  // namespace rocprofsys::backends::amd_smi
