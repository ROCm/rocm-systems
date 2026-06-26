// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/gpu_types.hpp"

#include <gmock/gmock.h>

#include <cstdint>
#include <memory>
#include <string>

namespace rocprofsys::backends::amd_smi::testing
{

using gpu::MAX_NUM_JPEG;
using gpu::MAX_NUM_JPEG_V1;
using gpu::MAX_NUM_VCN;
using gpu::MAX_NUM_XCP;
using gpu::MAX_NUM_XGMI_LINKS;
using gpu::METRIC_VALUE_NOT_SUPPORTED_16;
using gpu::METRIC_VALUE_NOT_SUPPORTED_64;

// ── Mock raw types ──────────────────────────────────────────────────────────
// Field names mirror amdsmi_* struct fields exactly so backend_proxy's
// convert_* methods compile.  snake_case is intentional; suppress naming lint.
// NOLINTBEGIN(readability-identifier-naming)

struct mock_asic_info_t
{
    const char* market_name = "";
    const char* vendor_name = "";
};

struct mock_version_t
{
    std::uint32_t major   = 0;
    std::uint32_t minor   = 0;
    std::uint32_t release = 0;
    const char*   build   = nullptr;
};

struct mock_gpu_metrics_t
{
    std::uint32_t current_socket_power = 0;
    std::uint32_t average_socket_power = 0;
    std::uint16_t temperature_hotspot  = 0;
    std::uint16_t temperature_edge     = 0;
    std::uint16_t average_gfx_activity = 0;
    std::uint16_t average_umc_activity = 0;
    std::uint16_t average_mm_activity  = 0;

    struct xcp_stat_t
    {
        std::uint16_t
            vcn_busy[MAX_NUM_VCN] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
        std::uint16_t
            jpeg_busy[MAX_NUM_JPEG_V1] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    } xcp_stats[MAX_NUM_XCP] = {};            // NOLINT(cppcoreguidelines-avoid-c-arrays)

    std::uint16_t
        vcn_activity[MAX_NUM_VCN] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::uint16_t
        jpeg_activity[MAX_NUM_JPEG] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    std::uint16_t xgmi_link_width =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint16_t xgmi_link_speed =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint64_t xgmi_read_data_acc
        [MAX_NUM_XGMI_LINKS] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::uint64_t xgmi_write_data_acc
        [MAX_NUM_XGMI_LINKS] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    std::uint16_t pcie_link_width =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint16_t pcie_link_speed =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint64_t pcie_bandwidth_acc  = METRIC_VALUE_NOT_SUPPORTED_64;
    std::uint64_t pcie_bandwidth_inst = METRIC_VALUE_NOT_SUPPORTED_64;

    std::uint16_t current_gfxclk =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint16_t current_uclk =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
};

// NOLINTEND(readability-identifier-naming)

// ── Mock SDMA / NIC raw types ───────────────────────────────────────────────
// NOLINTBEGIN(readability-identifier-naming)
struct mock_proc_info_t
{
    std::uint64_t sdma_usage = 0;
};

struct mock_nic_asic_info_t
{
    const char* product_name = "";
    const char* vendor_name  = "";
};

struct mock_nic_port_t
{
    const char* netdev = "";
};

struct mock_nic_port_info_t
{
    std::uint32_t   num_ports = 0;
    mock_nic_port_t ports[1]  = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
};

struct mock_rdma_dev_info_t
{
    std::uint8_t num_rdma_ports = 0;
};

struct mock_nic_rdma_devices_info_t
{
    std::uint32_t num_rdma_dev = 0;
    mock_rdma_dev_info_t
        rdma_dev_info[1] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
};

struct mock_nic_stat_t
{
    const char*   name  = "";
    std::uint64_t value = 0;
};

// NOLINTEND(readability-identifier-naming)

using mock_status_t      = std::uint32_t;
using mock_temp_type_t   = std::uint32_t;
using mock_temp_metric_t = std::uint32_t;

// ── GMock class ─────────────────────────────────────────────────────────────
// Lifecycle/enumeration methods are mocked as regular instance methods
// (the real backend exposes them as static, but GMock needs instances).
// Per-device methods take an explicit handle (matches backend<> instance API).

struct gmock_backend_api
{
    // Lifecycle
    MOCK_METHOD(mock_status_t, init, ());
    MOCK_METHOD(mock_status_t, shutdown, ());
    MOCK_METHOD(mock_status_t, get_version, (mock_version_t * out));

    // Enumeration
    MOCK_METHOD(mock_status_t, get_socket_handles,
                (std::uint32_t * count, std::uint64_t* handles));
    MOCK_METHOD(mock_status_t, get_processor_handles,
                (std::uint64_t socket, std::uint32_t* count, std::uint64_t* handles));
    MOCK_METHOD(mock_status_t, get_processor_handles_by_type,
                (std::uint64_t socket, std::uint32_t type, std::uint64_t* handles,
                 std::uint32_t* count));

    // Per-device forwarding (explicit handle)
    MOCK_METHOD(mock_status_t, get_metrics_info,
                (std::uint64_t handle, mock_gpu_metrics_t* out));
    MOCK_METHOD(mock_status_t, get_gpu_asic_info,
                (std::uint64_t handle, mock_asic_info_t* out));
    MOCK_METHOD(mock_status_t, get_memory_usage,
                (std::uint64_t handle, std::uint32_t type, std::uint64_t* out));
    MOCK_METHOD(mock_status_t, get_temp_metric,
                (std::uint64_t handle, mock_temp_type_t sensor_type,
                 mock_temp_metric_t metric, std::int64_t* temperature));
    MOCK_METHOD(mock_status_t, get_gpu_process_list,
                (std::uint64_t handle, std::uint32_t* count, mock_proc_info_t* list));
    MOCK_METHOD(mock_status_t, get_nic_asic_info,
                (std::uint64_t handle, mock_nic_asic_info_t* out));
    MOCK_METHOD(mock_status_t, get_nic_port_info,
                (std::uint64_t handle, mock_nic_port_info_t* out));
    MOCK_METHOD(mock_status_t, get_nic_rdma_dev_info,
                (std::uint64_t handle, mock_nic_rdma_devices_info_t* out));
    MOCK_METHOD(mock_status_t, get_nic_rdma_port_statistics,
                (std::uint64_t handle, std::uint8_t port_idx, std::uint32_t* count,
                 mock_nic_stat_t* stats));
};

inline std::unique_ptr<gmock_backend_api> g_mock_backend;

// ── Policy struct ───────────────────────────────────────────────────────────
// Satisfies the AmdsmiBackend concept for backend<mock_backend>.
// Static methods delegate to g_mock_backend for lifecycle/enumeration.
// Instance methods (per-device forwarding) also delegate via g_mock_backend.

struct mock_backend
{
    using status_t                = mock_status_t;
    using version_t               = mock_version_t;
    using socket_handle           = std::uint64_t;
    using processor_handle        = std::uint64_t;
    using gpu_metrics_t           = mock_gpu_metrics_t;
    using asic_info_t             = mock_asic_info_t;
    using memory_type_t           = std::uint32_t;
    using proc_info_t             = mock_proc_info_t;
    using processor_type          = std::uint32_t;
    using nic_asic_info_t         = mock_nic_asic_info_t;
    using nic_port_info_t         = mock_nic_port_info_t;
    using nic_rdma_devices_info_t = mock_nic_rdma_devices_info_t;
    using nic_stat_t              = mock_nic_stat_t;
    using temperature_type_t      = mock_temp_type_t;
    using temperature_metric_t    = mock_temp_metric_t;

    static constexpr temperature_metric_t TEMP_CURRENT             = 0;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_HOTSPOT = 1;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_EDGE    = 0;

    static constexpr processor_type NIC_PROCESSOR_TYPE = 5;

    static constexpr bool          sdma_supported     = true;
    static constexpr bool          ainic_feature_gate = true;
    static constexpr status_t      STATUS_SUCCESS     = 0;
    static constexpr memory_type_t MEM_TYPE_VRAM      = 0;

    [[nodiscard]] static std::string status_to_string(status_t status)
    {
        return "mock error " + std::to_string(static_cast<int>(status));
    }

    // Lifecycle (static)
    static status_t init() { return g_mock_backend->init(); }
    static status_t shutdown() { return g_mock_backend->shutdown(); }
    static status_t get_version(version_t* out)
    {
        return g_mock_backend->get_version(out);
    }

    // Enumeration (static)
    static status_t get_socket_handles(std::uint32_t* count, socket_handle* handles)
    {
        return g_mock_backend->get_socket_handles(count, handles);
    }

    static status_t get_processor_handles(socket_handle socket, std::uint32_t* count,
                                          processor_handle* handles)
    {
        return g_mock_backend->get_processor_handles(socket, count, handles);
    }

    static status_t get_processor_handles_by_type(socket_handle     socket,
                                                  processor_type    type,
                                                  processor_handle* handles,
                                                  std::uint32_t*    count)
    {
        return g_mock_backend->get_processor_handles_by_type(socket, type, handles,
                                                             count);
    }

    // Per-device forwarding (instance — explicit handle, matches backend<> API)
    status_t get_metrics_info(processor_handle handle, gpu_metrics_t* out) const
    {
        return g_mock_backend->get_metrics_info(handle, out);
    }

    status_t get_gpu_asic_info(processor_handle handle, asic_info_t* out) const
    {
        return g_mock_backend->get_gpu_asic_info(handle, out);
    }

    status_t get_memory_usage(processor_handle handle, memory_type_t type,
                              std::uint64_t* out) const
    {
        return g_mock_backend->get_memory_usage(handle, type, out);
    }

    status_t get_temp_metric(processor_handle handle, temperature_type_t sensor_type,
                             temperature_metric_t metric, std::int64_t* temperature) const
    {
        return g_mock_backend->get_temp_metric(handle, sensor_type, metric, temperature);
    }

    status_t get_gpu_process_list(processor_handle handle, std::uint32_t* count,
                                  proc_info_t* list) const
    {
        return g_mock_backend->get_gpu_process_list(handle, count, list);
    }

    status_t get_nic_asic_info(processor_handle handle, nic_asic_info_t* out) const
    {
        return g_mock_backend->get_nic_asic_info(handle, out);
    }

    status_t get_nic_port_info(processor_handle handle, nic_port_info_t* out) const
    {
        return g_mock_backend->get_nic_port_info(handle, out);
    }

    status_t get_nic_rdma_dev_info(processor_handle         handle,
                                   nic_rdma_devices_info_t* out) const
    {
        return g_mock_backend->get_nic_rdma_dev_info(handle, out);
    }

    status_t get_nic_rdma_port_statistics(processor_handle handle, std::uint8_t port_idx,
                                          std::uint32_t* count, nic_stat_t* stats) const
    {
        return g_mock_backend->get_nic_rdma_port_statistics(handle, port_idx, count,
                                                            stats);
    }
};

}  // namespace rocprofsys::backends::amd_smi::testing
