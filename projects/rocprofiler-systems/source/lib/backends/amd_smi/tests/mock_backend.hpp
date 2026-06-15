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
// Field names mirror amdsmi_* struct fields exactly so device_backend's
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
    // Power — convert_power
    std::uint32_t current_socket_power = 0;
    std::uint32_t average_socket_power = 0;

    // Temperature — convert_temperature
    std::uint16_t temperature_hotspot = 0;
    std::uint16_t temperature_edge    = 0;

    // Activity — convert_activity
    std::uint16_t average_gfx_activity = 0;
    std::uint16_t average_umc_activity = 0;
    std::uint16_t average_mm_activity  = 0;

    // XCP stats — C arrays required so std::begin/std::end produce pointer pairs
    // matching the real amdsmi_gpu_metrics_t layout used by convert_xcp.
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

    // XGMI — default sentinel = unsupported
    std::uint16_t xgmi_link_width =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint16_t xgmi_link_speed =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint64_t xgmi_read_data_acc
        [MAX_NUM_XGMI_LINKS] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::uint64_t xgmi_write_data_acc
        [MAX_NUM_XGMI_LINKS] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    // PCIe — default sentinel = unsupported
    std::uint16_t pcie_link_width =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint16_t pcie_link_speed =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint64_t pcie_bandwidth_acc  = METRIC_VALUE_NOT_SUPPORTED_64;
    std::uint64_t pcie_bandwidth_inst = METRIC_VALUE_NOT_SUPPORTED_64;

    // Clocks — default sentinel = unsupported
    std::uint16_t current_gfxclk =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
    std::uint16_t current_uclk =
        static_cast<std::uint16_t>(METRIC_VALUE_NOT_SUPPORTED_16);
};

// NOLINTEND(readability-identifier-naming)

// ── GMock class ─────────────────────────────────────────────────────────────
// Methods match the raw backend API: each returns a status_t and writes output
// via pointer parameters, mirroring the AMD SMI C API shape.

using mock_status_t = std::uint32_t;  // same underlying type as amdsmi_status_t

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

    // Per-device GPU queries (output via pointer — same shape as amdsmi_*)
    MOCK_METHOD(mock_status_t, get_metrics_info,
                (std::uint64_t handle, mock_gpu_metrics_t* out));
    MOCK_METHOD(mock_status_t, get_gpu_asic_info,
                (std::uint64_t handle, mock_asic_info_t* out));
    MOCK_METHOD(mock_status_t, get_memory_usage,
                (std::uint64_t handle, std::uint32_t type, std::uint64_t* out));
};

inline std::unique_ptr<gmock_backend_api> g_mock_backend;

// ── Policy struct ───────────────────────────────────────────────────────────
// Static methods delegate to g_mock_backend. device_backend<mock_backend>
// compiles without AMD SMI headers and exercises check_call() error paths.

struct mock_backend
{
    using status_t         = mock_status_t;
    using version_t        = mock_version_t;
    using socket_handle    = std::uint64_t;
    using processor_handle = std::uint64_t;
    using gpu_metrics_t    = mock_gpu_metrics_t;
    using asic_info_t      = mock_asic_info_t;
    using memory_type_t    = std::uint32_t;

    static constexpr status_t      STATUS_SUCCESS = 0;
    static constexpr memory_type_t MEM_TYPE_VRAM  = 0;

    [[nodiscard]] static std::string status_to_string(status_t status)
    {
        return "mock error " + std::to_string(static_cast<int>(status));
    }

    // Lifecycle
    static status_t init() { return g_mock_backend->init(); }
    static status_t shutdown() { return g_mock_backend->shutdown(); }
    static status_t get_version(version_t* out)
    {
        return g_mock_backend->get_version(out);
    }

    // Enumeration
    static status_t get_socket_handles(std::uint32_t* count, socket_handle* handles)
    {
        return g_mock_backend->get_socket_handles(count, handles);
    }

    static status_t get_processor_handles(socket_handle socket, std::uint32_t* count,
                                          processor_handle* handles)
    {
        return g_mock_backend->get_processor_handles(socket, count, handles);
    }

    // Per-device GPU queries
    static status_t get_metrics_info(processor_handle handle, gpu_metrics_t* out)
    {
        return g_mock_backend->get_metrics_info(handle, out);
    }

    static status_t get_gpu_asic_info(processor_handle handle, asic_info_t* out)
    {
        return g_mock_backend->get_gpu_asic_info(handle, out);
    }

    static status_t get_memory_usage(processor_handle handle, memory_type_t type,
                                     std::uint64_t* out)
    {
        return g_mock_backend->get_memory_usage(handle, type, out);
    }
};

}  // namespace rocprofsys::backends::amd_smi::testing
