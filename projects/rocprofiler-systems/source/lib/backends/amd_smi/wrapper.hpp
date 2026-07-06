// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/sdma_feature.hpp"

#include <cstdint>
#include <string>

#include <amd_smi/amdsmi.h>

namespace rocprofsys::backends::amd_smi
{

/**
 * @brief 1:1 thin wrapper around the AMD SMI C API.
 *
 * Every public method maps directly to one amdsmi_* call and returns its raw
 * status code.  No error checking, no exceptions — this struct is a pure
 * dependency-injection seam that lets upper layers (backend<wrapper>get_temp_metric)
 * swap in a mock for testing without touching AMD SMI headers.
 *
 * Type aliases hide amdsmi_* names from callers: use wrapper::processor_handle,
 * wrapper::status_t, wrapper::STATUS_SUCCESS, etc.
 */
struct wrapper
{
private:
    // this type do not start with amdsmi_
    using amdsmi_processor_type = ::processor_type_t;

public:
    // ── Type aliases ──────────────────────────────────────────────────────────
    using status_t             = amdsmi_status_t;
    using version_t            = amdsmi_version_t;
    using socket_handle        = amdsmi_socket_handle;
    using processor_handle     = amdsmi_processor_handle;
    using gpu_metrics_t        = amdsmi_gpu_metrics_t;
    using asic_info_t          = amdsmi_asic_info_t;
    using memory_type_t        = amdsmi_memory_type_t;
    using temperature_type_t   = amdsmi_temperature_type_t;
    using temperature_metric_t = amdsmi_temperature_metric_t;
    using processor_type       = amdsmi_processor_type;
    using init_flags_t         = amdsmi_init_flags_t;

    static constexpr temperature_metric_t TEMP_CURRENT = AMDSMI_TEMP_CURRENT;
    static constexpr temperature_type_t   TEMPERATURE_TYPE_HOTSPOT =
        AMDSMI_TEMPERATURE_TYPE_HOTSPOT;
    static constexpr temperature_type_t TEMPERATURE_TYPE_EDGE =
        AMDSMI_TEMPERATURE_TYPE_EDGE;

    static constexpr init_flags_t INIT_AMD_GPUS = AMDSMI_INIT_AMD_GPUS;

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    using nic_asic_info_t         = amdsmi_nic_asic_info_t;
    using nic_port_info_t         = amdsmi_nic_port_info_t;
    using nic_rdma_devices_info_t = amdsmi_nic_rdma_devices_info_t;
    using nic_stat_t              = amdsmi_nic_stat_t;

    static constexpr processor_type NIC_PROCESSOR_TYPE = AMDSMI_PROCESSOR_TYPE_AMD_NIC;

    static constexpr init_flags_t INIT_AMD_NICS      = AMDSMI_INIT_AMD_NICS;
    static constexpr bool         ainic_feature_gate = true;
#else
    static constexpr bool ainic_feature_gate = false;
#endif

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    using proc_info_t                    = amdsmi_proc_info_t;
    static constexpr bool sdma_supported = true;
#else
    static constexpr bool sdma_supported = false;
#endif

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    static constexpr init_flags_t default_init_flags =
        static_cast<init_flags_t>(INIT_AMD_GPUS | INIT_AMD_NICS);
#else
    static constexpr init_flags_t default_init_flags = INIT_AMD_GPUS;
#endif

    // ── Status constants ──────────────────────────────────────────────────────
    static constexpr status_t      STATUS_SUCCESS = AMDSMI_STATUS_SUCCESS;
    static constexpr memory_type_t MEM_TYPE_VRAM  = AMDSMI_MEM_TYPE_VRAM;

    // ── Status helper ─────────────────────────────────────────────────────────
    [[nodiscard]] static std::string status_to_string(status_t status)
    {
        const char* msg = nullptr;
        if(amdsmi_status_code_to_string(status, &msg) == AMDSMI_STATUS_SUCCESS &&
           msg != nullptr)
        {
            return { msg };
        }
        return "error code: " + std::to_string(static_cast<int>(status));
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    static status_t init(init_flags_t flags = default_init_flags)
    {
        return amdsmi_init(flags);
    }

    static status_t shutdown() { return amdsmi_shut_down(); }

    // ── Library info ──────────────────────────────────────────────────────────
    static status_t get_version(version_t* out) { return amdsmi_get_lib_version(out); }

    // ── Socket / processor enumeration ───────────────────────────────────────
    static status_t get_socket_handles(std::uint32_t* count, socket_handle* handles)
    {
        return amdsmi_get_socket_handles(count, handles);
    }

    static status_t get_processor_handles(socket_handle socket, std::uint32_t* count,
                                          processor_handle* handles)
    {
        return amdsmi_get_processor_handles(socket, count, handles);
    }

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    static status_t get_processor_handles_by_type(socket_handle     socket,
                                                  processor_type    type,
                                                  processor_handle* handles,
                                                  std::uint32_t*    count)
    {
        return amdsmi_get_processor_handles_by_type(socket, type, handles, count);
    }
#endif

    // ── Per-device GPU queries ────────────────────────────────────────────────
    static status_t get_metrics_info(processor_handle handle, gpu_metrics_t* out)
    {
        return amdsmi_get_gpu_metrics_info(handle, out);
    }

    static status_t get_gpu_asic_info(processor_handle handle, asic_info_t* out)
    {
        return amdsmi_get_gpu_asic_info(handle, out);
    }

    static status_t get_memory_usage(processor_handle handle, memory_type_t type,
                                     std::uint64_t* out)
    {
        return amdsmi_get_gpu_memory_usage(handle, type, out);
    }

    static status_t get_temp_metric(processor_handle     handle,
                                    temperature_type_t   sensor_type,
                                    temperature_metric_t metric,
                                    std::int64_t*        temperature)
    {
        return amdsmi_get_temp_metric(handle, sensor_type, metric, temperature);
    }

#if defined(AMD_SMI_SDMA_SUPPORTED) && AMD_SMI_SDMA_SUPPORTED == 1
    static status_t get_gpu_process_list(processor_handle handle, std::uint32_t* count,
                                         proc_info_t* list)
    {
        return amdsmi_get_gpu_process_list(handle, count, list);
    }
#endif

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
    // ── Per-device NIC queries ────────────────────────────────────────────────
    static status_t get_nic_asic_info(processor_handle handle, nic_asic_info_t* out)
    {
        return amdsmi_get_nic_asic_info(handle, out);
    }

    static status_t get_nic_port_info(processor_handle handle, nic_port_info_t* out)
    {
        return amdsmi_get_nic_port_info(handle, out);
    }

    static status_t get_nic_rdma_dev_info(processor_handle         handle,
                                          nic_rdma_devices_info_t* out)
    {
        return amdsmi_get_nic_rdma_dev_info(handle, out);
    }

    static status_t get_nic_rdma_port_statistics(processor_handle handle,
                                                 std::uint8_t     port_idx,
                                                 std::uint32_t* count, nic_stat_t* stats)
    {
        return amdsmi_get_nic_rdma_port_statistics(handle, port_idx, count, stats);
    }
#endif
};

}  // namespace rocprofsys::backends::amd_smi
