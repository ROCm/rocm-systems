// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/common/types.hpp"
#include "library/pmc/gpu/metric_descriptors.hpp"

#include <cstdint>

#if ROCPROFSYS_USE_ROCM > 0
#    include <spdlog/fmt/fmt.h>

#    include <array>
#    include <stdexcept>

#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace pmc
{
namespace gpu
{

// Sentinel values used by AMD SMI to indicate unsupported/unavailable metrics
// These match the AMD SMI library's convention for indicating N/A values
constexpr uint16_t METRIC_VALUE_NOT_SUPPORTED    = 0xffff;              // 16-bit sentinel
constexpr uint64_t METRIC_VALUE_NOT_SUPPORTED_64 = 0xffffffffffffffff;  // 64-bit sentinel

// enabled_metrics class and metric_id enum are now defined in metric_descriptors.hpp
// This provides type-safe metric handling with explicit bit positions.
// See metric_descriptors.hpp for the full API.

#if ROCPROFSYS_USE_ROCM > 0

// Ensure we have the correct max values defined for older AMD SMI versions
#    ifndef AMDSMI_MAX_NUM_VCN
#        define AMDSMI_MAX_NUM_VCN 4
#    endif

#    ifndef AMDSMI_MAX_NUM_JPEG
#        define AMDSMI_MAX_NUM_JPEG 32
#    endif

#    ifndef AMDSMI_MAX_NUM_XCP
#        define AMDSMI_MAX_NUM_XCP 8
#    endif

// AMDSMI_MAX_NUM_JPEG_ENG_V1 (40) was introduced in gpu metrics v1.8
// Fall back to AMDSMI_MAX_NUM_JPEG (32) for older AMD SMI versions
#    ifdef AMDSMI_MAX_NUM_JPEG_ENG_V1
#        define ROCPROFSYS_MAX_NUM_JPEG_ENGINES AMDSMI_MAX_NUM_JPEG_ENG_V1
#    else
#        define ROCPROFSYS_MAX_NUM_JPEG_ENGINES AMDSMI_MAX_NUM_JPEG
#    endif

struct metrics
{
    struct xcp_metrics
    {
        std::array<uint16_t, ROCPROFSYS_MAX_NUM_JPEG_ENGINES> jpeg_busy;
        std::array<uint16_t, AMDSMI_MAX_NUM_VCN>              vcn_busy;
    };

    uint32_t                                    current_socket_power = 0;
    uint32_t                                    average_socket_power = 0;
    uint64_t                                    memory_usage         = 0;
    int64_t                                     hotspot_temperature  = 0;
    int64_t                                     edge_temperature     = 0;
    uint32_t                                    gfx_activity         = 0;
    uint32_t                                    umc_activity         = 0;
    uint32_t                                    mm_activity          = 0;
    std::array<xcp_metrics, AMDSMI_MAX_NUM_XCP> xcp_stats;

    // Device-level VCN/JPEG activity (Radeon GPUs)
    std::array<uint16_t, AMDSMI_MAX_NUM_VCN>  vcn_activity  = {};
    std::array<uint16_t, AMDSMI_MAX_NUM_JPEG> jpeg_activity = {};

    struct
    {
        struct
        {
            uint16_t width = 0;
            uint16_t speed = 0;
        } link;

        struct
        {
            std::array<uint64_t, AMDSMI_MAX_NUM_XGMI_LINKS> read;
            std::array<uint64_t, AMDSMI_MAX_NUM_XGMI_LINKS> write;
        } data_acc;
    } xgmi;

    struct
    {
        struct
        {
            uint16_t width = 0;
            uint16_t speed = 0;
        } link;

        struct
        {
            uint64_t acc  = 0;
            uint64_t inst = 0;
        } bandwidth;
    } pcie;
};

inline void
check_status(amdsmi_status_t status, const char* error_message)
{
    if(status != AMDSMI_STATUS_SUCCESS)
    {
        throw std::runtime_error(
            fmt::format("{} AMD SMI Error code: {}", error_message, status));
    }
}

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace gpu
}  // namespace pmc
}  // namespace rocprofsys
