// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/common/types.hpp"

#include <array>
#include <cstdint>

#include <amd_smi/amdsmi.h>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

// Sentinel value used by AMD SMI to indicate unsupported/unavailable 64-bit metrics
constexpr uint64_t METRIC_VALUE_NOT_SUPPORTED_64 = 0xffffffffffffffff;

/**
 * @brief Bitfield union for selecting which AMD SMI metrics to collect.
 *
 * Bit positions (for value access):
 *   - current_socket_power = 0
 *   - average_socket_power = 1
 *   - memory_usage = 2
 *   - hotspot_temperature = 3
 *   - edge_temperature = 4
 *   - gfx_activity = 5
 *   - umc_activity = 6
 *   - mm_activity = 7
 *   - vcn_activity = 8   (Device-level, Radeon GPUs)
 *   - jpeg_activity = 9  (Device-level, Radeon GPUs)
 *   - vcn_busy = 10      (Per-XCP, MI300 series)
 *   - jpeg_busy = 11     (Per-XCP, MI300 series)
 *   - xgmi = 12
 *   - pcie = 13
 *   - sdma_usage = 14
 */
union enabled_metrics
{
    struct
    {
        uint32_t current_socket_power : 1;
        uint32_t average_socket_power : 1;
        uint32_t memory_usage         : 1;
        uint32_t hotspot_temperature  : 1;
        uint32_t edge_temperature     : 1;
        uint32_t gfx_activity         : 1;
        uint32_t umc_activity         : 1;
        uint32_t mm_activity          : 1;
        uint32_t vcn_activity         : 1;  // Device-level VCN activity
        uint32_t jpeg_activity        : 1;  // Device-level JPEG activity
        uint32_t vcn_busy             : 1;  // Per-XCP VCN busy
        uint32_t jpeg_busy            : 1;  // Per-XCP JPEG busy
        uint32_t xgmi                 : 1;
        uint32_t pcie                 : 1;
        uint32_t sdma_usage           : 1;
    } bits;
    uint32_t value = 0;
};

// Get the actual JPEG engine count from the AMD SMI structure at compile time.
// This ensures compatibility across ROCm versions where the jpeg_busy array size
// may differ (32 in ROCm 6.x vs 40 in ROCm 7.x).
constexpr size_t ROCPROFSYS_AMDSMI_JPEG_ENGINE_COUNT =
    sizeof(amdsmi_gpu_xcp_metrics_t::jpeg_busy) / sizeof(uint16_t);

#ifndef AMDSMI_MAX_NUM_VCN
#    define AMDSMI_MAX_NUM_VCN 4
#endif

#ifndef AMDSMI_MAX_NUM_JPEG
#    define AMDSMI_MAX_NUM_JPEG 32
#endif

#ifndef AMDSMI_MAX_NUM_XCP
#    define AMDSMI_MAX_NUM_XCP 8
#endif

struct metrics
{
    struct xcp_metrics
    {
        std::array<uint16_t, ROCPROFSYS_AMDSMI_JPEG_ENGINE_COUNT> jpeg_busy = {};
        std::array<uint16_t, AMDSMI_MAX_NUM_VCN>                  vcn_busy  = {};
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
            std::array<uint64_t, AMDSMI_MAX_NUM_XGMI_LINKS> read  = {};
            std::array<uint64_t, AMDSMI_MAX_NUM_XGMI_LINKS> write = {};
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

    uint32_t sdma_usage = 0;  // SDMA utilization percentage (0-100)
};

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
