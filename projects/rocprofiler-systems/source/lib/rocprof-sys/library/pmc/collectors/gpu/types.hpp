// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/gpu_types.hpp"
#include "library/pmc/common/types.hpp"

#include <cstdint>

namespace rocprofsys::pmc::collectors::gpu
{

// Data types are owned by the backend layer (the producer); re-exported here so
// pmc consumers keep their existing pmc::collectors::gpu::* spellings.
namespace backend = ::rocprofsys::backends::amd_smi::gpu;

using backend::asic_info;
using backend::is_metric_supported;
using backend::metrics;
using backend::populate_if_supported;

using backend::MAX_NUM_JPEG;
using backend::MAX_NUM_JPEG_V1;
using backend::MAX_NUM_VCN;
using backend::MAX_NUM_XCP;
using backend::MAX_NUM_XGMI_LINKS;
using backend::METRIC_VALUE_NOT_SUPPORTED_16;
using backend::METRIC_VALUE_NOT_SUPPORTED_64;

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
 *   - gfx_clock = 15
 *   - mem_clock = 16
 */
union enabled_metrics
{
    struct
    {
        std::uint32_t current_socket_power : 1;
        std::uint32_t average_socket_power : 1;
        std::uint32_t memory_usage         : 1;
        std::uint32_t hotspot_temperature  : 1;
        std::uint32_t edge_temperature     : 1;
        std::uint32_t gfx_activity         : 1;
        std::uint32_t umc_activity         : 1;
        std::uint32_t mm_activity          : 1;
        std::uint32_t vcn_activity         : 1;  // Device-level VCN activity
        std::uint32_t jpeg_activity        : 1;  // Device-level JPEG activity
        std::uint32_t vcn_busy             : 1;  // Per-XCP VCN busy
        std::uint32_t jpeg_busy            : 1;  // Per-XCP JPEG busy
        std::uint32_t xgmi                 : 1;
        std::uint32_t pcie                 : 1;
        std::uint32_t sdma_usage           : 1;
        std::uint32_t gfx_clock            : 1;  // current_gfxclk (MHz)
        std::uint32_t mem_clock            : 1;  // current_uclk (MHz)
    } bits;
    std::uint32_t value = 0;
};

/// enabled_metrics bit masks, in the bit order documented above.
namespace metric_bit
{
inline constexpr std::uint32_t k_current_socket_power = 1U << 0U;
inline constexpr std::uint32_t k_average_socket_power = 1U << 1U;
inline constexpr std::uint32_t k_memory_usage         = 1U << 2U;
inline constexpr std::uint32_t k_hotspot_temperature  = 1U << 3U;
inline constexpr std::uint32_t k_edge_temperature     = 1U << 4U;
inline constexpr std::uint32_t k_gfx_activity         = 1U << 5U;
inline constexpr std::uint32_t k_umc_activity         = 1U << 6U;
inline constexpr std::uint32_t k_mm_activity          = 1U << 7U;
inline constexpr std::uint32_t k_vcn_activity         = 1U << 8U;
inline constexpr std::uint32_t k_jpeg_activity        = 1U << 9U;
inline constexpr std::uint32_t k_vcn_busy             = 1U << 10U;
inline constexpr std::uint32_t k_jpeg_busy            = 1U << 11U;
inline constexpr std::uint32_t k_xgmi                 = 1U << 12U;
inline constexpr std::uint32_t k_pcie                 = 1U << 13U;
inline constexpr std::uint32_t k_sdma_usage           = 1U << 14U;
inline constexpr std::uint32_t k_gfx_clock            = 1U << 15U;
inline constexpr std::uint32_t k_mem_clock            = 1U << 16U;
}  // namespace metric_bit

}  // namespace rocprofsys::pmc::collectors::gpu
