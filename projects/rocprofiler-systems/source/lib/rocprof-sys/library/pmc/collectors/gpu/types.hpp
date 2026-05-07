// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/common/types.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

// Sentinel values used by AMD SMI to indicate unsupported/unavailable metrics.
// AMD SMI returns these per-field; the POD widens some fields to 32/64 bits so the
// 16-bit sentinel must be checked explicitly when reading from the wider POD field.
constexpr std::uint32_t METRIC_VALUE_NOT_SUPPORTED_16 = 0xFFFF;
constexpr std::uint64_t METRIC_VALUE_NOT_SUPPORTED_64 = 0xFFFFFFFFFFFFFFFFULL;

constexpr size_t MAX_NUM_VCN        = 4;
constexpr size_t MAX_NUM_JPEG       = 32;
constexpr size_t MAX_NUM_JPEG_V1    = 40;
constexpr size_t MAX_NUM_XCP        = 8;
constexpr size_t MAX_NUM_XGMI_LINKS = 8;

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
    } bits;
    std::uint32_t value = 0;
};

/**
 * @brief GPU ASIC identification info.
 */
struct asic_info
{
    std::string product_name;
    std::string vendor_name;
};

struct metrics
{
    struct xcp_metrics
    {
        std::array<std::uint16_t, MAX_NUM_JPEG_V1> jpeg_busy = {};
        std::array<std::uint16_t, MAX_NUM_VCN>     vcn_busy  = {};
    };

    std::uint32_t                        current_socket_power = 0;
    std::uint32_t                        average_socket_power = 0;
    std::uint64_t                        memory_usage         = 0;
    std::uint32_t                        hotspot_temperature  = 0;
    std::uint32_t                        edge_temperature     = 0;
    std::uint32_t                        gfx_activity         = 0;
    std::uint32_t                        umc_activity         = 0;
    std::uint32_t                        mm_activity          = 0;
    std::array<xcp_metrics, MAX_NUM_XCP> xcp_stats;

    // Device-level VCN/JPEG activity (Radeon GPUs)
    std::array<std::uint16_t, MAX_NUM_VCN>  vcn_activity  = {};
    std::array<std::uint16_t, MAX_NUM_JPEG> jpeg_activity = {};

    struct
    {
        struct
        {
            std::uint16_t width = 0;
            std::uint16_t speed = 0;
        } link;

        struct
        {
            std::array<std::uint64_t, MAX_NUM_XGMI_LINKS> read  = {};
            std::array<std::uint64_t, MAX_NUM_XGMI_LINKS> write = {};
        } data_acc;
    } xgmi;

    struct
    {
        struct
        {
            std::uint16_t width = 0;
            std::uint16_t speed = 0;
        } link;

        struct
        {
            std::uint64_t acc  = 0;
            std::uint64_t inst = 0;
        } bandwidth;
    } pcie;

    std::uint32_t sdma_usage = 0;  // SDMA utilization percentage (0-100)
};

template <typename T>
[[nodiscard]] constexpr bool
is_metric_supported(T value, T invalid_sentinel = std::numeric_limits<T>::max())
{
    return value != invalid_sentinel;
}

template <typename T>
constexpr bool
populate_if_supported(T& dest, T src, T invalid_sentinel = std::numeric_limits<T>::max())
{
    const bool valid = is_metric_supported(src, invalid_sentinel);
    dest             = valid ? src : T{ 0 };
    return valid;
}

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
