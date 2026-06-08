// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace amd_smi
{
constexpr std::size_t MAX_NUM_VCN        = 4;
constexpr std::size_t MAX_NUM_JPEG       = 32;
constexpr std::size_t MAX_NUM_JPEG_V1    = 40;
constexpr std::size_t MAX_NUM_XCP        = 8;
constexpr std::size_t MAX_NUM_XGMI_LINKS = 8;

constexpr std::uint32_t METRIC_VALUE_NOT_SUPPORTED_16 = 0xFFFF;
constexpr std::uint64_t METRIC_VALUE_NOT_SUPPORTED_64 = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Bitfield for AMD SMI metric availability on a GPU.
 *
 * Bit layout matches pmc::collectors::gpu::enabled_metrics so callers can copy
 * .value directly.
 */
union metric_availability
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
        std::uint32_t vcn_activity         : 1;
        std::uint32_t jpeg_activity        : 1;
        std::uint32_t vcn_busy             : 1;
        std::uint32_t jpeg_busy            : 1;
        std::uint32_t xgmi                 : 1;
        std::uint32_t pcie                 : 1;
        std::uint32_t sdma_usage           : 1;
        std::uint32_t gfx_clock            : 1;
        std::uint32_t mem_clock            : 1;
    } bits;
    std::uint32_t value = 0;
};

struct metric_entry
{
    std::string symbol;
    std::string category;
    std::string summary;
    bool        available = false;
};

struct probe_input
{
    std::uint32_t                           current_socket_power = 0;
    std::uint32_t                           average_socket_power = 0;
    std::uint32_t                           hotspot_temperature  = 0;
    std::uint32_t                           edge_temperature     = 0;
    std::uint32_t                           gfx_activity         = 0;
    std::uint32_t                           umc_activity         = 0;
    std::uint32_t                           mm_activity          = 0;
    std::uint32_t                           gfx_clock_mhz        = 0;
    std::uint32_t                           mem_clock_mhz        = 0;
    std::array<std::uint16_t, MAX_NUM_VCN>  vcn_activity         = {};
    std::array<std::uint16_t, MAX_NUM_JPEG> jpeg_activity        = {};
    struct xcp_metrics
    {
        std::array<std::uint16_t, MAX_NUM_JPEG_V1> jpeg_busy = {};
        std::array<std::uint16_t, MAX_NUM_VCN>     vcn_busy  = {};
    };
    std::array<xcp_metrics, MAX_NUM_XCP> xcp_stats{};
    struct
    {
        std::uint16_t width = 0;
        std::uint16_t speed = 0;
        struct
        {
            std::array<std::uint64_t, MAX_NUM_XGMI_LINKS> read  = {};
            std::array<std::uint64_t, MAX_NUM_XGMI_LINKS> write = {};
        } data_acc;
    } xgmi;
    struct
    {
        std::uint16_t width = 0;
        std::uint16_t speed = 0;
        struct
        {
            std::uint64_t acc  = 0;
            std::uint64_t inst = 0;
        } bandwidth;
    } pcie;
};

struct device_probe_result
{
    std::size_t               device_index = 0;
    std::string               product_name;
    std::string               vendor_name;
    metric_availability       metrics{};
    std::vector<metric_entry> entries;
};

metric_availability
compute_availability(const probe_input& raw, bool memory_usage, bool sdma_supported);

void
build_metric_entries(std::size_t device_index, const probe_input& raw,
                     const metric_availability& metrics,
                     std::vector<metric_entry>& entries);
}  // namespace amd_smi
}  // namespace rocprofsys
