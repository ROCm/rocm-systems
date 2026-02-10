// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "library/pmc/common/types.hpp"

#include <cstdint>
#include <sstream>
#include <string>

#if ROCPROFSYS_USE_ROCM > 0
#    include <array>
#    include <set>
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
        uint32_t vcn_activity         : 1;
        uint32_t jpeg_activity        : 1;
        uint32_t vcn_busy             : 1;  // Per-XCP VCN busy
        uint32_t jpeg_busy            : 1;  // Per-XCP JPEG busy
        uint32_t xgmi                 : 1;
        uint32_t pcie                 : 1;
    } bits;
    uint32_t value = 0;
};

inline std::string
to_string(const enabled_metrics& metrics)
{
    std::stringstream ss;
    ss << "[SMI enabled metrics] ";
    ss << "Current socket power: " << static_cast<bool>(metrics.bits.current_socket_power)
       << ", Average socket power: "
       << static_cast<bool>(metrics.bits.average_socket_power)
       << ", Memory usage: " << static_cast<bool>(metrics.bits.memory_usage)
       << ", Hotspot temperature: " << static_cast<bool>(metrics.bits.hotspot_temperature)
       << ", Edge temperature: " << static_cast<bool>(metrics.bits.edge_temperature)
       << ", GFX activity: " << static_cast<bool>(metrics.bits.gfx_activity)
       << ", UMC activity: " << static_cast<bool>(metrics.bits.umc_activity)
       << ", MM activity: " << static_cast<bool>(metrics.bits.mm_activity)
       << ", VCN activity: " << static_cast<bool>(metrics.bits.vcn_activity)
       << ", JPEG activity: " << static_cast<bool>(metrics.bits.jpeg_activity)
       << ", VCN busy: " << static_cast<bool>(metrics.bits.vcn_busy)
       << ", JPEG busy: " << static_cast<bool>(metrics.bits.jpeg_busy)
       << ", XGMI: " << static_cast<bool>(metrics.bits.xgmi)
       << ", PCIE: " << static_cast<bool>(metrics.bits.pcie) << "\n";
    return ss.str();
}

#if ROCPROFSYS_USE_ROCM > 0

// Ensure we have the correct max values defined
#    ifdef AMDSMI_MAX_NUM_JPEG_ENG_V1
#        define ROCPROFSYS_MAX_NUM_JPEG_ENGINES AMDSMI_MAX_NUM_JPEG_ENG_V1
#    else
#        define ROCPROFSYS_MAX_NUM_JPEG_ENGINES 40
#    endif

#    ifndef AMDSMI_MAX_NUM_VCN
#        define AMDSMI_MAX_NUM_VCN 4
#    endif

#    ifndef AMDSMI_MAX_NUM_JPEG
#        define AMDSMI_MAX_NUM_JPEG 32
#    endif

#    ifndef AMDSMI_MAX_NUM_XCP
#        define AMDSMI_MAX_NUM_XCP 8
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
        std::stringstream ss;
        ss << error_message << " AMD SMI Error code: " << status;
        throw std::runtime_error(ss.str());
    }
}

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace gpu
}  // namespace pmc
}  // namespace rocprofsys
