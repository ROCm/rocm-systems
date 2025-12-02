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

#include <array>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace amd_smi
{

constexpr auto METRIC_VALUE_NOT_SUPPORTED = 0xffff;

union enabled_metric
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
        uint32_t xgmi                 : 1;
        uint32_t pcie                 : 1;
    } bits;
    uint32_t value = 0;
};

struct version
{
    struct
    {
        uint32_t major   = 0;
        uint32_t minor   = 0;
        uint32_t release = 0;
    } numeric_representation;
    std::string string_representation;
};

#if ROCPROFSYS_USE_ROCM > 0

// Ensure we have the correct max values defined
#    ifdef AMDSMI_MAX_NUM_JPEG_ENG_V1
#        define ROCPROFSYS_MAX_NUM_JPEG_ENGINES AMDSMI_MAX_NUM_JPEG_ENG_V1
#    else
#        define ROCPROFSYS_MAX_NUM_JPEG_ENGINES 40
#    endif

#    ifndef AMDSMI_MAX_NUM_XCP
#        define AMDSMI_MAX_NUM_XCP 8
#    endif

struct smi_metrics
{
    struct xcp_metrics
    {
        std::array<uint16_t, ROCPROFSYS_MAX_NUM_JPEG_ENGINES> jpeg_busy;
        std::array<uint16_t, AMDSMI_MAX_NUM_XCP>              vcn_busy;
    };

    uint32_t                                        current_socket_power = 0;
    uint32_t                                        average_socket_power = 0;
    uint64_t                                        memory_usage         = 0;
    int64_t                                         hotspot_temperature  = 0;
    int64_t                                         edge_temperature     = 0;
    uint32_t                                        gfx_activity         = 0;
    uint32_t                                        umc_activity         = 0;
    uint32_t                                        mm_activity          = 0;
    std::array<xcp_metrics, AMDSMI_MAX_NUM_XCP>     xcp_stats;
    uint16_t                                        xgmi_link_width = 0;
    uint16_t                                        xgmi_link_speed = 0;
    std::array<uint64_t, AMDSMI_MAX_NUM_XGMI_LINKS> xgmi_read_data_acc;
    std::array<uint64_t, AMDSMI_MAX_NUM_XGMI_LINKS> xgmi_write_data_acc;
    uint16_t                                        pcie_link_width     = 0;
    uint16_t                                        pcie_link_speed     = 0;
    uint64_t                                        pcie_bandwidth_acc  = 0;
    uint64_t                                        pcie_bandwidth_inst = 0;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

enum class device_selection_mode : uint8_t
{
    ALL,
    NONE,
    SPECIFIC
};

struct device_filter
{
    device_selection_mode mode = device_selection_mode::ALL;
    std::set<size_t>      indices;
};

#if ROCPROFSYS_USE_ROCM > 0

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

}  // namespace amd_smi
}  // namespace rocprofsys
