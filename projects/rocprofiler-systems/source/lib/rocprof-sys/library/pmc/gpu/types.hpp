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
#include "library/pmc/gpu/metric_descriptors.hpp"

#include <cstdint>

#if ROCPROFSYS_USE_ROCM > 0
#    include <array>
#    include <sstream>
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

    /// SDMA usage for the target process (root process), in microseconds.
    uint64_t sdma = 0;
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
