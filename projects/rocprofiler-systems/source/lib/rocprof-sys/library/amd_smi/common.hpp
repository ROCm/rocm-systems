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

#include <amd_smi/amdsmi.h>
#include <fmt/format.h>
#include <set>
#include <sstream>
#include <stdexcept>

namespace rocprofsys
{
namespace amd_smi
{

union smi_metric_options
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
    } bits;
    uint32_t value;
};

struct version
{
    struct
    {
        uint32_t major;
        uint32_t minor;
        uint32_t release;
    } numeric_representation;
    std::string string_representation;
};

inline void
check_status(const amdsmi_status_t& status, const char* error_message)
{
    if(status != AMDSMI_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << error_message << " Error: " << status;
        throw std::runtime_error(ss.str());
        // throw std::runtime_error(fmt::format("{} Error: {}", error_message, status));
    }
};

enum class device_selection_mode : std::uint8_t
{
    all,
    none,
    specific
};

struct device_filter
{
    device_selection_mode mode;
    std::set<size_t>      indices;
};

#ifdef AMDSMI_MAX_NUM_JPEG_ENG_V1
#    define AMDSMI_MAX_NUM_JPEG_ENGINES AMDSMI_MAX_NUM_JPEG_ENG_V1
#else
#    define AMDSMI_MAX_NUM_JPEG_ENGINES 40
#endif

#ifndef AMDSMI_MAX_NUM_XCP
#    define AMDSMI_MAX_NUM_XCP 8
#endif
struct smi_metrics
{
    uint32_t current_socket_power;
    uint32_t average_socket_power;
    uint32_t memory_usage;
    uint16_t hotspot_temperature;
    uint16_t edge_temperature;
    uint32_t gfx_activity;
    uint32_t umc_activity;
    uint32_t mm_activity;
    struct
    {
        uint16_t vcn_busy[AMDSMI_MAX_NUM_VCN];
        uint16_t jpeg_busy[AMDSMI_MAX_NUM_JPEG_ENGINES];
    } xcp_stats[AMDSMI_MAX_NUM_XCP];
};

}  // namespace amd_smi
}  // namespace rocprofsys
