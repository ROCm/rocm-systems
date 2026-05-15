// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#define TOOL_VERSION_MAJOR 3
#define TOOL_VERSION_MINOR 0
#define TOOL_VERSION_REV   0
#define TOOL_VERSION       "3.0.0"

#include "trace_decoder_api.h"
#include "sdk/code_printing.hpp"

#include <memory>
#include <string>
#include <string_view>

#define CHECK_DECODER(x) \
    if(auto _status = (x); _status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) { \
        std::cout << __FILE__ << ':' << __LINE__ << " Check failed: " << _status << std::endl; \
        std::terminate(); \
    }

#define CHECK_TRUE(x) \
    if(bool status = (x); status != true) { \
        std::cout << __FILE__ << ':' << __LINE__ << " Check failed: " << status << std::endl; \
        std::terminate(); \
    }

#define WARNING(x) \
    std::cout << __FILE__ << ':' << __LINE__ << " Warning: " << x << std::endl;

using pcinfo_t           = rocprofiler_thread_trace_decoder_pc_t;
using occupancy_t        = rocprofiler_thread_trace_decoder_occupancy_t;
using wave_t             = rocprofiler_thread_trace_decoder_wave_t;
using perfevent_t        = rocprofiler_thread_trace_decoder_perfevent_t;
using wave_instruction_t = rocprofiler_thread_trace_decoder_inst_t;

template <>
struct std::hash<pcinfo_t>
{
public:
    size_t operator()(const pcinfo_t& a) const
    {
        return (a.code_object_id << 32) ^ (a.code_object_id >> 32) ^ a.address;
    }
};

inline bool
operator==(const pcinfo_t& a, const pcinfo_t& b)
{
    return a.address == b.address && a.code_object_id == b.code_object_id;
};

inline bool
operator<(const pcinfo_t& a, const pcinfo_t& b)
{
    if(a.code_object_id == b.code_object_id) return a.address < b.address;
    return a.code_object_id < b.code_object_id;
};

namespace rocprofiler
{
namespace att_wrapper
{
struct KernelName
{
    std::string name{};
    std::string demangled{};
};

}  // namespace att_wrapper
}  // namespace rocprofiler
