// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

namespace rocprofiler
{
namespace hsa
{
constexpr hsa_ext_amd_aql_pm4_packet_t null_amd_aql_pm4_packet = {
    .header            = 0,
    .pm4_command       = {0},
    .completion_signal = {.handle = 0}};

/**
 * Struct containing AQL packet information. Including start/stop/read
 * packets along with allocated buffers
 */
struct AQLPacket
{
    using memory_pool_free_func_t = decltype(::hsa_amd_memory_pool_free)*;

    AQLPacket(memory_pool_free_func_t func);
    ~AQLPacket();

    // Keep move constuctors (i.e. std::move())
    AQLPacket(AQLPacket&& other) = default;
    AQLPacket& operator=(AQLPacket&& other) = default;

    // Do not allow copying this class
    AQLPacket(const AQLPacket&) = delete;
    AQLPacket& operator=(const AQLPacket&) = delete;

    hsa_ven_amd_aqlprofile_profile_t profile                = {};
    hsa_ext_amd_aql_pm4_packet_t     start                  = null_amd_aql_pm4_packet;
    hsa_ext_amd_aql_pm4_packet_t     stop                   = null_amd_aql_pm4_packet;
    hsa_ext_amd_aql_pm4_packet_t     read                   = null_amd_aql_pm4_packet;
    bool                             command_buf_mallocd    = false;
    bool                             output_buffer_malloced = false;
    memory_pool_free_func_t          free_func              = nullptr;
};

inline AQLPacket::AQLPacket(memory_pool_free_func_t func)
: free_func{func}
{}
}  // namespace hsa
}  // namespace rocprofiler
