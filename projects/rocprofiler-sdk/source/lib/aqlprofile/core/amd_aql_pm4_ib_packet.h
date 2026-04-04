// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef SRC_CORE_AMD_AQL_PM4_IB_PACKET_H_
#define SRC_CORE_AMD_AQL_PM4_IB_PACKET_H_

#include <hsa/hsa.h>

// Value of 'pm4_ib_format' field of amd_aql_pm4_ib_packet_t packet
static const uint32_t AMD_AQL_PM4_IB_FORMAT = 1;
// Value of 'dw_count_remain' field of amd_aql_pm4_ib_packet_t packet
static const uint32_t AMD_AQL_PM4_IB_DW_COUNT_REMAIN = 10;
// Size of 'reserved' array of amd_aql_pm4_ib_packet_t packet
static const uint32_t AMD_AQL_PM4_IB_RESERVED_COUNT = 8;

// AQL Vendor Specific Packet which carry PM4 IB command
typedef struct {
  uint16_t header;
  uint16_t pm4_ib_format;
  uint32_t pm4_ib_command[4];
  uint32_t dw_count_remain;
  uint32_t reserved[AMD_AQL_PM4_IB_RESERVED_COUNT];
  hsa_signal_t completion_signal;
} amd_aql_pm4_ib_packet_t;

#endif  // SRC_CORE_AMD_AQL_PM4_IB_PACKET_H_
