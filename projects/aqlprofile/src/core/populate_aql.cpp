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

#include <assert.h>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "core/amd_aql_pm4_ib_packet.h"
#include "core/aql_profile.hpp"
#include "pm4/cmd_builder.h"

namespace aql_profile {

void PopulateAql(const uint32_t* ib_packet, packet_t* aql_packet) {
  // Populate relevant fields of Aql pkt
  // Size of IB pkt is four DWords
  // Header and completion sinal are not set
  amd_aql_pm4_ib_packet_t* aql_pm4_ib = reinterpret_cast<amd_aql_pm4_ib_packet_t*>(aql_packet);
  aql_pm4_ib->pm4_ib_format = AMD_AQL_PM4_IB_FORMAT;
  aql_pm4_ib->pm4_ib_command[0] = ib_packet[0];
  aql_pm4_ib->pm4_ib_command[1] = ib_packet[1];
  aql_pm4_ib->pm4_ib_command[2] = ib_packet[2];
  aql_pm4_ib->pm4_ib_command[3] = ib_packet[3];
  aql_pm4_ib->dw_count_remain = AMD_AQL_PM4_IB_DW_COUNT_REMAIN;
  for (unsigned i = 0; i < AMD_AQL_PM4_IB_RESERVED_COUNT; ++i) {
    aql_pm4_ib->reserved[i] = 0;
  }

#if defined(DEBUG_TRACE)
  const uint32_t* dwords = (uint32_t*)aql_packet;
  const uint32_t dword_count = sizeof(*aql_packet) / sizeof(uint32_t);
  std::ostringstream oss;
  oss << "AQL 'IB' size(" << dword_count << ")";
  std::clog << std::setw(40) << std::left << "AQL 'IB' size(16)"
            << ":";
  for (unsigned idx = 0; idx < dword_count; idx++) {
    std::clog << " " << std::hex << std::setw(8) << std::setfill('0') << dwords[idx];
  }
  std::clog << std::setfill(' ') << std::endl;
#endif
}

void PopulateAql(const void* cmd_buffer, uint32_t cmd_size, pm4_builder::CmdBuilder* cmd_writer,
                 packet_t* aql_packet) {
  pm4_builder::CmdBuffer ib_buffer;
  cmd_writer->BuildIndirectBufferCmd(&ib_buffer, cmd_buffer, (size_t)cmd_size);
  PopulateAql((const uint32_t*)ib_buffer.Data(), aql_packet);
}

}  // namespace aql_profile
