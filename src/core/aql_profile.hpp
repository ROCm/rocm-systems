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

#ifndef SRC_CORE_AQL_PROFILE_H_
#define SRC_CORE_AQL_PROFILE_H_

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <iostream>
#include <string>
#include "include/aql_profile_v2.h"

#include "core/aql_profile_exception.h"

#define PUBLIC_API __attribute__((visibility("default")))

namespace pm4_builder {
class CmdBuilder;
}

namespace aql_profile {
typedef hsa_ven_amd_aqlprofile_descriptor_t descriptor_t;
typedef hsa_ven_amd_aqlprofile_profile_t profile_t;
typedef hsa_ven_amd_aqlprofile_info_type_t info_type_t;
typedef hsa_ven_amd_aqlprofile_data_callback_t data_callback_t;
typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
typedef hsa_ven_amd_aqlprofile_event_t event_t;

void PopulateAql(const void* cmd_buffer, uint32_t cmd_size, pm4_builder::CmdBuilder* cmd_writer,
                 packet_t* aql_packet);
void* LegacyAqlAcquire(const packet_t* aql_packet, void* data);
void* LegacyAqlRelease(const packet_t* aql_packet, void* data);
void* LegacyPm4(const packet_t* aql_packet, void* data);

class event_exception : public aql_profile_exc_val<event_t> {
 public:
  event_exception(const std::string& m, const event_t& ev) : aql_profile_exc_val(m, ev) {}
};

}  // namespace aql_profile

static std::ostream& operator<<(std::ostream& os, const aql_profile::event_t& ev) {
  os << "event( block(" << ev.block_name << "." << ev.block_index << "), Id(" << ev.counter_id
     << "))";
  return os;
}

#endif  // SRC_CORE_AQL_PROFILE_H_
