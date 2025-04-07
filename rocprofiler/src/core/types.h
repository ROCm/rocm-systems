/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef SRC_CORE_TYPES_H_
#define SRC_CORE_TYPES_H_

#include <iostream>

#include <hsa/hsa_ven_amd_aqlprofile.h>

namespace rocprofiler {
typedef hsa_ven_amd_aqlprofile_pfn_t pfn_t;
typedef hsa_ven_amd_aqlprofile_event_t event_t;
typedef hsa_ven_amd_aqlprofile_parameter_t parameter_t;
typedef hsa_ven_amd_aqlprofile_profile_t profile_t;
typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
typedef uint32_t packet_word_t;
typedef uint64_t timestamp_t;

inline std::ostream& operator<<(std::ostream& out, const event_t& event) {
  out << "[block_name(" << event.block_name << "). block_index(" << event.block_index
      << "). counter_id(" << event.counter_id << ")]";
  return out;
}
inline std::ostream& operator<<(std::ostream& out, const parameter_t& parameter) {
  out << "[parameter_name(" << parameter.parameter_name << "). value(" << parameter.value << ")]";
  return out;
}

}  // namespace rocprofiler

#endif  // SRC_CORE_TYPES_H_
