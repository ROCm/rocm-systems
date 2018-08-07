////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCTRACER_HSA_RT_UTILS_HPP_
#define INC_ROCTRACER_HSA_RT_UTILS_HPP_

#include <hsa.h>

#include <cstdint>
#include <cstddef>

#define HSART_CALL(call)                                                                           \
  do {                                                                                             \
    hsa_status_t status = call;                                                                    \
    if (status != HSA_STATUS_SUCCESS) {                                                            \
      std::cerr << "HSA-rt call '" << #call << "' error(" << std::hex << status << ")"             \
        << std::dec << std::endl << std::flush;                                                    \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

namespace hsa_rt_utils {

// HSA runtime timer implementation
class Timer {
  public:
  typedef uint64_t timestamp_t;
  typedef long double freq_t;
  
  Timer() {
    timestamp_t timestamp_hz = 0;
    HSART_CALL(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_hz));
    timestamp_rate_ = (freq_t)1000000000 / (freq_t)timestamp_hz;
  }

  // Returns HSA runtime timestamp rate
  freq_t timestamp_rate() const { return timestamp_rate_; }

  // Convert a given timestamp to ns
  timestamp_t timestamp_to_ns(const timestamp_t &timestamp) const {
    return timestamp_t((freq_t)timestamp * timestamp_rate_);
  }

  // Return timestamp in 'ns'
  timestamp_t timestamp_ns() const {
    timestamp_t timestamp;
    HSART_CALL(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &timestamp));
    return timestamp_to_ns(timestamp);
  }

  private:
  // Timestamp rate
  freq_t timestamp_rate_;
};

}  // namespace hsa_rt_utils

#endif  // INC_ROCTRACER_HSA_RT_UTILS_HPP_
