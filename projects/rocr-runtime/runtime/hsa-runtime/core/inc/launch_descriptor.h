////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
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

// HSA runtime C++ interface file for launch descriptors.

#ifndef HSA_RUNTME_CORE_INC_LAUNCH_DESCRIPTOR_H_
#define HSA_RUNTME_CORE_INC_LAUNCH_DESCRIPTOR_H_

#include <cstring>
#include <vector>

#include "core/inc/exceptions.h"
#include "core/util/utils.h"
#include "inc/amd_launch_descriptor.h"
#include "inc/hsa_ext_amd.h"

namespace rocr {
namespace core {

/// @brief Simple wrapper for launch descriptor data.
/// Just the 52-byte data - handle points directly to this.
struct LaunchDescriptor {
  amd_launch_descriptor_data_t data;

  LaunchDescriptor() {
    std::memset(&data, 0, sizeof(data));
  }

  /// @brief Convert handle to descriptor pointer.
  static __forceinline LaunchDescriptor* Convert(amd_launch_descriptor_t handle) {
    return reinterpret_cast<LaunchDescriptor*>(handle.handle);
  }

  /// @brief Convert descriptor pointer to handle.
  static __forceinline amd_launch_descriptor_t Convert(LaunchDescriptor* desc) {
    amd_launch_descriptor_t handle;
    handle.handle = reinterpret_cast<uint64_t>(desc);
    return handle;
  }
};

/// @brief Set a field in the descriptor.
inline void SetDescriptorField(LaunchDescriptor* desc, uint32_t field, uint64_t value) {
  switch (field) {
    case 0: desc->data.version = static_cast<uint8_t>(value); break;
    case 1: desc->data.priority = static_cast<uint8_t>(value); break;
    case 2: desc->data.pm_hint = static_cast<uint8_t>(value); break;
    case 3:
      if (value >= 16) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "CU start must be < 16");
      desc->data.cu_start = static_cast<uint32_t>(value);
      break;
    case 4:
      if (value >= 16) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "CU count must be < 16");
      desc->data.cu_count = static_cast<uint32_t>(value);
      break;
    case 5:
      if (value >= 4) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "SE enable must be < 4");
      desc->data.se_en = static_cast<uint32_t>(value);
      break;
    case 6: desc->data.dispatch_granularity_limiter = static_cast<uint16_t>(value); break;
    case 7: desc->data.tg_chunk_size = static_cast<uint16_t>(value); break;
    default: throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Unknown field");
  }
}

/// @brief Set a prefetch region.
inline void SetDescriptorPrefetch(LaunchDescriptor* desc, uint32_t index, const amd_data_prefetch_t* prefetch) {
  if (index >= AMD_LAUNCH_DESCRIPTOR_MAX_PREFETCH_REGIONS) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Prefetch index out of range");
  }
  if (prefetch) {
    desc->data.prefetch[index] = *prefetch;
  } else {
    std::memset(&desc->data.prefetch[index], 0, sizeof(amd_data_prefetch_t));
  }
}

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTME_CORE_INC_LAUNCH_DESCRIPTOR_H_
