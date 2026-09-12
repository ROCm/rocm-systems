////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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


#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_ARG_SYNC_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_ARG_SYNC_H_

#include <cstddef>
#include <cstdint>

#include "inc/hsa_ext_amd_aie.h"

namespace rocr {
namespace AMD {

/// @brief Calls @p sync for each argument range of @p pkt the runtime has to
/// synchronize before and after a dispatch.
///
/// An argument's size is the length of the range the runtime keeps coherent for
/// it, and a size of zero means the caller synchronizes that range itself (see
/// ::hsa_amd_aie_kernel_dispatch_packet_s::kernarg_address), so it is skipped.
/// That matters for a large argument the host writes once and the device then
/// owns -- a weight set, say -- where the flush would otherwise be repeated in
/// full on every dispatch, at a cost proportional to its size rather than to
/// the work.
///
/// The kernarg buffer holds the addresses first and the sizes second, both
/// @c num_kernargs long.
///
/// @param[in] pkt packet whose arguments to walk. Must not be null.
/// @param[in] sync callable invoked as @c sync(void*,size_t) per range.
template <typename SyncFn> void ForEachArgumentSyncRange(
    const hsa_amd_aie_kernel_dispatch_packet_t* pkt, SyncFn&& sync) {
  auto* kernarg_address = static_cast<uint64_t*>(pkt->kernarg_address);
  if (kernarg_address == nullptr) return;

  for (uint32_t kernarg_idx = 0; kernarg_idx < pkt->num_kernargs; ++kernarg_idx) {
    const size_t size = kernarg_address[kernarg_idx + pkt->num_kernargs];
    if (size == 0) continue;
    sync(reinterpret_cast<void*>(kernarg_address[kernarg_idx]), size);
  }
}

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_ARG_SYNC_H_
