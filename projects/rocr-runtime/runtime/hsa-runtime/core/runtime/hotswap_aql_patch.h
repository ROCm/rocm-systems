/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_RUNTIME_HOTSWAP_AQL_PATCH_H_
#define HSA_RUNTIME_CORE_RUNTIME_HOTSWAP_AQL_PATCH_H_

#include <cstdio>
#include <cstdlib>

#include "core/inc/queue.h"

namespace rocr {
namespace AMD {
namespace hotswap {
namespace lazy {

using PrepareDispatchKernelObjectCallback = bool (*)(uint64_t*,
                                                     uint32_t*,
                                                     uint32_t*);
using PacketBodyFenceCallback = void (*)();

inline bool ComputeDoorbellPatchRange(uint64_t last_doorbell, uint64_t end,
                                      uint64_t queue_size, uint64_t& first,
                                      bool& has_work,
                                      const char*& failure_reason) {
  has_work = false;
  failure_reason = nullptr;
  if (queue_size == 0) {
    failure_reason = "queue size is zero";
    return false;
  }

  if (last_doorbell != UINT64_MAX && end <= last_doorbell) {
    first = end + 1;
    return true;
  }

  first = last_doorbell == UINT64_MAX ? 0 : last_doorbell + 1;
  if (end - first >= queue_size) {
    failure_reason = "doorbell range aliases the AQL ring";
    return false;
  }
  has_work = true;
  return true;
}

inline bool PatchPublishedKernelDispatchPacket(
    core::AqlPacket& packet,
    PrepareDispatchKernelObjectCallback prepare_kernel_object,
    PacketBodyFenceCallback packet_body_fence) {
  const uint16_t header =
      __atomic_load_n(reinterpret_cast<uint16_t*>(&packet.packet.header),
                      __ATOMIC_ACQUIRE);
  const hsa_packet_type_t type =
      static_cast<hsa_packet_type_t>((header >> HSA_PACKET_HEADER_TYPE) & 0xff);
  uint64_t* kernel_object = nullptr;
  uint32_t* private_segment_size = nullptr;
  uint32_t* group_segment_size = nullptr;
  if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
    kernel_object = &packet.dispatch.kernel_object;
    private_segment_size = &packet.dispatch.private_segment_size;
    group_segment_size = &packet.dispatch.group_segment_size;
  } else if (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
             packet.ext_dispatch.amd_format ==
                 HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
    kernel_object = &packet.ext_dispatch.kernel_object;
    private_segment_size = &packet.ext_dispatch.private_segment_size;
    group_segment_size = &packet.ext_dispatch.group_segment_size;
  } else {
    return false;
  }

  const uint16_t invalid_header =
      HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  __atomic_store_n(reinterpret_cast<uint16_t*>(&packet.packet.header),
                   invalid_header, __ATOMIC_RELEASE);

  if (!prepare_kernel_object ||
      !prepare_kernel_object(kernel_object, private_segment_size,
                             group_segment_size)) {
    fprintf(stderr, "hotswap: refusing kernel dispatch: packet patch failed\n");
    std::abort();
  }

  if (packet_body_fence) packet_body_fence();
  __atomic_store_n(reinterpret_cast<uint16_t*>(&packet.packet.header), header,
                   __ATOMIC_RELEASE);
  return true;
}

}  // namespace lazy
}  // namespace hotswap
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_RUNTIME_HOTSWAP_AQL_PATCH_H_
