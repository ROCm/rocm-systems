////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/host_queue.h"

#include "core/inc/runtime.h"
#include "core/util/utils.h"

namespace rocr {
namespace core {

HostQueue::HostQueue(core::SharedQueue* shared_queue, hsa_region_t region, uint32_t ring_size,
                     hsa_queue_type32_t type, uint32_t features, hsa_signal_t doorbell_signal)
    : Queue(shared_queue, 0), size_(ring_size) {
  HSA::hsa_memory_register(this, sizeof(HostQueue));
  MAKE_NAMED_SCOPE_GUARD(registerGuard,
                         [&]() { HSA::hsa_memory_deregister(this, sizeof(HostQueue)); });

  const size_t queue_buffer_size = size_ * sizeof(AqlPacket);
  if (HSA_STATUS_SUCCESS !=
      HSA::hsa_memory_allocate(region, queue_buffer_size, &ring_)) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES, "Host queue buffer alloc failed\n");
  }
  MAKE_NAMED_SCOPE_GUARD(bufferGuard, [&]() { HSA::hsa_memory_free(&ring_); });

  assert(IsMultipleOf(ring_, kRingAlignment));
  assert(ring_ != NULL);

  // Fill the ring buffer with invalid packet headers.
  // Leave packet content uninitialized to help track errors.
  for (uint32_t pkt_id = 0; pkt_id < size_; pkt_id++) {
    (((AqlPacket*)ring_)[pkt_id]).dispatch.header = HSA_PACKET_TYPE_INVALID;
  }

  amd_queue_.hsa_queue.base_address = ring_;
  amd_queue_.hsa_queue.size = size_;
  amd_queue_.hsa_queue.doorbell_signal = doorbell_signal;
  amd_queue_.hsa_queue.id = this->GetQueueId();
  amd_queue_.hsa_queue.type = type;
  amd_queue_.hsa_queue.features = features;
#ifdef HSA_LARGE_MODEL
  AMD_HSA_BITS_SET(
      amd_queue_.queue_properties, AMD_QUEUE_PROPERTIES_IS_PTR64, 1);
#else
  AMD_HSA_BITS_SET(
      amd_queue_.queue_properties, AMD_QUEUE_PROPERTIES_IS_PTR64, 0);
#endif
  amd_queue_.write_dispatch_id = amd_queue_.read_dispatch_id = 0;
  AMD_HSA_BITS_SET(
      amd_queue_.queue_properties, AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, 0);

  bufferGuard.Dismiss();
  registerGuard.Dismiss();
}

HostQueue::~HostQueue() {
  HSA::hsa_memory_free(shared_queue_);
  HSA::hsa_memory_free(ring_);
  HSA::hsa_memory_deregister(this, sizeof(HostQueue));
}

}  // namespace core
}  // namespace rocr
