////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/amd_aie_aql_queue.h"

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#endif

#include <atomic>
#include <cassert>
#include <cstring>

#include "inc/hsa_ext_amd_aie.h"
#include "core/inc/amd_xdna_driver.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {

static_assert(sizeof(hsa_amd_aie_kernel_dispatch_packet_t) == sizeof(core::AqlPacket),
              "hsa_amd_aie_kernel_dispatch_packet_t must be the same size as core::AqlPacket");

AieAqlQueue::AieAqlQueue(core::SharedQueue* shared_queue, AieAgent* agent, size_t req_size_pkts,
                         uint32_t node_id, uint64_t flags)
    : Queue(shared_queue, flags, agent),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      agent_(*agent),
      active_(false) {
  if (agent_.device_type() != core::Agent::DeviceType::kAmdAieDevice) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_AGENT,
                        "Attempting to create an AIE queue on a non-AIE agent.");
  }
  queue_size_bytes_ = req_size_pkts * sizeof(hsa_amd_aie_kernel_dispatch_packet_t);
  ring_buf_ = agent_.system_allocator()(queue_size_bytes_, 4096,
                                        core::MemoryRegion::AllocateNoFlags);

  if (!ring_buf_) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
                        "Could not allocate a ring buffer for an AIE queue.");
  }

  // Populate hsa_queue_t fields.
  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_SINGLE;
  amd_queue_.hsa_queue.features = 0;  // KMQ queues do not support any features.
  amd_queue_.hsa_queue.size = req_size_pkts;
  amd_queue_.hsa_queue.base_address = ring_buf_;
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);
  amd_queue_.hsa_queue.id = GetQueueId();

  // Populate AMD queue fields.
  amd_queue_.write_dispatch_id = 0;
  amd_queue_.read_dispatch_id = 0;

  signal_.hardware_doorbell_ptr = nullptr;
  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.queue_ptr = &amd_queue_;
  active_ = true;

  HsaQueueResource queue_resource = {};
  hsa_status_t status =
      agent_.driver().CreateQueue(node_id, HSA_QUEUE_COMPUTE_AQL, 0, rocr::HSA::HSA_AMD_QUEUE_PRIORITY_NORMAL, 0,
                                  nullptr, queue_size_bytes_, 0, nullptr, queue_resource);
  if (status != HSA_STATUS_SUCCESS) {
    throw hsa_exception(status, "Failed to create a hardware context for an AIE queue.");
  }
  queue_id_ = queue_resource.QueueId;
}

AieAqlQueue::~AieAqlQueue() {
  AieAqlQueue::Inactivate();
  if (ring_buf_) {
    agent_.system_deallocator()(ring_buf_);
  }
  if (shared_queue_) {
    core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);
  }
}

hsa_status_t AieAqlQueue::Inactivate() {
  bool active = active_.exchange(false, std::memory_order_relaxed);
  if (active) {
    auto err = agent_.driver().DestroyQueue(queue_id_);
    assert(err == HSA_STATUS_SUCCESS && "Destroy queue failed.");
    atomic::Fence(std::memory_order_acquire);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieAqlQueue::SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) {
  return HSA_STATUS_SUCCESS;
}

void AieAqlQueue::Destroy() { delete this; }

// Atomic Reads/Writes
uint64_t AieAqlQueue::LoadReadIndexRelaxed() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieAqlQueue::LoadReadIndexAcquire() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
}

uint64_t AieAqlQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieAqlQueue::LoadWriteIndexAcquire() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
}

void AieAqlQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_relaxed);
}

void AieAqlQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_release);
}

uint64_t AieAqlQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_relaxed);
}

uint64_t AieAqlQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acquire);
}

uint64_t AieAqlQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_release);
}

uint64_t AieAqlQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acq_rel);
}

uint64_t AieAqlQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_relaxed);
}

uint64_t AieAqlQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acquire);
}

uint64_t AieAqlQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_release);
}

uint64_t AieAqlQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acq_rel);
}

void AieAqlQueue::StoreRelaxed(hsa_signal_value_t value) { SubmitPackets(); }

void AieAqlQueue::SubmitPackets() {
  if (!active_.load(std::memory_order_relaxed)) {
    return;
  }

  auto& driver = static_cast<XdnaDriver&>(agent_.driver());

  const uint64_t first_pkt_idx = LoadReadIndexRelaxed();
  const uint64_t last_pkt_idx = LoadWriteIndexAcquire();

  if (first_pkt_idx >= last_pkt_idx) {
    // No packets to submit.
    return;
  }

  const auto num_pkts = last_pkt_idx - first_pkt_idx;
  hsa_status_t status = driver.SubmitCmdChain(amd_queue_.hsa_queue, queue_id_, first_pkt_idx,
                                              num_pkts, agent_.properties().NumNeuralCores);
  if (status != HSA_STATUS_SUCCESS) {
    throw hsa_exception(status, "Could not submit packets");
  }

  atomic::Store(&amd_queue_.read_dispatch_id, last_pkt_idx, std::memory_order_release);
}

void AieAqlQueue::StoreRelease(hsa_signal_value_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}

hsa_status_t AieAqlQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
      *static_cast<hsa_agent_t*>(value) = agent_.public_handle();
      break;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      // Hardware doorbell supports AQL semantics.
      *static_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(signal_.hardware_doorbell_ptr);
      break;
    case HSA_QUEUE_INFO_USE_COUNT:
      // AIE queues do not support counted queue features
      *static_cast<uint32_t*>(value) = static_cast<uint32_t>(-1);
      break;
    case HSA_QUEUE_INFO_HW_ID:
      *static_cast<uint32_t*>(value) = public_handle()->id;
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieAqlQueue::GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t AieAqlQueue::SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

void AieAqlQueue::ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b, hsa_fence_scope_t acquireFence,
                             hsa_fence_scope_t releaseFence, hsa_signal_t* signal) {
  assert(false && "AIE AQL queue does not support PM4 packets.");
}

} // namespace AMD
} // namespace rocr
