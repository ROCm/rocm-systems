/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_aie_umq_queue.h"

#include <atomic>
#include <cassert>

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

AieUmqQueue::AieUmqQueue(core::SharedQueue* shared_queue, AieAgent* agent, size_t req_size_pkts,
                         uint32_t node_id, core::HsaEventCallback callback, void* err_data,
                         uint64_t flags)
    : Queue(shared_queue, flags, agent),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      queue_size_bytes_(req_size_pkts * sizeof(hsa_amd_aie_kernel_dispatch_packet_t)),
      suspended_(false),
      errors_callback_(callback),
      errors_data_(err_data) {
  if (agent->device_type() != core::Agent::DeviceType::kAmdAieDevice) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_AGENT,
                        "Attempting to create an AIE queue on a non-AIE agent.");
  }

  ring_buf_ =
      agent->system_allocator()(queue_size_bytes_, 4096, core::MemoryRegion::AllocateNoFlags);
  if (!ring_buf_) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
                        "Could not allocate a ring buffer for an AIE queue.");
  }
  MAKE_NAMED_SCOPE_GUARD(ring_buf_guard, [&] { agent->system_deallocator()(ring_buf_); });

  // Populate hsa_queue_t fields.
  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_SINGLE;
  amd_queue_.hsa_queue.features = 0;  // UMQ queues do not support any features.
  amd_queue_.hsa_queue.size = req_size_pkts;
  amd_queue_.hsa_queue.base_address = ring_buf_;
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);
  amd_queue_.hsa_queue.id = GetQueueId();

  // Populate AMD queue fields.
  amd_queue_.write_dispatch_id = 0;
  amd_queue_.read_dispatch_id = 0;

  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.queue_ptr = &amd_queue_;

  auto& driver = static_cast<XdnaDriver&>(agent->driver());
  // The hardware context spans every column; CERT reads AQL packets directly out of ring_buf_,
  // the same ring the application already writes them into.
  hsa_status_t err =
      driver.CreateUserModeQueue(ring_buf_, *agent, agent->num_cols(), agent->num_core_rows(),
                                 &umq_metadata_, &doorbell_);
  if (err != HSA_STATUS_SUCCESS) {
    throw hsa_exception(err, "Failed to create UMQ metadata for the AIE queue.");
  }
  // Hardware doorbell supports AQL semantics, the same as a GPU queue. The UMQ doorbell register
  // is 32 bits (unlike signal_.hardware_doorbell_ptr's 64-bit pointee); this field is only used
  // for address reporting (HSA_AMD_QUEUE_INFO_DOORBELL_ID), never dereferenced through its own
  // type, so the width mismatch does not matter.
  signal_.hardware_doorbell_ptr = reinterpret_cast<volatile uint64_t*>(doorbell_);

  active_ = true;

  ring_buf_guard.Dismiss();
}

AieUmqQueue::~AieUmqQueue() {
  auto err = AieUmqQueue::Inactivate();
  assert(err == HSA_STATUS_SUCCESS && "Destroy queue failed.");
  (void)err;
  if (ring_buf_) {
    auto& agent = static_cast<AieAgent&>(*GetAgent());
    agent.system_deallocator()(ring_buf_);
  }
  if (shared_queue_) {
    core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);
  }
}

hsa_status_t AieUmqQueue::Inactivate() {
  bool active = active_.exchange(false, std::memory_order_relaxed);
  if (!active) {
    return HSA_STATUS_SUCCESS;
  }

  auto& driver = static_cast<XdnaDriver&>(GetAgent()->driver());
  hsa_status_t err = driver.DestroyUserModeQueue(umq_metadata_);
  umq_metadata_ = nullptr;
  atomic::Fence(std::memory_order_acquire);
  return err;
}

hsa_status_t AieUmqQueue::SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) {
  return HSA_STATUS_SUCCESS;
}

void AieUmqQueue::Destroy() { delete this; }

// Atomic Reads/Writes
uint64_t AieUmqQueue::LoadReadIndexRelaxed() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieUmqQueue::LoadReadIndexAcquire() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
}

uint64_t AieUmqQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieUmqQueue::LoadWriteIndexAcquire() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
}

void AieUmqQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_relaxed);
}

void AieUmqQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_release);
}

uint64_t AieUmqQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_relaxed);
}

uint64_t AieUmqQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acquire);
}

uint64_t AieUmqQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_release);
}

uint64_t AieUmqQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acq_rel);
}

uint64_t AieUmqQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_relaxed);
}

uint64_t AieUmqQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acquire);
}

uint64_t AieUmqQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_release);
}

uint64_t AieUmqQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acq_rel);
}

void AieUmqQueue::StoreRelaxed(hsa_signal_value_t value) {
  if (!active_.load(std::memory_order_relaxed) || suspended_) {
    return;
  }

  // Hardware doorbell supports AQL semantics: the application already wrote its packets directly
  // into ring_buf_, so there is nothing left to build or submit. Ringing the doorbell with the new
  // write index is the entire dispatch path, the same as AqlQueue::StoreRelaxed for a GPU queue.
  // CERT advances the ring's read index on its own as it consumes packets; there is no per-batch
  // driver call here to learn how many completed.
  store_fence();
  *doorbell_ = static_cast<uint32_t>(value);
}

void AieUmqQueue::StoreRelease(hsa_signal_value_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}

hsa_status_t AieUmqQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
      *static_cast<hsa_agent_t*>(value) = GetAgent()->public_handle();
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
    case HSA_AMD_QUEUE_INFO_ENGINE_TYPE:
      *static_cast<hsa_amd_queue_engine_t*>(value) = HSA_AMD_QUEUE_ENGINE_AIE;
      break;
    case HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieUmqQueue::GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t AieUmqQueue::SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

void AieUmqQueue::ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b, hsa_fence_scope_t acquireFence,
                             hsa_fence_scope_t releaseFence, hsa_signal_t* signal) {
  assert(false && "AIE UMQ queue does not support PM4 packets.");
}

} // namespace AMD
} // namespace rocr
