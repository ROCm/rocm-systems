/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_sdma_queue.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/runtime.h"
#include "core/inc/amd_memory_region.h"
#include "core/util/atomic_helpers.h"

#include <atomic>
#include <cstring>

namespace rocr {
namespace AMD {

namespace {
  core::SharedQueue* AllocateSdmaSharedQueue(core::Agent* agent) {
    /* The amd_queue_v2_t struct is only used internally by ROCr for SDMA queues. 
    SDMA FW does not use it. */
    core::SharedQueue* shared_queue = static_cast<core::SharedQueue*>(core::Runtime::runtime_singleton_->system_allocator()(
          sizeof(core::SharedQueue), MemoryRegion::GetPageSize(), 0, agent->node_id()));
    memset(shared_queue, 0, sizeof(core::SharedQueue));
    return shared_queue;
  }
} // namespace

SdmaQueue::SdmaQueue(core::Agent* agent, size_t size, bool use_xgmi, int sdma_engine_id)
  : core::Queue(AllocateSdmaSharedQueue(agent), 0, agent),
      agent_(agent),
      queue_start_addr_(nullptr),
      queue_size_(size),
      queue_wptr_(nullptr),
      queue_rptr_(nullptr),
      queue_doorbell_(nullptr),
      sdma_engine_id_(sdma_engine_id),
      is_xgmi_(use_xgmi) {
  memset(&queue_resource_, 0, sizeof(queue_resource_));
}

SdmaQueue::~SdmaQueue() {
  if (queue_resource_.QueueId != 0) {
    AMD::GpuAgent* gpu_agent = static_cast<AMD::GpuAgent*>(agent_);
    gpu_agent->driver().DestroyQueue(queue_resource_.QueueId);
    memset(&queue_resource_, 0, sizeof(queue_resource_));
  }
  FreeQueueBuffer();
  if (shared_queue_) {
    core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);
  }
}

hsa_status_t SdmaQueue::AllocateQueueBuffer() {
  if (queue_start_addr_ != nullptr) {
    return HSA_STATUS_SUCCESS;
  }

  AMD::GpuAgent* gpu_agent = static_cast<AMD::GpuAgent*>(agent_);
  queue_start_addr_ = reinterpret_cast<char*>(
      gpu_agent->system_allocator()(queue_size_, 0x1000, core::MemoryRegion::AllocateExecutable));

  if (!queue_start_addr_) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  memset(queue_start_addr_, 0, queue_size_);
  return HSA_STATUS_SUCCESS;
}

void SdmaQueue::FreeQueueBuffer() {
  if (queue_start_addr_ != nullptr) {
    AMD::GpuAgent* gpu_agent = static_cast<AMD::GpuAgent*>(agent_);
    gpu_agent->system_deallocator()(queue_start_addr_);
    queue_start_addr_ = nullptr;
  }
}

hsa_status_t SdmaQueue::Initialize() {
  AMD::GpuAgent* gpu_agent = static_cast<AMD::GpuAgent*>(agent_);

  // Allocate queue ring buffer
  hsa_status_t status = AllocateQueueBuffer();
  if (status != HSA_STATUS_SUCCESS) return status;

  // Determine queue type based on configuration
  HSA_QUEUE_TYPE queue_type;
  if (sdma_engine_id_ >= 0) {
    queue_type = HSA_QUEUE_SDMA_BY_ENG_ID;
  } else {
    queue_type = is_xgmi_ ? HSA_QUEUE_SDMA_XGMI : HSA_QUEUE_SDMA;
  }

  // Create the queue via KFD
  status = gpu_agent->driver().CreateQueue(
      gpu_agent->node_id(), queue_type,
      100,
      HSA::HSA_AMD_QUEUE_PRIORITY_MAXIMUM, // ignored by lower layers
      sdma_engine_id_,
      queue_start_addr_, queue_size_,
      0,
      nullptr,
      queue_resource_);

  if (status != HSA_STATUS_SUCCESS) {
    FreeQueueBuffer();
    return status;
  }

  // Cache MMIO pointers for performance
  queue_wptr_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_write_ptr);
  queue_rptr_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_read_ptr);
  queue_doorbell_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_DoorBell);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t SdmaQueue::GetInfo(hsa_amd_sdma_queue_info_attribute_t attribute,
                                void* value) const {
  if (value == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  AMD::GpuAgent* gpu_agent = static_cast<AMD::GpuAgent*>(agent_);
  const auto isa_version = gpu_agent->supported_isas()[0]->GetVersion();

  size_t min_submission_size = 0;
  if (isa_version >= core::Isa::Version(9, 0, 0) &&
      (isa_version <= core::Isa::Version(9, 0, 4) ||
       isa_version == core::Isa::Version(9, 0, 12))) {
    min_submission_size = 256;
  }

  switch (attribute) {
    case HSA_AMD_SDMA_QUEUE_INFO_RESOURCE: {
      auto* res = static_cast<hsa_amd_sdma_queue_resource_t*>(value);
      res->ring_base = static_cast<void*>(queue_start_addr_);
      res->ring_size = queue_size_;
      res->read_ptr = queue_rptr_;
      res->write_ptr = queue_wptr_;
      res->doorbell = queue_doorbell_;
      break;
    }
    case HSA_AMD_SDMA_QUEUE_INFO_QUEUE_ID:
      *static_cast<uint64_t*>(value) = static_cast<uint64_t>(queue_resource_.QueueId);
      break;
    case HSA_AMD_SDMA_QUEUE_INFO_ENGINE_ID:
      *static_cast<int32_t*>(value) = sdma_engine_id_;
      break;
    case HSA_AMD_SDMA_QUEUE_INFO_IS_XGMI:
      *static_cast<bool*>(value) = is_xgmi_;
      break;
    case HSA_AMD_SDMA_QUEUE_INFO_AGENT:
      *static_cast<hsa_agent_t*>(value) = gpu_agent->public_handle();
      break;
    case HSA_AMD_SDMA_QUEUE_INFO_MIN_SUBMISSION_SIZE:
      *static_cast<size_t*>(value) = min_submission_size;
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t SdmaQueue::RingDoorbell(uint64_t write_index) {
  if (queue_wptr_ == nullptr || queue_doorbell_ == nullptr) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  *queue_wptr_ = write_index;
  std::atomic_thread_fence(std::memory_order_release);

  *queue_doorbell_ = write_index;

  if (core::Runtime::runtime_singleton_->thunkLoader()->IsDXG() ||
      core::Runtime::runtime_singleton_->thunkLoader()->IsDTIF()) {
    HSAKMT_CALL(hsaKmtQueueRingDoorbell(queue_resource_.QueueId, write_index));
  }

  return HSA_STATUS_SUCCESS;
}

uint64_t SdmaQueue::LoadReadIndexAcquire() {
  return atomic::Load(queue_rptr_, std::memory_order_acquire);
}

uint64_t SdmaQueue::LoadReadIndexRelaxed() {
  return atomic::Load(queue_rptr_, std::memory_order_relaxed);
}

uint64_t SdmaQueue::LoadWriteIndexAcquire() {
  return atomic::Load(queue_wptr_, std::memory_order_acquire);
}

uint64_t SdmaQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(queue_wptr_, std::memory_order_relaxed);
}

void SdmaQueue::StoreReadIndexRelaxed(uint64_t value) {
  atomic::Store(queue_rptr_, value, std::memory_order_relaxed);
}

void SdmaQueue::StoreReadIndexRelease(uint64_t value) {
  atomic::Store(queue_rptr_, value, std::memory_order_release);
}

void SdmaQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(queue_wptr_, value, std::memory_order_relaxed);
}

void SdmaQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(queue_wptr_, value, std::memory_order_release);
}

uint64_t SdmaQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(queue_wptr_, value, expected, std::memory_order_acq_rel);
}

uint64_t SdmaQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(queue_wptr_, value, expected, std::memory_order_acquire);
}

uint64_t SdmaQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(queue_wptr_, value, expected, std::memory_order_relaxed);
}

uint64_t SdmaQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(queue_wptr_, value, expected, std::memory_order_release);
}

uint64_t SdmaQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(queue_wptr_, value, std::memory_order_acq_rel);
}

uint64_t SdmaQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(queue_wptr_, value, std::memory_order_acquire);
}

uint64_t SdmaQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(queue_wptr_, value, std::memory_order_relaxed);
}

uint64_t SdmaQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(queue_wptr_, value, std::memory_order_release);
}

}  // namespace AMD
}  // namespace rocr