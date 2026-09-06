/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_sdma_queue.h"

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/exceptions.h"
#include "core/inc/runtime.h"
#include "core/util/atomic_helpers.h"

#include <atomic>
#include <cstring>

namespace rocr {
namespace AMD {

namespace {

core::SharedQueue* AllocateSdmaSharedQueue(core::Agent* agent, uint64_t flags) {
  // SDMA firmware does not consume amd_queue_v2_t, but ROCr needs the shared
  // queue wrapper so hsa_queue_t can round-trip to core::Queue.
  auto* gpu_agent = static_cast<AMD::GpuAgent*>(agent);
  const bool device_mem_descriptor =
      (flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR) != 0;

  core::SharedQueue* shared_queue = nullptr;
  if (device_mem_descriptor) {
    if (!gpu_agent->LargeBarEnabled()) {
      throw hsa_exception(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
                          "Device-memory SDMA queue descriptors require Large BAR.");
    }
    shared_queue = static_cast<core::SharedQueue*>(gpu_agent->finegrain_allocator()(
        sizeof(core::SharedQueue),
        core::MemoryRegion::AllocateUncached | core::MemoryRegion::AllocateQueueObject));
  } else {
    shared_queue = static_cast<core::SharedQueue*>(
        gpu_agent->system_allocator()(sizeof(core::SharedQueue), MemoryRegion::GetPageSize(),
                                      core::MemoryRegion::AllocateQueueObject));
  }

  if (shared_queue == nullptr) {
    throw hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                        "Failed to allocate shared SDMA queue handle.");
  }
  memset(shared_queue, 0, sizeof(core::SharedQueue));
  return shared_queue;
}

}  // namespace

SdmaQueue::SdmaQueue(core::Agent* agent, size_t size_bytes, uint64_t flags, int32_t sdma_engine_id,
                     core::HsaEventCallback callback, void* err_data)
    : core::Queue(AllocateSdmaSharedQueue(agent, flags), flags,
                  !static_cast<AMD::GpuAgent*>(agent)->is_xgmi_cpu_gpu(), agent),
      core::LocalSignal(0, false),
      core::DoorbellSignal(signal()),
      agent_(agent),
      queue_start_addr_(nullptr),
      queue_size_(size_bytes),
      queue_wptr_(nullptr),
      queue_rptr_(nullptr),
      queue_doorbell_(nullptr),
      sdma_engine_id_(sdma_engine_id),
      active_(false),
      errors_callback_(callback),
      errors_data_(err_data) {
  memset(&queue_resource_, 0, sizeof(queue_resource_));
}

SdmaQueue::~SdmaQueue() {
  // Detach before tearing the queue down: destroying a queue whose packets hung
  // the engine is what triggers the driver's reset, so the exception arrives
  // after this point and must still find the registration.
  core::Runtime::runtime_singleton_->DetachSdmaErrorHandler(public_handle());

  hsa_status_t err = Inactivate();
  assert(err == HSA_STATUS_SUCCESS && "Destroy SDMA queue failed.");
  (void)err;
  FreeQueueBuffer();
  if (shared_queue_ != nullptr) {
    if (IsDeviceMemQueueDescriptor()) {
      gpu_agent()->finegrain_deallocator()(shared_queue_);
    } else {
      gpu_agent()->system_deallocator()(shared_queue_);
    }
    shared_queue_ = nullptr;
  }
}

AMD::GpuAgent* SdmaQueue::gpu_agent() const { return static_cast<AMD::GpuAgent*>(agent_); }

hsa_status_t SdmaQueue::AllocateQueueBuffer() {
  if (queue_start_addr_ != nullptr) {
    return HSA_STATUS_SUCCESS;
  }

  if (IsDeviceMemRingBuf()) {
    if (!gpu_agent()->LargeBarEnabled()) {
      return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
    }
    queue_start_addr_ = reinterpret_cast<char*>(gpu_agent()->coarsegrain_allocator()(
        queue_size_,
        core::MemoryRegion::AllocateExecutable | core::MemoryRegion::AllocateUncached));
  } else {
    queue_start_addr_ = reinterpret_cast<char*>(gpu_agent()->system_allocator()(
        queue_size_, 0x1000,
        core::MemoryRegion::AllocateExecutable | core::MemoryRegion::AllocateUncached));
  }
  if (queue_start_addr_ == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  memset(queue_start_addr_, 0, queue_size_);
  return HSA_STATUS_SUCCESS;
}

void SdmaQueue::FreeQueueBuffer() {
  if (queue_start_addr_ != nullptr) {
    if (IsDeviceMemRingBuf()) {
      gpu_agent()->coarsegrain_deallocator()(queue_start_addr_);
    } else {
      gpu_agent()->system_deallocator()(queue_start_addr_);
    }
    queue_start_addr_ = nullptr;
  }
}

hsa_status_t SdmaQueue::Initialize() {
  hsa_status_t status = AllocateQueueBuffer();
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  AMD::GpuAgent* agent = gpu_agent();

  // SDMA queues created through this API always target a specific engine using
  // HSA_QUEUE_SDMA_BY_ENG_ID. XGMI-optimized engines live in the upper part of
  // the same engine-id space, so no separate HSA_QUEUE_SDMA_XGMI type is
  // needed here. This requires kernel support for targeted SDMA engines.
  if (!agent->SupportsSdmaQueueByEngineId()) {
    FreeQueueBuffer();
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  const uint32_t num_engines = agent->NumSdmaEnginesTotal();
  if (num_engines == 0) {
    FreeQueueBuffer();
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  // Resolve the engine: an explicit id is used as-is; automatic selection
  // rotates round-robin across all SDMA engines so concurrently created queues
  // spread across the hardware.
  const uint32_t engine_id = (sdma_engine_id_ >= 0) ? static_cast<uint32_t>(sdma_engine_id_)
                                                    : agent->NextSdmaUserQueueEngineId();
  sdma_engine_id_ = static_cast<int32_t>(engine_id);

  const uint32_t kQueuePercent = 100;  // Request the full KFD queue scheduling allocation.
  status = agent->driver().CreateQueue(
      agent->node_id(), HSA_QUEUE_SDMA_BY_ENG_ID, kQueuePercent,
      HSA::HSA_AMD_QUEUE_PRIORITY_MAXIMUM,  // Ignored by lower layers for SDMA queues.
      engine_id, queue_start_addr_, queue_size_, 0, nullptr, queue_resource_);

  if (status != HSA_STATUS_SUCCESS) {
    FreeQueueBuffer();
    return status;
  }
  active_ = true;

  queue_wptr_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_write_ptr);
  queue_rptr_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_read_ptr);
  queue_doorbell_ = reinterpret_cast<volatile uint64_t*>(queue_resource_.Queue_DoorBell);
  if (queue_wptr_ == nullptr || queue_rptr_ == nullptr || queue_doorbell_ == nullptr) {
    Inactivate();
    FreeQueueBuffer();
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_SINGLE;
  amd_queue_.hsa_queue.features = 0;
  amd_queue_.hsa_queue.base_address = queue_start_addr_;
  amd_queue_.hsa_queue.doorbell_signal = core::Signal::Convert(this);
  // SDMA write/read pointers are byte offsets, so the public queue size is the
  // ring capacity in bytes for SDMA queues.
  amd_queue_.hsa_queue.size = static_cast<uint32_t>(queue_size_);
  amd_queue_.hsa_queue.id = queue_resource_.QueueId;
  amd_queue_.write_dispatch_id = 0;
  amd_queue_.read_dispatch_id = 0;

  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.hardware_doorbell_ptr = queue_doorbell_;
  signal_.queue_ptr = &amd_queue_;

  core::Runtime::runtime_singleton_->RegisterSdmaErrorHandler(
      public_handle(), agent->node_id(), errors_callback_, errors_data_);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t SdmaQueue::Inactivate() {
  if (!active_) {
    return HSA_STATUS_SUCCESS;
  }

  active_ = false;
  hsa_status_t status = gpu_agent()->driver().DestroyQueue(queue_resource_.QueueId);
  memset(&queue_resource_, 0, sizeof(queue_resource_));
  queue_wptr_ = nullptr;
  queue_rptr_ = nullptr;
  queue_doorbell_ = nullptr;
  return status;
}

hsa_status_t SdmaQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_ENGINE_TYPE:
      *static_cast<hsa_amd_queue_engine_t*>(value) = HSA_AMD_QUEUE_ENGINE_SDMA;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID:
      *static_cast<uint32_t*>(value) = static_cast<uint32_t>(sdma_engine_id_);
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      *static_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(queue_doorbell_);
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_READ_POINTER:
      *static_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(queue_rptr_);
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_QUEUE_INFO_WRITE_POINTER:
      *static_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(queue_wptr_);
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t SdmaQueue::RingDoorbell(uint64_t write_index) {
  if (queue_wptr_ == nullptr || queue_doorbell_ == nullptr) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  // Publish step of the submission protocol. SDMA queues are externally
  // synchronized: by the time the doorbell is stored the caller must have
  // written complete packets into the ring and handled wrap/space checks.
  //
  // The public write-index operations use queue_wptr_ directly. Ensure the
  // canonical write pointer and packet stores are visible before the doorbell.
  atomic::Store(queue_wptr_, write_index, std::memory_order_release);
  std::atomic_thread_fence(std::memory_order_release);
  *queue_doorbell_ = write_index;

  if (core::Runtime::runtime_singleton_->thunkLoader()->IsDXG() ||
      core::Runtime::runtime_singleton_->thunkLoader()->IsDTIF()) {
    HSAKMT_CALL(hsaKmtQueueRingDoorbell(queue_resource_.QueueId, write_index));
  }

  return HSA_STATUS_SUCCESS;
}

// Public read/write-index operations use KFD's canonical hardware control
// words. Updating the write pointer stages work; the engine is notified only
// when the caller subsequently rings the doorbell.
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

// SDMA is exposed as HSA_QUEUE_TYPE_SINGLE. The CAS/add operations are present
// only because hsa_queue_t helpers dispatch through core::Queue; callers must
// not treat them as a BlitSdma-style multi-producer reservation protocol.
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

void SdmaQueue::StoreRelaxed(hsa_signal_value_t value) {
  RingDoorbell(static_cast<uint64_t>(value));
}

void SdmaQueue::StoreRelease(hsa_signal_value_t value) {
  RingDoorbell(static_cast<uint64_t>(value));
}

}  // namespace AMD
}  // namespace rocr
