////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/amd_aql_queue.h"

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#endif

#include <stdio.h>
#include <string.h>

#include "core/inc/runtime.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/signal.h"
#include "core/inc/queue.h"
#include "core/util/utils.h"
#include "core/inc/registers.h"
#include "core/inc/interrupt_signal.h"
#include "core/inc/default_signal.h"
#include "core/inc/hsa_ext_amd_impl.h"
#include "core/inc/amd_gpu_pm4.h"
#include "core/inc/hsa_amd_tool_int.hpp"
#include "core/inc/amd_core_dump.hpp"

namespace rocr {
namespace AMD {

#define SCRATCH_ALT_RATIO 4

AqlQueue::AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, size_t req_size_pkts,
                   HSAuint32 node_id, ScratchInfo& scratch, core::HsaEventCallback callback,
                   void* err_data, uint64_t flags, bool is_kv)
    : Queue(shared_queue, flags, !agent->is_xgmi_cpu_gpu()),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      ring_buf_(nullptr),
      ring_buf_alloc_bytes_(0),
      queue_id_(HSA_QUEUEID(-1)),
      active_(false),
      agent_(agent),
      queue_scratch_(scratch),
      errors_callback_(callback),
      errors_data_(err_data),
      is_kv_queue_(is_kv),
      pm4_ib_buf_(nullptr),
      pm4_ib_size_b_(0x1000),
      dynamicScratchState(0),
      exceptionState(0),
      suspended_(false),
      priority_(HSA_QUEUE_PRIORITY_NORMAL),
      exception_signal_(nullptr) {

  const core::Isa* isa = agent_->supported_isas()[0];

  // Queue size is a function of several restrictions.
  const uint32_t min_pkts = ComputeRingBufferMinPkts();
  const uint32_t max_pkts = ComputeRingBufferMaxPkts();

  // Apply sizing constraints to the ring buffer.
  uint32_t queue_size_pkts = uint32_t(req_size_pkts);
  queue_size_pkts = Min(queue_size_pkts, max_pkts);
  queue_size_pkts = Max(queue_size_pkts, min_pkts);

  uint32_t queue_size_bytes = queue_size_pkts * sizeof(core::AqlPacket);
  if ((queue_size_bytes & (queue_size_bytes - 1)) != 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
                             "Requested queue with non-power of two packet capacity.\n");

  // Allocate the AQL packet ring buffer.
  AllocRegisteredRingBuffer(queue_size_pkts);
  if (ring_buf_ == nullptr) throw std::bad_alloc();
  MAKE_NAMED_SCOPE_GUARD(RingGuard, [&]() { FreeQueueMemory(); });

  // Fill the ring buffer with invalid packet headers.
  // Leave packet content uninitialized to help track errors.
  for (uint32_t pkt_id = 0; pkt_id < queue_size_pkts; ++pkt_id) {
    (((core::AqlPacket*)ring_buf_)[pkt_id]).dispatch.header = HSA_PACKET_TYPE_INVALID;
  }

  // Zero the amd_queue_ structure to clear RPTR/WPTR before queue attach.
  memset(&amd_queue_, 0, sizeof(amd_queue_));

  // Initialize and map a HW AQL queue.
  HsaQueueResource queue_rsrc = {0};
  queue_rsrc.Queue_read_ptr_aql = (uint64_t*)&amd_queue_.read_dispatch_id;

  // Hardware write pointer supports AQL semantics.
  queue_rsrc.Queue_write_ptr_aql = (uint64_t*)&amd_queue_.write_dispatch_id;

  // Populate amd_queue_ structure.
  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_MULTI;
  amd_queue_.hsa_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
  amd_queue_.hsa_queue.base_address = ring_buf_;
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);
  amd_queue_.hsa_queue.size = queue_size_pkts;
  amd_queue_.hsa_queue.id = INVALID_QUEUEID;
  amd_queue_.read_dispatch_id_field_base_byte_offset = uint32_t(
      uintptr_t(&amd_queue_.read_dispatch_id) - uintptr_t(&amd_queue_));
  // Initialize the doorbell signal structure.
  memset(&signal_, 0, sizeof(signal_));
  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.hardware_doorbell_ptr = nullptr;
  signal_.queue_ptr = &amd_queue_;

  const auto& props = agent->properties();
  amd_queue_.max_cu_id = (props.NumFComputeCores / props.NumSIMDPerCU) - 1;
  amd_queue_.max_wave_id = (props.MaxWavesPerSIMD * props.NumSIMDPerCU) - 1;

#ifdef HSA_LARGE_MODEL
  AMD_HSA_BITS_SET(amd_queue_.queue_properties, AMD_QUEUE_PROPERTIES_IS_PTR64,
                   1);
#else
  AMD_HSA_BITS_SET(amd_queue_.queue_properties, AMD_QUEUE_PROPERTIES_IS_PTR64,
                   0);
#endif

  // Set group and private memory apertures in amd_queue_.
  auto& regions = agent->regions();

  for (auto region : regions) {
    const MemoryRegion* amdregion = static_cast<const AMD::MemoryRegion*>(region);
    uint64_t base = amdregion->GetBaseAddress();

    if (amdregion->IsLDS()) {
#ifdef HSA_LARGE_MODEL
      amd_queue_.group_segment_aperture_base_hi =
          uint32_t(uintptr_t(base) >> 32);
#else
      amd_queue_.group_segment_aperture_base_hi = uint32_t(base);
#endif
    }

    if (amdregion->IsScratch()) {
#ifdef HSA_LARGE_MODEL
      amd_queue_.private_segment_aperture_base_hi =
          uint32_t(uintptr_t(base) >> 32);
#else
      amd_queue_.private_segment_aperture_base_hi = uint32_t(base);
#endif
    }
  }

  assert(amd_queue_.group_segment_aperture_base_hi != 0 && "No group region found.");

  if (core::Runtime::runtime_singleton_->flag().check_flat_scratch()) {
    assert(amd_queue_.private_segment_aperture_base_hi != 0 && "No private region found.");
  }

  if (agent_->supported_isas()[0]->GetMajorVersion() >= 11)
    queue_scratch_.mem_alignment_size = 256;
  else
    queue_scratch_.mem_alignment_size = 1024;

  queue_scratch_.use_once_limit = core::Runtime::runtime_singleton_->flag().scratch_single_limit();
  if (queue_scratch_.use_once_limit > agent_->MaxScratchDevice()) {
    fprintf(stdout, "User specified scratch limit exceeds device limits (requested:%lu max:%lu)!\n",
                    queue_scratch_.use_once_limit, agent_->MaxScratchDevice());
    queue_scratch_.use_once_limit = agent_->MaxScratchDevice();
  }

  queue_scratch_.use_alt_limit = 0;

  queue_scratch_.async_reclaim = agent_->AsyncScratchReclaimEnabled();
  if (queue_scratch_.async_reclaim) {
    queue_scratch_.use_once_limit = agent_->ScratchSingleLimitAsyncThreshold();
    queue_scratch_.use_alt_limit = core::Runtime::runtime_singleton_->flag().enable_scratch_alt()
        ? (queue_scratch_.use_once_limit / SCRATCH_ALT_RATIO)
        : 0;
  }

  MAKE_NAMED_SCOPE_GUARD(EventGuard, [&]() {
    ScopedAcquire<KernelMutex> _lock(&queue_lock());
    queue_count()--;
    if (queue_count() == 0) {
      core::InterruptSignal::DestroyEvent(queue_event());
      queue_event() = nullptr;
    }
  });

  MAKE_NAMED_SCOPE_GUARD(SignalGuard, [&]() {
    if (amd_queue_.queue_inactive_signal.handle != 0)
      HSA::hsa_signal_destroy(amd_queue_.queue_inactive_signal);
    if (exception_signal_ != nullptr) exception_signal_->DestroySignal();
  });

  if (core::g_use_interrupt_wait) {
    ScopedAcquire<KernelMutex> _lock(&queue_lock());
    queue_count()++;
    if (queue_event() == nullptr) {
      assert(queue_count() == 1 && "Inconsistency in queue event reference counting found.\n");

      queue_event() = core::InterruptSignal::CreateEvent(HSA_EVENTTYPE_SIGNAL, false);
      if (queue_event() == nullptr)
        throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                 "Queue event creation failed.\n");
    }
    auto Signal = new core::InterruptSignal(0, queue_event());
    assert(Signal != nullptr && "Should have thrown!\n");
    amd_queue_.queue_inactive_signal = core::InterruptSignal::Convert(Signal);
    exception_signal_ = new core::InterruptSignal(0, queue_event());
    assert(exception_signal_ != nullptr && "Should have thrown!\n");
  } else {
    EventGuard.Dismiss();
    auto Signal = new core::DefaultSignal(0);
    assert(Signal != nullptr && "Should have thrown!\n");
    amd_queue_.queue_inactive_signal = core::DefaultSignal::Convert(Signal);
    exception_signal_ = new core::DefaultSignal(0);
    assert(exception_signal_ != nullptr && "Should have thrown!\n");
  }

  // Make sure the queue signal always has a waiting_ > 0 so that
  // so that we call hsakmtSetEvent to force hsaKmtWaitOnEvent to return.
  exception_signal_->WaitingInc();

  // Ensure the amd_queue_ is fully initialized before creating the KFD queue.
  // This ensures that the debugger can access the fields once it detects there
  // is a KFD queue. The debugger may access the aperture addresses, queue
  // scratch base, and queue type.

  HSAKMT_STATUS kmt_status;
  if (core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging) {
    queue_rsrc.ErrorReason = &exception_signal_->signal_.value;
    kmt_status = HSAKMT_CALL(hsaKmtCreateQueueExt(node_id, HSA_QUEUE_COMPUTE_AQL, 100, priority_, 0, ring_buf_,
                                   ring_buf_alloc_bytes_, queue_event(), &queue_rsrc));
  } else {
    kmt_status = HSAKMT_CALL(hsaKmtCreateQueueExt(node_id, HSA_QUEUE_COMPUTE_AQL, 100, priority_, 0, ring_buf_,
                                   ring_buf_alloc_bytes_, NULL, &queue_rsrc));
  }
  if (kmt_status != HSAKMT_STATUS_SUCCESS)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                             "Queue create failed at hsaKmtCreateQueue\n");
  // Complete populating the doorbell signal structure.
  signal_.hardware_doorbell_ptr = queue_rsrc.Queue_DoorBell_aql;

  // Bind Id of Queue such that is unique i.e. it is not re-used by another
  // queue (AQL, HOST) in the same process during its lifetime.
  amd_queue_.hsa_queue.id = this->GetQueueId();

  queue_id_ = queue_rsrc.QueueId;
  MAKE_NAMED_SCOPE_GUARD(QueueGuard, [&]() { HSAKMT_CALL(hsaKmtDestroyQueue(queue_id_)); });

  amd_queue_.scratch_max_use_index = UINT64_MAX;
  amd_queue_.alt_scratch_max_use_index = UINT64_MAX;

  // Set flag to notify CP FW that SW supports the new amd_queue_v2
  if (agent_->AsyncScratchReclaimEnabled())
    amd_queue_.caps |= AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM;

  // On the first queue creation, reserve some scratch memory on this agent.
  agent_->ReserveScratch();

  // Initialize scratch memory related entities
  queue_scratch_.queue_retry = amd_queue_.queue_inactive_signal;
  InitScratchSRD();

  if (core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging) {
    if (AMD::hsa_amd_signal_async_handler(amd_queue_.queue_inactive_signal, HSA_SIGNAL_CONDITION_NE,
                                          0, DynamicQueueEventsHandler<false>,
                                          this) != HSA_STATUS_SUCCESS)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                               "Queue event handler failed registration.\n");
    if (AMD::hsa_amd_signal_async_handler(core::Signal::Convert(exception_signal_),
                                          HSA_SIGNAL_CONDITION_NE, 0, ExceptionHandler,
                                          this) != HSA_STATUS_SUCCESS)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                               "Queue event handler failed registration.\n");
  } else {
    if (AMD::hsa_amd_signal_async_handler(amd_queue_.queue_inactive_signal, HSA_SIGNAL_CONDITION_NE,
                                          0, DynamicQueueEventsHandler<true>,
                                          this) != HSA_STATUS_SUCCESS)
      throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                               "Queue event handler failed registration.\n");
    exceptionState = ERROR_HANDLER_DONE;
  }

  // Allocate IB for icache flushes.
  pm4_ib_buf_ =
      agent_->system_allocator()(pm4_ib_size_b_, 0x1000, core::MemoryRegion::AllocateExecutable);
  if (pm4_ib_buf_ == nullptr)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES, "PM4 IB allocation failed.\n");

  MAKE_NAMED_SCOPE_GUARD(PM4IBGuard, [&]() { agent_->system_deallocator()(pm4_ib_buf_); });

  // Set initial CU mask
  if (!core::Runtime::runtime_singleton_->flag().cu_mask_skip_init()) SetCUMasking(0, nullptr);

  active_ = true;

  PM4IBGuard.Dismiss();
  RingGuard.Dismiss();
  QueueGuard.Dismiss();
  EventGuard.Dismiss();
  SignalGuard.Dismiss();
}

AqlQueue::~AqlQueue() {
  // Remove error handler synchronously.
  // Sequences error handler callbacks with queue destroy.
  dynamicScratchState |= ERROR_HANDLER_TERMINATE;
  while ((dynamicScratchState & ERROR_HANDLER_DONE) != ERROR_HANDLER_DONE) {
    HSA::hsa_signal_store_screlease(amd_queue_.queue_inactive_signal, 0x8000000000000000ull);
    HSA::hsa_signal_wait_relaxed(amd_queue_.queue_inactive_signal, HSA_SIGNAL_CONDITION_NE,
                                 0x8000000000000000ull, -1ull, HSA_WAIT_STATE_BLOCKED);
  }

  // Remove kfd exception handler
  if (core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging) {
    exceptionState |= ERROR_HANDLER_TERMINATE;
    while ((exceptionState & ERROR_HANDLER_DONE) != ERROR_HANDLER_DONE) {
      const uint64_t timeout_ms = 5000;

      exception_signal_->StoreRelease(-1ull);
      exception_signal_->WaitRelaxed(HSA_SIGNAL_CONDITION_NE, -1ull, timeout_ms,
                                     HSA_WAIT_STATE_BLOCKED);
    }
  }

  Inactivate();

  if (queue_scratch_.main_queue_base) agent_->ReleaseQueueMainScratch(queue_scratch_);
  if (queue_scratch_.alt_queue_base) agent_->ReleaseQueueAltScratch(queue_scratch_);

  exception_signal_->WaitingDec();
  exception_signal_->DestroySignal();
  HSA::hsa_signal_destroy(amd_queue_.queue_inactive_signal);
  FreeQueueMemory();

  if (core::g_use_interrupt_wait) {
    ScopedAcquire<KernelMutex> lock(&queue_lock());
    queue_count()--;
    if (queue_count() == 0) {
      core::InterruptSignal::DestroyEvent(queue_event());
      queue_event() = nullptr;
    }
  }
  agent_->system_deallocator()(pm4_ib_buf_);
}

void AqlQueue::Destroy() {
  if (amd_queue_.hsa_queue.type == HSA_QUEUE_TYPE_COOPERATIVE) {
    agent_->GWSRelease();
    return;
  }
  delete this;
}

uint64_t AqlQueue::LoadReadIndexAcquire() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
}

uint64_t AqlQueue::LoadReadIndexRelaxed() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
}

uint64_t AqlQueue::LoadWriteIndexAcquire() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
}

uint64_t AqlQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
}

void AqlQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_relaxed);
}

void AqlQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_release);
}

uint64_t AqlQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acq_rel);
}
uint64_t AqlQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acquire);
}
uint64_t AqlQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_relaxed);
}
uint64_t AqlQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_release);
}

uint64_t AqlQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acq_rel);
}

uint64_t AqlQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acquire);
}

uint64_t AqlQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_relaxed);
}

uint64_t AqlQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_release);
}

void AqlQueue::StoreRelaxed(hsa_signal_value_t value) {
  if (core::Runtime::runtime_singleton_->flag().enable_dtif()) {
    HSAKMT_CALL(hsaKmtQueueRingDoorbell(queue_id_));
  } else {
    // Hardware doorbell supports AQL semantics.
    _mm_sfence();
    *(signal_.hardware_doorbell_ptr) = uint64_t(value);
    /* signal_ is allocated as uncached so we do not need read-back to flush WC */
  }
  return;
}

void AqlQueue::StoreRelease(hsa_signal_value_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}

hsa_status_t AqlQueue::GetInfo(hsa_queue_info_attribute_t attribute, void* value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
      *(reinterpret_cast<hsa_agent_t*>(value)) = agent_->public_handle();
      break;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      *(reinterpret_cast<uint64_t*>(value)) =
          reinterpret_cast<uint64_t>(signal_.hardware_doorbell_ptr);
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

uint32_t AqlQueue::ComputeRingBufferMinPkts() {
  // From CP_HQD_PQ_CONTROL.QUEUE_SIZE specification:
  //   Size of the primary queue (PQ) will be: 2^(HQD_QUEUE_SIZE+1) DWs.
  //   Min Size is 7 (2^8 = 256 DWs) and max size is 29 (2^30 = 1 G-DW)
  uint32_t min_bytes = 0x400;

  return uint32_t(min_bytes / sizeof(core::AqlPacket));
}

uint32_t AqlQueue::ComputeRingBufferMaxPkts() {
  // From CP_HQD_PQ_CONTROL.QUEUE_SIZE specification:
  //   Size of the primary queue (PQ) will be: 2^(HQD_QUEUE_SIZE+1) DWs.
  //   Min Size is 7 (2^8 = 256 DWs) and max size is 29 (2^30 = 1 G-DW)
  uint64_t max_bytes = 0x100000000;

  return uint32_t(max_bytes / sizeof(core::AqlPacket));
}

void AqlQueue::AllocRegisteredRingBuffer(uint32_t queue_size_pkts) {
  // Allocate storage for the ring buffer.
  ring_buf_alloc_bytes_ = queue_size_pkts * sizeof(core::AqlPacket);
  assert(IsMultipleOf(ring_buf_alloc_bytes_, 4096) && "Ring buffer sizes must be 4KiB aligned.");

  if (IsDeviceMemRingBuf()) {
    if (!agent_->LargeBarEnabled()) {
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
                                "Trying to allocate an AQL ring buffer in device memory without "
                                "large BAR PCIe enabled.");
    }
    ring_buf_ = agent_->coarsegrain_allocator()(
        ring_buf_alloc_bytes_,
        core::MemoryRegion::AllocateExecutable | core::MemoryRegion::AllocateUncached);
  } else {
    ring_buf_ = agent_->system_allocator()(
        ring_buf_alloc_bytes_, 0x1000,
        core::MemoryRegion::AllocateExecutable);
  }

  assert(ring_buf_ != NULL && "AQL queue memory allocation failure");
}

void AqlQueue::FreeQueueMemory() {
  if (shared_queue_) {
    if (IsDeviceMemQueueDescriptor())
      agent_->coarsegrain_deallocator()(shared_queue_);
    else
      core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);

    shared_queue_ = nullptr;
  }

  if (ring_buf_) {
    if (IsDeviceMemRingBuf()) {
      agent_->coarsegrain_deallocator()(ring_buf_);
    } else {
      agent_->system_deallocator()(ring_buf_);
    }
  }

  ring_buf_ = NULL;
  ring_buf_alloc_bytes_ = 0;
}

void AqlQueue::CloseRingBufferFD(const char* ring_buf_shm_path, int fd) const {
#ifdef __linux__
#if !defined(HAVE_MEMFD_CREATE)
  shm_unlink(ring_buf_shm_path);
#endif
  close(fd);
#else
  assert(false && "Function only needed on Linux.");
#endif
}

int AqlQueue::CreateRingBufferFD(const char* ring_buf_shm_path,
                                 uint32_t ring_buf_phys_size_bytes) const {
#ifdef __linux__
  int fd;
#ifdef HAVE_MEMFD_CREATE
  fd = syscall(__NR_memfd_create, ring_buf_shm_path, 0);

  if (fd == -1) return -1;

  if (ftruncate(fd, ring_buf_phys_size_bytes) == -1) {
    CloseRingBufferFD(ring_buf_shm_path, fd);
    return -1;
  }
#else
  fd = shm_open(ring_buf_shm_path, O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR);

  if (fd == -1) return -1;

  if (posix_fallocate(fd, 0, ring_buf_phys_size_bytes) != 0) {
    CloseRingBufferFD(ring_buf_shm_path, fd);
    return -1;
  }
#endif
  return fd;
#else
  assert(false && "Function only needed on Linux.");
  return -1;
#endif
}

void AqlQueue::Suspend() {
  suspended_ = true;
  auto err = HSAKMT_CALL(hsaKmtUpdateQueue(queue_id_, 0, priority_, ring_buf_, ring_buf_alloc_bytes_, NULL));
  assert(err == HSAKMT_STATUS_SUCCESS && "hsaKmtUpdateQueue failed.");
}

void AqlQueue::Resume() {
  if (suspended_) {
    suspended_ = false;
    auto err = HSAKMT_CALL(hsaKmtUpdateQueue(queue_id_, 100, priority_, ring_buf_, ring_buf_alloc_bytes_, NULL));
    assert(err == HSAKMT_STATUS_SUCCESS && "hsaKmtUpdateQueue failed.");
  }
}

hsa_status_t AqlQueue::Inactivate() {
  bool active = active_.exchange(false, std::memory_order_relaxed);
  if (active) {
    auto err = HSAKMT_CALL(hsaKmtDestroyQueue(queue_id_));
    assert(err == HSAKMT_STATUS_SUCCESS && "hsaKmtDestroyQueue failed.");
    atomic::Fence(std::memory_order_acquire);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AqlQueue::SetPriority(HSA_QUEUE_PRIORITY priority) {
  if (suspended_) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  priority_ = priority;
  auto err = HSAKMT_CALL(hsaKmtUpdateQueue(queue_id_, 100, priority_, ring_buf_, ring_buf_alloc_bytes_, NULL));
  return (err == HSAKMT_STATUS_SUCCESS ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES);
}

void AqlQueue::CheckScratchLimits() {
  auto& scratch = queue_scratch_;
  if (!scratch.async_reclaim) return;

  scratch.use_once_limit = agent_->ScratchSingleLimitAsyncThreshold();
  scratch.use_alt_limit = core::Runtime::runtime_singleton_->flag().enable_scratch_alt()
      ? (scratch.use_once_limit / SCRATCH_ALT_RATIO)
      : 0;

  if (scratch.main_size > scratch.use_once_limit)
    AsyncReclaimMainScratch();

  if (scratch.alt_size > scratch.use_alt_limit)
    AsyncReclaimAltScratch();

  return;
}

void AqlQueue::FreeMainScratchSpace() {
  auto& scratch = queue_scratch_;
  agent_->ReleaseQueueMainScratch(scratch);
  scratch.main_size = 0;
  scratch.main_size_per_thread = 0;
  scratch.main_queue_process_offset = 0;
  InitScratchSRD();
}

void AqlQueue::AsyncReclaimMainScratch() {
  /*
   * Pseudocode for scratch memory management when asynchronous scratch is
   * supported
   *
   * Notes:
   * - CP FW only updates its copy of amd_queue_ (scratch_copy) on queue_connect
   * so changes to amd_queue_ by ROCr are only visible to CP FW after a queue
   * re-map.
   *
   * - CP sets AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM bit to indicate that this version
   * of CP FW supports asynchronous scratch reclaim. But CP will only update
   * amd_queue_.caps on queue-connect so ROCr assumes that async scratch reclaim
   * is supported based on the CP FW version.
   *
   * - ROCR sets AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM bit to indicate to CP that this
   * version of FW supports asynchronous scratch and therefore CP is allowed to
   * access the extra fields that exist in amd_queue_v2.
   *
   * CP FW Pseudocode:
   * On doorbell-ring:
   * <start>
   *    Start processing AQL dispatch packet at read_index
   *    if (packet->private_segment_size > 0) {
   *      // This dispatch needs scratch
   *      if (packet->private_segment_size <= scratch_copy.scratch_wave64_lane_byte_size) {
   *         if (read_index <= scratch_max_use_index) {
   *           scratch_copy->scratch_last_used_index = current_index
   *           dispatch-uses-primary-scratch
   *           goto proceed-with-dispatch
   *         }
   *      } else if (packet->private_segment_size <= scratch_copy.alt_scratch_wave64_lane_byte_size
   *              && packet->grid_size_x <= scratch_copy.alt_scratch_dispatch_limit_x
   *              && packet->grid_size_y <= scratch_copy.alt_scratch_dispatch_limit_y
   *              && packet->grid_size_z <= scratch_copy.alt_scratch_dispatch_limit_z) {
   *         if (read_index <= alt_scratch_max_use_index) {
   *           scratch_copy->alt_scratch_last_used_index = current_index
   *           dispatch-uses-alternate-scratch
   *           goto proceed-with-dispatch
   *         }
   *      }
   *      request-more-scratch
   *    }
   *    goto proceed-with-dispatch
   * <end>
   *
   * On queue-connect:
   * <start>
   *    set AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM to indicate that this version of CP
   *    FW supports asynchronous scratch reclaim
   * <end>
   *
   * On queue-disconnect:
   * <start>
   *     // This guarantees that ROCr sees updated values of scratch_last_used_index
   *     // and alt_scratch_last_used_index after queue is unmapped.
   *     queue->scratch_last_used_index= scratch_copy->scratch_last_used_index
   *     queue->alt_scratch_last_used_index= scratch_copy->alt_scratch_last_used_index
   * <end>
   *
   * ROCr Pseudocode:
   * On init:
   *     queue->scratch_max_use_index = UINT64_MAX
   *     queue->alt_scratch_max_use_index = UINT64_MAX
   *
   * To reclaim scratch:
   * <start>
   *      // mutex blocks async-thread in case CP raises signal to request more scratch
   *     acquire(scratch-mutex)
   *     queue-unmap
   *     // Tell CP that it cannot use scratch after current packet
   *     queue->scratch_last_used_index = max(amd_queue_->scratch_last_used_index_per_xcc[])
   *
   *     queue-map
   *     // wait for CP to finish current packet
   *     while (queue->max_scratch_use_index >= queue->read_dispatch_id)
   *         sched_yield();
   *
   *     free-scratch
   *     release(scratch-mutex)
   * <end>
   */
  auto getMaxMainScratchUseIndex = [&]() {
    uint64_t max = 0;
    for (int i = 0; i < agent_->properties().NumXcc; i++) {
      if (amd_queue_.scratch_last_used_index[i].main > max)
        max = amd_queue_.scratch_last_used_index[i].main;
    }
    return max;
  };

  auto& scratch = queue_scratch_;
  if (!scratch.async_reclaim || !scratch.main_size) {
    return;
  }

  assert((amd_queue_.caps & AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM) &&
          "This version of CP FW should support async scratch, but flag is not set");

  tool::notify_event_scratch_async_reclaim_start(public_handle(),
                                                 HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_NONE);

  ScopedAcquire<KernelMutex> lock(&scratch_lock_);

  // Unmap the queue. CP will check amd_queue_ fields on re-map
  Suspend();

  /*
   * amd_queue_.scratch_last_used_index[*].main is updated by CP FW every time a
   * dispatch packet is launched and it needs scratch memory.
   * If amd_queue_.scratch_last_used_index[*].main >= amd_queue_.read_dispatch_id
   * then this XCC is currently running a dispatch that uses scratch.
   * Setting max_scratch_use_index to max(amd_queue_.scratch_last_used_index[*].main)
   * prevents CP from trying to use main-scratch after
   * amd_queue_.scratch_max_use_index. If CP sees a dispatch that needs scratch,
   * it will raise a new signal. CP may use alt-scratch in the meantime.
   */
  amd_queue_.scratch_max_use_index = getMaxMainScratchUseIndex();

  Resume();

  // If current dispatch is using scratch, wait for it to finish
  while (amd_queue_.scratch_max_use_index >= LoadReadIndexRelaxed()) {
    //TODO: if mwaitx supported, //mwaitx(amd_queue_.read_dispatch_id);
    os::YieldThread();
  }

  FreeMainScratchSpace();
  tool::notify_event_scratch_async_reclaim_end(public_handle(),
                                                HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_NONE);

  return;
}

void AqlQueue::FreeAltScratchSpace() {
  auto& scratch = queue_scratch_;
  agent_->ReleaseQueueAltScratch(scratch);
  scratch.alt_size = 0;
  scratch.alt_size_per_thread = 0;
  scratch.alt_queue_process_offset = 0;
  InitScratchSRD();
}

void AqlQueue::AsyncReclaimAltScratch() {
  /*
   * See AsyncReclaimMainScratch() for scratch reclaim handshake protocol with
   * CP FW.
   */
  auto getMaxAltScratchUseIndex = [&]() {
    uint64_t max = 0;
    for (int i = 0; i < agent_->properties().NumXcc; i++) {
      if (amd_queue_.scratch_last_used_index[i].alt > max)
        max = amd_queue_.scratch_last_used_index[i].alt;
    }
    return max;
  };

  auto& scratch = queue_scratch_;
  if (!scratch.async_reclaim || !scratch.alt_size) {
    return;
  }

  assert((amd_queue_.caps & AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM) &&
          "This version of CP FW should support async scratch, but flag is not set");

  tool::notify_event_scratch_async_reclaim_start(public_handle(),
                                                 HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT);

  ScopedAcquire<KernelMutex> lock(&scratch_lock_);

  // Unmap the queue. CP will check amd_queue_ fields on re-map
  Suspend();

  amd_queue_.alt_scratch_max_use_index = getMaxAltScratchUseIndex();

  Resume();

  // If current dispatch is using alt scratch, wait for it to finish
  while (amd_queue_.alt_scratch_max_use_index >= LoadReadIndexRelaxed()) {
    //TODO: if mwaitx supported, //mwaitx(amd_queue_.read_dispatch_id);
    os::YieldThread();
  }

  FreeAltScratchSpace();
  tool::notify_event_scratch_async_reclaim_end(public_handle(),
                                                HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT);
  return;
}

void AqlQueue::HandleInsufficientScratch(hsa_signal_value_t& error_code,
                                         hsa_signal_value_t& waitVal, bool& changeWait) {
  // Insufficient scratch - recoverable, don't process dynamic scratch if errors are present.
  auto& scratch = queue_scratch_;

  /*******************************************************************************************
   * uint32_t max_scratch_slots;   // Maximum number of slots for this device based on num CUs
   * uint64_t dispatch_slots;      // Number of slots wanted for this dispatch
   *
   * uint64_t all_slots_size;      // Size needed to fill all slots on this device
   * uint64_t dispatch_size;       // Size needed to fill wanted slots for this dispatch
   *
   * //Default values:
   * size_t use_once_limit = 128 MB      // When async reclaim not supported
   *                                     // DEFAULT_SCRATCH_SINGLE_LIMIT
   *                       = 3GB per-XCC // When async reclaim is supported
   *                                     // DEFAULT_SCRATCH_SINGLE_LIMIT_ASYNC_PER_XCC
   *
   * size_t use_alt_limit  = 768 MB per-XCC // use_once_limit/SCRATCH_ALT_RATIO
   *
   * if (async-scratch-reclaim-supported
   *     && dispatch_slots < max_scratch_slots
   *     && dispatch_size < use_alt_limit) {
   *   // This dispatch wants less waves than number of slots, use alternate scratch
   *   // alt_tmpring_size will have limited waves
   *  use_alt()
   * } else if (all_slots_size <= use_once_limit) {
   *  use_main()
   *
   *  //If we failed to allocate memory to fill all slots, scratch.use_once will be set
   *  if (scratch.use_once) {
   *    use_once
   *  } else if (all_slots_size > scratch.alt_size) {
   *    //Primary scratch is large enough to handle needs of alt-scratch
   *    free_alt()
   *  }
   * }
   *
   *******************************************************************************************/

  core::AqlPacket *pkt = NULL;
  uint64_t dispatch_id = UINT64_MAX;

  auto get_dispatch_pkt = [&]() {
    dispatch_id = amd_queue_.read_dispatch_id;
    do {
      // On GPUs where EOP is handled in asic, the read_dispatch_id is not
      // updated after each packet so look for the first dispatch that needs
      // scratch
      const uint64_t pkt_slot_idx =
          dispatch_id & (amd_queue_.hsa_queue.size - 1);

      core::AqlPacket *dispatch_pkt =
          &((core::AqlPacket *)amd_queue_.hsa_queue.base_address)[pkt_slot_idx];
      if (dispatch_pkt->IsDispatchAndNeedsScratch()) return dispatch_pkt;

      dispatch_id++;
    } while (dispatch_id <= LoadWriteIndexRelaxed());

    return (core::AqlPacket *)NULL;
  };

  auto calc_dispatch_waves_per_group = [&](core::AqlPacket& pkt) {
    const uint64_t lanes_per_group =
        (uint64_t(pkt.dispatch.workgroup_size_x) * pkt.dispatch.workgroup_size_y) *
        pkt.dispatch.workgroup_size_z;

    const uint32_t lanes_per_wave = (error_code & 0x400) ? 32 : 64;
    return (lanes_per_group + lanes_per_wave - 1) / lanes_per_wave;
  };

  auto calc_dispatch_groups = [&](core::AqlPacket& pkt) {
    const uint64_t lanes_per_group =
        (uint64_t(pkt.dispatch.workgroup_size_x) * pkt.dispatch.workgroup_size_y) *
        pkt.dispatch.workgroup_size_z;

    uint64_t groups = ((uint64_t(pkt.dispatch.grid_size_x) + pkt.dispatch.workgroup_size_x - 1) /
                       pkt.dispatch.workgroup_size_x) *
                      ((uint64_t(pkt.dispatch.grid_size_y) + pkt.dispatch.workgroup_size_y - 1) /
                       pkt.dispatch.workgroup_size_y) *
                      ((uint64_t(pkt.dispatch.grid_size_z) + pkt.dispatch.workgroup_size_z - 1) /
                       pkt.dispatch.workgroup_size_z);
    const uint32_t cu_count = amd_queue_.max_cu_id + 1;

    const uint32_t engines = agent_->properties().NumShaderBanks;

    const uint32_t symmetric_cus = AlignDown(cu_count, engines);
    const uint32_t asymmetryPerRound = cu_count - symmetric_cus;
    const uint64_t rounds = groups / cu_count;
    const uint64_t asymmetricGroups = rounds * asymmetryPerRound;
    const uint64_t symmetricGroups = groups - asymmetricGroups;
    uint64_t maxGroupsPerEngine =
        ((symmetricGroups + engines - 1) / engines) + (asymmetryPerRound ? rounds : 0);

    // For gfx10+ devices we must attempt to assign the smaller of 256 lanes or 16 groups to each
    // engine.
    if (agent_->supported_isas()[0]->GetMajorVersion() >= 10 &&
        maxGroupsPerEngine < 16 &&
                              lanes_per_group * maxGroupsPerEngine < 256) {
      uint64_t groups_per_interleave = (256 + lanes_per_group - 1) / lanes_per_group;
      maxGroupsPerEngine = Min(groups_per_interleave, 16ul);
    }

    // Populate all engines at max group occupancy, then clip down to device limits.
    return maxGroupsPerEngine * engines;
  };

  // TODO: Move this to queue constructor since it does not depend on pkt, must be re-computed if
  // CU Masking is enabled
  auto calc_device_slots = [&]() {
    // Get the hw maximum scratch slot count taking into consideration asymmetric harvest.
    const uint32_t engines = agent_->properties().NumShaderBanks;
    const uint32_t cu_count = amd_queue_.max_cu_id + 1;
    return AlignUp(cu_count, engines) * agent_->properties().MaxSlotsScratchCU;
  };

  assert(core::Runtime::runtime_singleton_->flag().enable_scratch_async_reclaim() &&
         (!scratch.async_reclaim || (amd_queue_.caps & AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM)) &&
          "Asynchronous scratch reclaim capability not set, but this FW version should support it");

  scratch.cooperative = (amd_queue_.hsa_queue.type == HSA_QUEUE_TYPE_COOPERATIVE);

  pkt = get_dispatch_pkt(); // Sets dispatch_id
  assert((pkt && dispatch_id != UINT64_MAX) &&
         "Could not find dispatch packet with private_segment_size > 0");

  tool::notify_event_scratch_alloc_start(
      public_handle(), HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_NONE, dispatch_id);

  uint32_t device_slots = calc_device_slots();
  uint32_t groups = calc_dispatch_groups(*pkt);
  uint32_t waves_per_group = calc_dispatch_waves_per_group(*pkt);

  uint32_t dispatch_slots = groups * waves_per_group;
  dispatch_slots = std::min(dispatch_slots, device_slots);

  const uint64_t lanes_per_wave = (error_code & 0x400) ? 32 : 64;

  const uint64_t size_per_thread =
      AlignUp(pkt->dispatch.private_segment_size,
              scratch.mem_alignment_size / lanes_per_wave);
  const uint64_t device_size = size_per_thread * lanes_per_wave * device_slots;
  const uint64_t dispatch_size = size_per_thread * lanes_per_wave * dispatch_slots;

  ScopedAcquire<KernelMutex> lock(&scratch_lock_);

  // scratch.use_alt_limit will be 0 if alt scratch is not supported or disabled
  if (dispatch_size < scratch.use_alt_limit && dispatch_slots < device_slots) {
    // Try to use ALT scratch
    agent_->ReleaseQueueAltScratch(scratch);

    scratch.alt_size = dispatch_size;
    scratch.alt_size_per_thread = size_per_thread;
    scratch.alt_lanes_per_wave = lanes_per_wave;
    scratch.alt_waves_per_group = waves_per_group;

    agent_->AcquireQueueAltScratch(scratch);
    if (scratch.alt_queue_base) {
      scratch.alt_dispatch_limit_x = pkt->dispatch.grid_size_x;
      scratch.alt_dispatch_limit_y = pkt->dispatch.grid_size_y;
      scratch.alt_dispatch_limit_z = pkt->dispatch.grid_size_z;

      InitScratchSRD();
      /*
       * Indicate to CP FW that any dispatch may use alt scratch memory.
       * If ROCr wants to reclain scratch memory, it will set
       * amd_queue_.alt_scratch_max_use_index to a lower value
       */
      amd_queue_.alt_scratch_max_use_index = UINT64_MAX;
      // Restart the queue.
      HSA::hsa_signal_store_screlease(amd_queue_.queue_inactive_signal, 0);
      tool::notify_event_scratch_alloc_end(public_handle(), HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT,
                                           dispatch_id, scratch.alt_size, dispatch_slots);
      return;
    }
    // Could not allocate enough memory for alternate scratch fallback to primary scratch
    scratch.alt_size = 0;
    scratch.alt_size_per_thread = 0;
  }

  // Use PRIMARY scratch
  agent_->ReleaseQueueMainScratch(scratch);
  scratch.main_size = device_size;
  scratch.main_size_per_thread = size_per_thread;
  scratch.main_lanes_per_wave = lanes_per_wave;
  scratch.main_waves_per_group = waves_per_group;

  scratch.dispatch_size = dispatch_size;
  scratch.dispatch_slots = dispatch_slots;

  agent_->AcquireQueueMainScratch(scratch);

  if (scratch.retry) {
    dynamicScratchState |= ERROR_HANDLER_SCRATCH_RETRY;
    changeWait = true;
    waitVal = error_code;
  } else if (scratch.main_queue_base == nullptr) {
    // We could not allocate memory to fit even 1 wave
    tool::notify_event_scratch_alloc_end(public_handle(), HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE,
                                         dispatch_id, scratch.main_size, dispatch_slots);
    return;
  }

  // If we had to reduce number of waves
  if (scratch.large) {
    amd_queue_.queue_properties |= AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE;
    // Set system release fence to flush scratch stores with older firmware versions.
    if ((agent_->supported_isas()[0]->GetMajorVersion() == 8) && (agent_->GetMicrocodeVersion() < 729)) {
      pkt->dispatch.header &=
          ~(((1 << HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE) - 1)
            << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      pkt->dispatch.header |=
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    }
  } else if (scratch.alt_size && scratch.main_size > scratch.alt_size) {
    // Not using use-scratch-once, and dispatches that would fit in alt-scratch would also fit in
    // main scratch. No need for alt-scratch.
    tool::notify_event_scratch_async_reclaim_start(public_handle(),
                                                 HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT);
    FreeAltScratchSpace();
    tool::notify_event_scratch_async_reclaim_end(public_handle(),
                                                 HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT);
  }

  // Reset scratch memory related entities for the queue
  InitScratchSRD();
  /*
   * Indicate to CP FW that any dispatch may use alt scratch memory.
   * If ROCr wants to reclain scratch memory, it will set
   * amd_queue_.alt_scratch_max_use_index to a lower value
   */
  amd_queue_.scratch_max_use_index = UINT64_MAX;

  // Restart the queue.
  HSA::hsa_signal_store_screlease(amd_queue_.queue_inactive_signal, 0);

  auto alloc_flag = (scratch.large) ? HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE
                                    : HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_NONE;

  tool::notify_event_scratch_alloc_end(public_handle(), alloc_flag, dispatch_id, scratch.main_size,
                                       dispatch_slots);

  return;
}

template <bool HandleExceptions>
bool AqlQueue::DynamicQueueEventsHandler(hsa_signal_value_t error_code, void* arg) {
  AqlQueue* queue = (AqlQueue*)arg;
  hsa_status_t errorCode = HSA_STATUS_SUCCESS;
  bool fatal = false;
  bool changeWait = false;
  hsa_signal_value_t waitVal;

  if ((queue->dynamicScratchState & ERROR_HANDLER_SCRATCH_RETRY) == ERROR_HANDLER_SCRATCH_RETRY) {
    queue->dynamicScratchState &= ~ERROR_HANDLER_SCRATCH_RETRY;
    changeWait = true;
    waitVal = 0;
    HSA::hsa_signal_and_relaxed(queue->amd_queue_.queue_inactive_signal, ~0x8000000000000000ull);
    error_code &= ~0x8000000000000000ull;
  }

  // Process errors only if queue is not terminating.
  if ((queue->dynamicScratchState & ERROR_HANDLER_TERMINATE) != ERROR_HANDLER_TERMINATE) {
    if (error_code == 512) {  // Large scratch reclaim
      tool::notify_event_scratch_free_start(queue->public_handle(),
                                            HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE);

      auto& scratch = queue->queue_scratch_;
      queue->agent_->ReleaseQueueMainScratch(scratch);
      scratch.main_queue_base = nullptr;
      scratch.main_size = 0;
      scratch.main_size_per_thread = 0;
      scratch.main_queue_process_offset = 0;
      queue->InitScratchSRD();

      HSA::hsa_signal_store_relaxed(queue->amd_queue_.queue_inactive_signal, 0);
      // Resumes queue processing.
      atomic::Store(&queue->amd_queue_.queue_properties,
                    queue->amd_queue_.queue_properties & (~AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE),
                    std::memory_order_release);
      atomic::Fence(std::memory_order_release);
      tool::notify_event_scratch_free_end(queue->public_handle(),
                                          HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE);
      return true;
    }

    // Process only one queue error.
    if (error_code & 0x401) {  // insufficient scratch, wave64 or wave32
      queue->HandleInsufficientScratch(error_code, waitVal, changeWait);

      // Out of scratch - promote error
      if (queue->queue_scratch_.main_queue_base == nullptr &&
          queue->queue_scratch_.alt_queue_base == nullptr)
        errorCode = HSA_STATUS_ERROR_OUT_OF_RESOURCES;


    } else if (HandleExceptions) {
      if ((error_code & 2) == 2) {  // Invalid dim
        errorCode = HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;

      } else if ((error_code & 4) == 4) {  // Invalid group memory
        errorCode = HSA_STATUS_ERROR_INVALID_ALLOCATION;

      } else if ((error_code & 8) == 8) {  // Invalid (or NULL) code
        errorCode = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

      } else if (((error_code & 32) == 32) ||    // Invalid format: 32 is generic,
                 ((error_code & 256) == 256)) {  // 256 is vendor specific packets
        errorCode = HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;

      } else if ((error_code & 64) == 64) {  // Group is too large
        errorCode = HSA_STATUS_ERROR_INVALID_ARGUMENT;

      } else if ((error_code & 128) == 128) {  // Out of VGPRs
        errorCode = hsa_status_t(HSA_STATUS_ERROR_OUT_OF_REGISTERS);

      } else if ((error_code & 0x20000000) == 0x20000000) {  // Memory violation (>48-bit)
        errorCode = hsa_status_t(HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION);

      } else if ((error_code & 0x40000000) == 0x40000000) {  // Illegal instruction
        errorCode = hsa_status_t(HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION);

      } else if ((error_code & 0x80000000) == 0x80000000) {  // Debug trap
        errorCode = HSA_STATUS_ERROR_EXCEPTION;
        fatal = true;

      } else {  // Undefined code
        assert(false && "Undefined queue error code");
        errorCode = HSA_STATUS_ERROR;
        fatal = true;
      }
    } else {
      // Not handling exceptions, clear so that ExceptionHandler can run.
      HSA::hsa_signal_store_relaxed(queue->amd_queue_.queue_inactive_signal, 0);
    }

    if (errorCode == HSA_STATUS_SUCCESS) {
      if (changeWait) {
        core::Runtime::runtime_singleton_->SetAsyncSignalHandler(
            queue->amd_queue_.queue_inactive_signal, HSA_SIGNAL_CONDITION_NE, waitVal,
            DynamicQueueEventsHandler<HandleExceptions>, queue);
        return false;
      }
      return true;
    }

    queue->Suspend();
    if (queue->errors_callback_ != nullptr) {
      queue->errors_callback_(errorCode, queue->public_handle(), queue->errors_data_);
    }
    if (fatal) {
      // Temporarilly removed until there is clarity on exactly what debugtrap's semantics are.
      // assert(false && "Fatal queue error");
      // std::abort();
    }
  }
  // Copy here is to protect against queue being released between setting the scratch state and
  // updating the signal value.  The signal itself is safe to use because it is ref counted rather
  // than being released with the queue.
  hsa_signal_t signal = queue->amd_queue_.queue_inactive_signal;
  queue->dynamicScratchState = ERROR_HANDLER_DONE;
  HSA::hsa_signal_store_screlease(signal, -1ull);
  return false;
}

bool AqlQueue::ExceptionHandler(hsa_signal_value_t error_code, void* arg) {
  struct queue_error_t {
    uint32_t code;
    hsa_status_t status;
  };
  static const queue_error_t QueueErrors[] = {
      // EC_QUEUE_WAVE_ABORT
      { 1, HSA_STATUS_ERROR_EXCEPTION },
      // EC_QUEUE_WAVE_TRAP
      { 2, HSA_STATUS_ERROR_EXCEPTION },
      // EC_QUEUE_WAVE_MATH_ERROR
      { 3, HSA_STATUS_ERROR_EXCEPTION },
      // EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION
      { 4, (hsa_status_t)HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION },
      // EC_QUEUE_WAVE_MEMORY_VIOLATION
      { 5, (hsa_status_t)HSA_STATUS_ERROR_MEMORY_FAULT },
      // EC_QUEUE_WAVE_APERTURE_VIOLATION
      { 6, (hsa_status_t)HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION },
      // EC_QUEUE_PACKET_DISPATCH_DIM_INVALID
      { 16, HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS },
      // EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID
      { 17, HSA_STATUS_ERROR_INVALID_ALLOCATION },
      // EC_QUEUE_PACKET_DISPATCH_CODE_INVALID
      { 18, HSA_STATUS_ERROR_INVALID_CODE_OBJECT },
      // EC_QUEUE_PACKET_UNSUPPORTED
      { 20, HSA_STATUS_ERROR_INVALID_PACKET_FORMAT },
      // EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID
      { 21, HSA_STATUS_ERROR_INVALID_ARGUMENT },
      // EC_QUEUE_PACKET_DISPATCH_REGISTER_SIZE_INVALID
      { 22, HSA_STATUS_ERROR_INVALID_ISA },
      // EC_QUEUE_PACKET_VENDOR_UNSUPPORTED
      { 23, HSA_STATUS_ERROR_INVALID_PACKET_FORMAT },
      // EC_QUEUE_PREEMPTION_ERROR
      { 31, HSA_STATUS_ERROR },
      // EC_DEVICE_MEMORY_VIOLATION
      { 33, (hsa_status_t)HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION },
      // EC_DEVICE_RAS_ERROR
      { 34, HSA_STATUS_ERROR },
      // EC_DEVICE_FATAL_HALT
      { 35, HSA_STATUS_ERROR },
      // EC_DEVICE_NEW
      { 36, HSA_STATUS_ERROR },
      // EC_PROCESS_DEVICE_REMOVE
      { 50, HSA_STATUS_ERROR }};

  AqlQueue* queue = (AqlQueue*)arg;
  hsa_status_t errorCode = HSA_STATUS_ERROR;

  if (queue->exceptionState == ERROR_HANDLER_TERMINATE) {
    Signal* signal = queue->exception_signal_;
    queue->exceptionState = ERROR_HANDLER_DONE;
    signal->StoreRelease(0);
    return false;
  }

  for (auto& error : QueueErrors) {
    if (error_code & (1UL << (error.code - 1))) {
      errorCode = error.status;
      break;
    }
  }

  // Undefined or unexpected code
  assert((errorCode != HSA_STATUS_ERROR) && "Undefined or unexpected queue error code");

  // Suppress VM fault reporting.  This is more useful when reported through the system error
  // handler.
  if (errorCode == static_cast<hsa_status_t>(HSA_STATUS_ERROR_MEMORY_FAULT)) {
    debug_print("Queue error - HSA_STATUS_ERROR_MEMORY_FAULT\n");
    return false;
  }

  // Fallback if KFD does not support GPU core dump. In this case, there core dump is
  // generated by hsa-runtime.
  if (!core::Runtime::runtime_singleton_->KfdVersion().supports_core_dump &&
                queue->agent_->supported_isas()[0]->GetMajorVersion() != 11) {

    if (pcs::PcsRuntime::instance()->SessionsActive())
      fprintf(stderr, "GPU core dump skipped because PC Sampling active\n");
    else if (amd::coredump::dump_gpu_core())
      fprintf(stderr, "GPU core dump failed\n");
    // supports_core_dump flag is overwritten to avoid generate core dump file again
    // caught by a different exception handler. Such as VMFaultHandler.
    core::Runtime::runtime_singleton_->KfdVersion(
      core::Runtime::runtime_singleton_->KfdVersion().supports_exception_debugging, true);
  }

  queue->Suspend();
  if (queue->errors_callback_ != nullptr) {
    queue->errors_callback_(errorCode, queue->public_handle(), queue->errors_data_);
  }
  Signal* signal = queue->exception_signal_;
  queue->exceptionState = ERROR_HANDLER_DONE;
  signal->StoreRelease(0);
  return false;
}

hsa_status_t AqlQueue::SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
  uint32_t cu_count;
  agent_->GetInfo((hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cu_count);
  size_t mask_dwords = (cu_count + 31) / 32;
  // Mask to trim the last uint32_t in cu_mask to the physical CU count
  uint32_t tail_mask = (1 << (cu_count % 32)) - 1;

  auto global_mask = core::Runtime::runtime_singleton_->flag().cu_mask(agent_->enumeration_index());
  std::vector<uint32_t> mask;

  bool clipped = false;

  // num_cu_mask_count = 0 resets the CU mask.
  if (num_cu_mask_count == 0) {
    for (int i = 0; i < mask_dwords; i++) mask.push_back(-1);
  } else {
    for (int i = 0; i < num_cu_mask_count / 32; i++) mask.push_back(cu_mask[i]);
  }

  // Apply global mask to user mask
  if (!global_mask.empty()) {
    // Limit mask processing to smallest needed dword range
    size_t limit = Min(global_mask.size(), mask.size(), mask_dwords);

    // Check for disabling requested cus.
    for (int i = limit; i < mask.size(); i++) {
      if (mask[i] != 0) {
        clipped = true;
        break;
      }
    }

    mask.resize(limit, 0);
    for (size_t i = 0; i < limit; i++) {
      clipped |= ((mask[i] & (~global_mask[i])) != 0);
      mask[i] &= global_mask[i];
    }
  } else {
    // Limit to physical CU range only
    size_t limit = Min(mask.size(), mask_dwords);
    mask.resize(limit, 0);
  }

  // Clip last dword to physical CU limit if necessary
  if ((mask.size() == mask_dwords) && (tail_mask != 0)) mask[mask_dwords - 1] &= tail_mask;

  // Apply mask if non-default or not queue initialization.
  ScopedAcquire<KernelMutex> lock(&mask_lock_);
  if ((!cu_mask_.empty()) || (num_cu_mask_count != 0) || (!global_mask.empty())) {

    // Devices with WGPs must conform to even-indexed contiguous pairwise CU enablement.
    if (agent_->supported_isas()[0]->GetMajorVersion() >= 10) {
      for (int i = 0; i < mask.size() * 32; i += 2) {
        uint32_t cu_pair = (mask[i / 32] >> (i % 32)) & 0x3;
        if (cu_pair && cu_pair != 0x3) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
    }

    HSAKMT_STATUS ret =
        HSAKMT_CALL(hsaKmtSetQueueCUMask(queue_id_, mask.size() * 32, reinterpret_cast<HSAuint32*>(&mask[0])));
    if (ret != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR;
  }

  // update current cu masking tracking.
  cu_mask_ = std::move(mask);
  return clipped ? (hsa_status_t)HSA_STATUS_CU_MASK_REDUCED : HSA_STATUS_SUCCESS;
}

hsa_status_t AqlQueue::GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) {
  ScopedAcquire<KernelMutex> lock(&mask_lock_);
  assert(!cu_mask_.empty() && "No current cu_mask!");

  uint32_t user_dword_count = num_cu_mask_count / 32;
  if (user_dword_count > cu_mask_.size()) {
    memset(&cu_mask[cu_mask_.size()], 0, sizeof(uint32_t) * (user_dword_count - cu_mask_.size()));
    user_dword_count = cu_mask_.size();
  }
  memcpy(cu_mask, &cu_mask_[0], sizeof(uint32_t) * user_dword_count);
  return HSA_STATUS_SUCCESS;
}

void AqlQueue::SetProfiling(bool enabled) {
  Queue::SetProfiling(enabled);

  if (enabled) agent_->CheckClockTicks();
  return;
}

// If in_signal is NULL then this ExecutePM4 will block and wait for PM4 commands to complete
// If in_signal is provided, then ExecutePM4 will return and caller may wait for in_signal
// Note: On gfx8, there is no completion signal support, so ExecutePM4 will block even if
// in_signal is provided, and it is still valid to check in_signal after ExecutePM4 returns.
void AqlQueue::ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b, hsa_fence_scope_t acquireFence,
                          hsa_fence_scope_t releaseFence, hsa_signal_t* in_signal) {
  // pm4_ib_buf_ is a shared resource, so mutually exclude here.
  ScopedAcquire<KernelMutex> lock(&pm4_ib_mutex_);

  // Obtain reference to any container queue.
  core::Queue* queue = core::Queue::Convert(public_handle());

  // Obtain a queue slot for a single AQL packet.
  uint64_t write_idx = queue->AddWriteIndexAcqRel(1);

  while ((write_idx - queue->LoadReadIndexRelaxed()) >= queue->amd_queue_.hsa_queue.size) {
    os::YieldThread();
  }

  uint32_t slot_idx = uint32_t(write_idx % queue->amd_queue_.hsa_queue.size);
  constexpr uint32_t slot_size_b = 0x40;
  uint32_t* queue_slot =
      (uint32_t*)(uintptr_t(queue->amd_queue_.hsa_queue.base_address) + (slot_idx * slot_size_b));

  // Copy client PM4 command into IB.
  assert(cmd_size_b < pm4_ib_size_b_ && "PM4 exceeds IB size");
  memcpy(pm4_ib_buf_, cmd_data, cmd_size_b);

  // Construct a PM4 command to execute the IB.
  constexpr uint32_t ib_jump_size_dw = 4;

  uint32_t ib_jump_cmd[ib_jump_size_dw] = {
      PM4_HDR(PM4_HDR_IT_OPCODE_INDIRECT_BUFFER, ib_jump_size_dw,
                              agent_->supported_isas()[0]->GetMajorVersion()),
      PM4_INDIRECT_BUFFER_DW1_IB_BASE_LO(uint32_t(uintptr_t(pm4_ib_buf_) >> 2)),
      PM4_INDIRECT_BUFFER_DW2_IB_BASE_HI(uint32_t(uintptr_t(pm4_ib_buf_) >> 32)),
      (PM4_INDIRECT_BUFFER_DW3_IB_SIZE(uint32_t(cmd_size_b / sizeof(uint32_t))) |
       PM4_INDIRECT_BUFFER_DW3_IB_VALID(1))};

  // To respect multi-producer semantics, first buffer commands for the queue slot.
  constexpr uint32_t slot_size_dw = uint32_t(slot_size_b / sizeof(uint32_t));
  uint32_t slot_data[slot_size_dw];
  hsa_signal_t local_signal = {0};
  hsa_status_t err;

  if (agent_->supported_isas()[0]->GetMajorVersion() <= 8) {
    // Construct a set of PM4 to fit inside the AQL packet slot.
    uint32_t slot_dw_idx = 0;

    // Construct a no-op command to pad the queue slot.
    constexpr uint32_t rel_mem_size_dw = 7;
    constexpr uint32_t nop_pad_size_dw = slot_size_dw - (ib_jump_size_dw + rel_mem_size_dw);

    uint32_t* nop_pad = &slot_data[slot_dw_idx];
    slot_dw_idx += nop_pad_size_dw;

    nop_pad[0] = PM4_HDR(PM4_HDR_IT_OPCODE_NOP, nop_pad_size_dw,
                              agent_->supported_isas()[0]->GetMajorVersion());

    for (uint32_t i = 1; i < nop_pad_size_dw; ++i) {
      nop_pad[i] = 0;
    }

    // Copy in command to execute the IB.
    assert(slot_dw_idx + ib_jump_size_dw <= slot_size_dw && "PM4 exceeded queue slot size");
    uint32_t* ib_jump = &slot_data[slot_dw_idx];
    slot_dw_idx += ib_jump_size_dw;

    memcpy(ib_jump, ib_jump_cmd, sizeof(ib_jump_cmd));

    // Construct a command to advance the read index and invalidate the packet
    // header. This must be the last command since this releases the queue slot
    // for writing.
    assert(slot_dw_idx + rel_mem_size_dw <= slot_size_dw && "PM4 exceeded queue slot size");
    uint32_t* rel_mem = &slot_data[slot_dw_idx];

    rel_mem[0] = PM4_HDR(PM4_HDR_IT_OPCODE_RELEASE_MEM, rel_mem_size_dw,
                              agent_->supported_isas()[0]->GetMajorVersion());
    rel_mem[1] = PM4_RELEASE_MEM_DW1_EVENT_INDEX(PM4_RELEASE_MEM_EVENT_INDEX_AQL);
    rel_mem[2] = 0;
    rel_mem[3] = 0;
    rel_mem[4] = 0;
    rel_mem[5] = 0;
    rel_mem[6] = 0;
  } else if (agent_->supported_isas()[0]->GetMajorVersion() >= 9) {
    // Construct an AQL packet to jump to the PM4 IB.
    struct amd_aql_pm4_ib {
      uint16_t header;
      uint16_t ven_hdr;
      uint32_t ib_jump_cmd[4];
      uint32_t dw_cnt_remain;
      uint32_t reserved[8];
      hsa_signal_t completion_signal;
    };

    if (!in_signal) {
      err = hsa_signal_create(1, 0, NULL, &local_signal);
      assert(err == HSA_STATUS_SUCCESS);
    }

    constexpr uint32_t AMD_AQL_FORMAT_PM4_IB = 0x1;

    amd_aql_pm4_ib aql_pm4_ib{};
    aql_pm4_ib.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE |
                        (acquireFence << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                        (releaseFence << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

    aql_pm4_ib.ven_hdr = AMD_AQL_FORMAT_PM4_IB;
    aql_pm4_ib.ib_jump_cmd[0] = ib_jump_cmd[0];
    aql_pm4_ib.ib_jump_cmd[1] = ib_jump_cmd[1];
    aql_pm4_ib.ib_jump_cmd[2] = ib_jump_cmd[2];
    aql_pm4_ib.ib_jump_cmd[3] = ib_jump_cmd[3];
    aql_pm4_ib.dw_cnt_remain = 0xA;
    aql_pm4_ib.completion_signal = in_signal ? *in_signal : local_signal;

    memcpy(slot_data, &aql_pm4_ib, sizeof(aql_pm4_ib));
  } else {
    assert(false && "AqlQueue::ExecutePM4 not implemented");
  }

  // Copy buffered commands into the queue slot.
  // Overwrite the AQL invalid header (first dword) last.
  // This prevents the slot from being read until it's fully written.
  memcpy(&queue_slot[1], &slot_data[1], slot_size_b - sizeof(uint32_t));
  if (IsDeviceMemRingBuf() && needsPcieOrdering()) {
    // Ensure the packet body is written as header may get reordered when writing over PCIE
    _mm_sfence();
  }
  atomic::Store(&queue_slot[0], slot_data[0], std::memory_order_release);

  // Submit the packet slot.
  core::Signal* doorbell = core::Signal::Convert(queue->amd_queue_.hsa_queue.doorbell_signal);
  doorbell->StoreRelease(write_idx);

  // Wait for the packet to be consumed.
  if (agent_->supported_isas()[0]->GetMajorVersion() <= 8) {
    while (queue->LoadReadIndexRelaxed() <= write_idx)
      os::YieldThread();

    if (in_signal) hsa_signal_store_screlease(*in_signal, 0);
  } else if (!in_signal) {
    // On gfx9 and newer, if in_signal is not provided, we block and wait for own signal
    hsa_signal_value_t ret;
    ret = hsa_signal_wait_scacquire(local_signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                    HSA_WAIT_STATE_ACTIVE);
    err = hsa_signal_destroy(local_signal);
    assert(ret == 0 && err == HSA_STATUS_SUCCESS);
  }
}

void AqlQueue::FillBufRsrcWord0() {
  SQ_BUF_RSRC_WORD0 srd0;
  uintptr_t scratch_base = uintptr_t(queue_scratch_.main_queue_base);

  srd0.bits.BASE_ADDRESS = uint32_t(scratch_base);
  amd_queue_.scratch_resource_descriptor[0] = srd0.u32All;
}

void AqlQueue::FillBufRsrcWord1() {
  SQ_BUF_RSRC_WORD1 srd1;
  uint32_t scratch_base_hi = 0;

#ifdef HSA_LARGE_MODEL
  uintptr_t scratch_base = uintptr_t(queue_scratch_.main_queue_base);
  scratch_base_hi = uint32_t(scratch_base >> 32);
  #endif

  srd1.bits.BASE_ADDRESS_HI = scratch_base_hi;
  srd1.bits.STRIDE = 0;
  srd1.bits.CACHE_SWIZZLE = 0;
  srd1.bits.SWIZZLE_ENABLE = 1;

  amd_queue_.scratch_resource_descriptor[1] = srd1.u32All;
}

void AqlQueue::FillBufRsrcWord1_Gfx11() {
  SQ_BUF_RSRC_WORD1_GFX11 srd1;
  uint32_t scratch_base_hi = 0;

#ifdef HSA_LARGE_MODEL
  uintptr_t scratch_base = uintptr_t(queue_scratch_.main_queue_base);
  scratch_base_hi = uint32_t(scratch_base >> 32);
#endif

  srd1.bits.BASE_ADDRESS_HI = scratch_base_hi;
  srd1.bits.STRIDE = 0;
  srd1.bits.SWIZZLE_ENABLE = 1;

  amd_queue_.scratch_resource_descriptor[1] = srd1.u32All;
}

void AqlQueue::FillBufRsrcWord2() {
  SQ_BUF_RSRC_WORD2 srd2;
  const auto& agent_props = agent_->properties();
  const uint32_t num_xcc = agent_props.NumXcc;

   // report size per XCC
  srd2.bits.NUM_RECORDS = uint32_t(queue_scratch_.main_size / num_xcc);

  amd_queue_.scratch_resource_descriptor[2] = srd2.u32All;
}

void AqlQueue::FillBufRsrcWord3() {
  SQ_BUF_RSRC_WORD3 srd3;

  srd3.bits.DST_SEL_X = SQ_SEL_X;
  srd3.bits.DST_SEL_Y = SQ_SEL_Y;
  srd3.bits.DST_SEL_Z = SQ_SEL_Z;
  srd3.bits.DST_SEL_W = SQ_SEL_W;
  srd3.bits.NUM_FORMAT = BUF_NUM_FORMAT_UINT;
  srd3.bits.DATA_FORMAT = BUF_DATA_FORMAT_32;
  srd3.bits.ELEMENT_SIZE = 1;  // 4
  srd3.bits.INDEX_STRIDE = 3;  // 64
  srd3.bits.ADD_TID_ENABLE = 1;
  srd3.bits.ATC__CI__VI = (agent_->profile() == HSA_PROFILE_FULL);
  srd3.bits.HASH_ENABLE = 0;
  srd3.bits.HEAP = 0;
  srd3.bits.MTYPE__CI__VI = 0;
  srd3.bits.TYPE = SQ_RSRC_BUF;

  amd_queue_.scratch_resource_descriptor[3] = srd3.u32All;
}

void AqlQueue::FillBufRsrcWord3_Gfx10() {
  SQ_BUF_RSRC_WORD3_GFX10 srd3;

  srd3.bits.DST_SEL_X = SQ_SEL_X;
  srd3.bits.DST_SEL_Y = SQ_SEL_Y;
  srd3.bits.DST_SEL_Z = SQ_SEL_Z;
  srd3.bits.DST_SEL_W = SQ_SEL_W;
  srd3.bits.FORMAT = BUF_FORMAT_32_UINT;
  srd3.bits.RESERVED1 = 0;
  srd3.bits.INDEX_STRIDE = 0;  // filled in by CP
  srd3.bits.ADD_TID_ENABLE = 1;
  srd3.bits.RESOURCE_LEVEL = 1;
  srd3.bits.RESERVED2 = 0;
  srd3.bits.OOB_SELECT = 2;  // no bounds check in swizzle mode
  srd3.bits.TYPE = SQ_RSRC_BUF;

  amd_queue_.scratch_resource_descriptor[3] = srd3.u32All;
}

void AqlQueue::FillBufRsrcWord3_Gfx11() {
  SQ_BUF_RSRC_WORD3_GFX11 srd3;

  srd3.bits.DST_SEL_X = SQ_SEL_X;
  srd3.bits.DST_SEL_Y = SQ_SEL_Y;
  srd3.bits.DST_SEL_Z = SQ_SEL_Z;
  srd3.bits.DST_SEL_W = SQ_SEL_W;
  srd3.bits.FORMAT = BUF_FORMAT_32_UINT;
  srd3.bits.RESERVED1 = 0;
  srd3.bits.INDEX_STRIDE = 0;  // filled in by CP
  srd3.bits.ADD_TID_ENABLE = 1;
  srd3.bits.RESERVED2 = 0;
  srd3.bits.OOB_SELECT = 2;  // no bounds check in swizzle mode
  srd3.bits.TYPE = SQ_RSRC_BUF;

  amd_queue_.scratch_resource_descriptor[3] = srd3.u32All;
}

void AqlQueue::FillBufRsrcWord3_Gfx12() {
  SQ_BUF_RSRC_WORD3_GFX12 srd3;

  srd3.bits.DST_SEL_X = SQ_SEL_X;
  srd3.bits.DST_SEL_Y = SQ_SEL_Y;
  srd3.bits.DST_SEL_Z = SQ_SEL_Z;
  srd3.bits.DST_SEL_W = SQ_SEL_W;
  srd3.bits.FORMAT = BUF_FORMAT_32_UINT;
  srd3.bits.RESERVED1 = 0;
  srd3.bits.INDEX_STRIDE = 0;  // filled in by CP
  srd3.bits.ADD_TID_ENABLE = 1;
  srd3.bits.WRITE_COMPRESS_ENABLE = 0;
  srd3.bits.COMPRESSION_EN = 0;
  srd3.bits.COMPRESSION_ACCESS_MODE = 0;
  srd3.bits.OOB_SELECT = 2;  // no bounds check in swizzle mode
  srd3.bits.TYPE = SQ_RSRC_BUF;

  amd_queue_.scratch_resource_descriptor[3] = srd3.u32All;
}

// Set concurrent wavefront limits only when scratch is being used.
void AqlQueue::FillComputeTmpRingSize() {
  COMPUTE_TMPRING_SIZE tmpring_size = {};
  if (queue_scratch_.main_size == 0) {
    amd_queue_.compute_tmpring_size = tmpring_size.u32All;
    return;
  }

  const auto& agent_props = agent_->properties();
  const uint32_t num_xcc = agent_props.NumXcc;

  // Determine the maximum number of waves device can support
  uint32_t num_cus = agent_props.NumFComputeCores / agent_props.NumSIMDPerCU;
  uint32_t max_scratch_waves = num_cus * agent_props.MaxSlotsScratchCU;

  // Scratch is allocated program COMPUTE_TMPRING_SIZE register
  // Scratch Size per Wave is specified in terms of kilobytes
  uint32_t wave_scratch =
      (((queue_scratch_.main_lanes_per_wave * queue_scratch_.main_size_per_thread) +
        queue_scratch_.mem_alignment_size - 1) /
       queue_scratch_.mem_alignment_size);
  tmpring_size.bits.WAVESIZE = wave_scratch;
  assert(wave_scratch == tmpring_size.bits.WAVESIZE && "WAVESIZE Overflow.");
  uint32_t num_waves = (queue_scratch_.main_size / num_xcc) /
      (tmpring_size.bits.WAVESIZE * queue_scratch_.mem_alignment_size);

  tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves);
  amd_queue_.compute_tmpring_size = tmpring_size.u32All;
  assert((tmpring_size.bits.WAVES % (agent_props.NumShaderBanks / num_xcc) == 0) &&
         "Invalid scratch wave count.  Must be divisible by #SEs.");
}

// Set concurrent wavefront limits only when scratch is being used.
void AqlQueue::FillAltComputeTmpRingSize() {
  COMPUTE_TMPRING_SIZE tmpring_size = {};
  if (queue_scratch_.alt_size == 0) {
    amd_queue_.alt_compute_tmpring_size = tmpring_size.u32All;
    return;
  }

  const auto& agent_props = agent_->properties();
  const uint32_t num_xcc = agent_props.NumXcc;

  // Determine the maximum number of waves device can support
  uint32_t num_cus = agent_props.NumFComputeCores / agent_props.NumSIMDPerCU;
  uint32_t max_scratch_waves = num_cus * agent_props.MaxSlotsScratchCU;

  // Scratch is allocated program COMPUTE_TMPRING_SIZE register
  // Scratch Size per Wave is specified in terms of kilobytes
  uint32_t wave_scratch =
      (((queue_scratch_.alt_lanes_per_wave * queue_scratch_.alt_size_per_thread) +
        queue_scratch_.mem_alignment_size - 1) /
       queue_scratch_.mem_alignment_size);
  tmpring_size.bits.WAVESIZE = wave_scratch;
  assert(wave_scratch == tmpring_size.bits.WAVESIZE && "WAVESIZE Overflow.");
  uint32_t num_waves = (queue_scratch_.alt_size / num_xcc) /
      (tmpring_size.bits.WAVESIZE * queue_scratch_.mem_alignment_size);

  tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves);
  amd_queue_.alt_compute_tmpring_size = tmpring_size.u32All;
  assert((tmpring_size.bits.WAVES % (agent_props.NumShaderBanks / num_xcc) == 0) &&
         "Invalid scratch wave count.  Must be divisible by #SEs.");
}

// Set concurrent wavefront limits only when scratch is being used.
void AqlQueue::FillComputeTmpRingSize_Gfx11() {
  COMPUTE_TMPRING_SIZE_GFX11 tmpring_size = {};
  if (queue_scratch_.main_size == 0) {
    amd_queue_.compute_tmpring_size = tmpring_size.u32All;
    return;
  }

  const auto& agent_props = agent_->properties();
  const uint32_t num_xcc = agent_props.NumXcc;

  // Determine the maximum number of waves device can support
  uint32_t num_cus = agent_props.NumFComputeCores / (agent_props.NumSIMDPerCU * num_xcc);
  uint32_t max_scratch_waves = num_cus * agent_props.MaxSlotsScratchCU;

  // Scratch is allocated program COMPUTE_TMPRING_SIZE register
  // Scratch Size per Wave is specified in terms of kilobytes
  uint32_t wave_scratch =
      (((queue_scratch_.main_lanes_per_wave * queue_scratch_.main_size_per_thread) +
        queue_scratch_.mem_alignment_size - 1) /
       queue_scratch_.mem_alignment_size);

  tmpring_size.bits.WAVESIZE = wave_scratch;
  assert(wave_scratch == tmpring_size.bits.WAVESIZE && "WAVESIZE Overflow.");

  uint32_t num_waves =
      queue_scratch_.main_size / (tmpring_size.bits.WAVESIZE * queue_scratch_.mem_alignment_size);

  // For GFX11 we specify number of waves per engine instead of total
  num_waves /= agent_->properties().NumShaderBanks;
  tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves);
  amd_queue_.compute_tmpring_size = tmpring_size.u32All;
}

// Set concurrent wavefront limits only when scratch is being used.
void AqlQueue::FillComputeTmpRingSize_Gfx12() {
  // For GFX12, struct field size changes.
  // Consider refactoring code for GFX11/GFX12 if no other changes.
  COMPUTE_TMPRING_SIZE_GFX12 tmpring_size = {};
  if (queue_scratch_.main_size == 0) {
    amd_queue_.compute_tmpring_size = tmpring_size.u32All;
    return;
  }

  const auto& agent_props = agent_->properties();
  const uint32_t num_xcc = agent_props.NumXcc;

  // Determine the maximum number of waves device can support
  uint32_t num_cus = agent_props.NumFComputeCores / (agent_props.NumSIMDPerCU * num_xcc);
  uint32_t max_scratch_waves = num_cus * agent_props.MaxSlotsScratchCU;

  // Scratch is allocated program COMPUTE_TMPRING_SIZE register
  // Scratch Size per Wave is specified in terms of kilobytes
  uint32_t wave_scratch = (((queue_scratch_.main_lanes_per_wave * queue_scratch_.main_size_per_thread) +
                            queue_scratch_.mem_alignment_size - 1) /
                           queue_scratch_.mem_alignment_size);

  tmpring_size.bits.WAVESIZE = wave_scratch;
  assert(wave_scratch == tmpring_size.bits.WAVESIZE && "WAVESIZE Overflow.");

  uint32_t num_waves =
      queue_scratch_.main_size / (tmpring_size.bits.WAVESIZE * queue_scratch_.mem_alignment_size);

  // For GFX11 we specify number of waves per engine instead of total
  num_waves /= agent_->properties().NumShaderBanks;
  tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves);
  amd_queue_.compute_tmpring_size = tmpring_size.u32All;
}

// @brief Define the Scratch Buffer Descriptor and related parameters
// that enable kernel access scratch memory
void AqlQueue::InitScratchSRD() {
  switch (agent_->supported_isas()[0]->GetMajorVersion()) {
    case 12:
      FillBufRsrcWord0();
      FillBufRsrcWord1_Gfx11();
      FillBufRsrcWord2();
      FillBufRsrcWord3_Gfx12();
      FillComputeTmpRingSize_Gfx12();
      break;
    case 11:
      FillBufRsrcWord0();
      FillBufRsrcWord1_Gfx11();
      FillBufRsrcWord2();
      FillBufRsrcWord3_Gfx11();
      FillComputeTmpRingSize_Gfx11();
      break;
    case 10:
      FillBufRsrcWord0();
      FillBufRsrcWord1();
      FillBufRsrcWord2();
      FillBufRsrcWord3_Gfx10();
      FillComputeTmpRingSize();
      break;
    default:
      FillBufRsrcWord0();
      FillBufRsrcWord1();
      FillBufRsrcWord2();
      FillBufRsrcWord3();
      FillComputeTmpRingSize();
      FillAltComputeTmpRingSize();
      break;
  }

  // Populate flat scratch parameters in amd_queue_.
  amd_queue_.scratch_backing_memory_location = queue_scratch_.main_queue_process_offset;
  amd_queue_.alt_scratch_backing_memory_location = queue_scratch_.alt_queue_process_offset;

  const auto& agent_props = agent_->properties();
  const uint32_t num_xcc = agent_props.NumXcc;

  // FIXME:  amd_queue_.scratch_backing_memory_byte_size is not used by CP, but it
  // is used by the debugger. Putting back the scratch_backing_memory_byte_size
  // field. But we need to find a location for alt_scratch_backing_memory_byte_size

  // report size per XCC
  amd_queue_.scratch_backing_memory_byte_size = queue_scratch_.main_size / num_xcc;
  //amd_queue_.alt_scratch_backing_memory_byte_size = queue_scratch_.alt_size / num_xcc;

  // For backwards compatibility this field records the per-lane scratch
  // for a 64 lane wavefront. If scratch was allocated for 32 lane waves
  // then the effective size for a 64 lane wave is halved.
  amd_queue_.scratch_wave64_lane_byte_size =
      uint32_t((queue_scratch_.main_size_per_thread * queue_scratch_.main_lanes_per_wave) / 64);

  amd_queue_.alt_scratch_wave64_lane_byte_size =
      uint32_t((queue_scratch_.alt_size_per_thread * queue_scratch_.alt_lanes_per_wave) / 64);

  amd_queue_.alt_scratch_dispatch_limit_x = queue_scratch_.alt_dispatch_limit_x;
  amd_queue_.alt_scratch_dispatch_limit_y = queue_scratch_.alt_dispatch_limit_y;
  amd_queue_.alt_scratch_dispatch_limit_z = queue_scratch_.alt_dispatch_limit_z;

  return;
}

hsa_status_t AqlQueue::EnableGWS(int gws_slot_count) {
  uint32_t discard;
  auto status = HSAKMT_CALL(hsaKmtAllocQueueGWS(queue_id_, gws_slot_count, &discard));
  if (status != HSAKMT_STATUS_SUCCESS) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_COOPERATIVE;
  return HSA_STATUS_SUCCESS;
}

}  // namespace amd
}  // namespace rocr
