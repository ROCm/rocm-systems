////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_MACOS_AQL_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_AMD_MACOS_AQL_QUEUE_H_

#if !defined(__APPLE__)
#error "amd_macos_aql_queue.h is Darwin-only"
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/inc/amd_macos_agent.h"
#include "core/inc/amd_macos_driver.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"

namespace rocr {
namespace AMD {

class MacAqlQueue : public core::Queue,
                    private core::LocalSignal,
                    public core::DoorbellSignal {
 public:
  static __forceinline bool IsType(core::Queue* queue) { return queue->IsType(&rtti_id()); }

  MacAqlQueue(core::SharedQueue* shared_queue, MacGpuAgent* agent, size_t req_size_pkts,
              uint64_t flags, core::HsaEventCallback callback, void* err_data);
  ~MacAqlQueue() override;

  hsa_status_t Inactivate() override;
  hsa_status_t SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) override;
  void Destroy() override;

  uint64_t LoadReadIndexRelaxed() override;
  uint64_t LoadReadIndexAcquire() override;
  uint64_t LoadWriteIndexRelaxed() override;
  uint64_t LoadWriteIndexAcquire() override;
  void StoreReadIndexRelaxed(uint64_t value) override;
  void StoreReadIndexRelease(uint64_t value) override;
  void StoreWriteIndexRelaxed(uint64_t value) override;
  void StoreWriteIndexRelease(uint64_t value) override;
  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override;
  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override;
  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override;
  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override;
  uint64_t AddWriteIndexRelaxed(uint64_t value) override;
  uint64_t AddWriteIndexAcquire(uint64_t value) override;
  uint64_t AddWriteIndexRelease(uint64_t value) override;
  uint64_t AddWriteIndexAcqRel(uint64_t value) override;

  void StoreRelaxed(hsa_signal_value_t value) override;
  void StoreRelease(hsa_signal_value_t value) override;

  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) override;
  hsa_status_t GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) override;
  hsa_status_t SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) override;
  void ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b,
                  hsa_fence_scope_t acquireFence = HSA_FENCE_SCOPE_NONE,
                  hsa_fence_scope_t releaseFence = HSA_FENCE_SCOPE_NONE,
                  hsa_signal_t* signal = nullptr) override;

 protected:
  bool _IsA(Queue::rtti_t id) const override { return id == &rtti_id(); }

 private:
  hsa_status_t SubmitPackets(uint64_t doorbell_value);
  hsa_status_t SubmitKernel(const hsa_kernel_dispatch_packet_t& packet);
  hsa_status_t SubmitBarrier(const hsa_barrier_and_packet_t& packet, bool is_or);
  hsa_status_t SubmitPm4AndWait(const std::vector<uint32_t>& pm4);
  hsa_status_t AllocateDispatchScratch(size_t size, size_t align, void** cpu, uint64_t* gpu);
  // Lazily allocate (grow-only) a dedicated GPU private-segment scratch backing
  // buffer for scratch-using kernels. Distinct from the dispatch-staging scratch
  // above (scratch_cpu_/AllocateDispatchScratch), which is a CPU-coherent kernarg
  // staging bump allocator, NOT GPU private memory.
  hsa_status_t EnsureGpuScratch(size_t size);
  void CompleteSignal(hsa_signal_t signal, hsa_signal_value_t value);
  void ReportAsyncError(hsa_status_t status);

  MacGpuAgent& agent_;
  MacOsDriver& driver_;
  core::HsaEventCallback errors_callback_;
  void* errors_data_;
  void* ring_buf_ = nullptr;
  uint32_t queue_size_pkts_ = 0;
  std::atomic<bool> active_;

  MacOsDriver::DirectComputeQueue direct_queue_{};
  void* marker_cpu_base_ = nullptr;
  volatile uint32_t* marker_cpu_ = nullptr;
  uint64_t marker_gpu_ = 0;
  uint32_t marker_value_ = 0;

  void* scratch_cpu_ = nullptr;
  uint64_t scratch_gpu_ = 0;
  size_t scratch_size_ = 0;
  size_t scratch_offset_ = 0;

  // Dedicated GPU private-segment (scratch) backing for kernels that spill to
  // private memory. Lazily allocated, grow-only; programmed via
  // COMPUTE_DISPATCH_SCRATCH_BASE + COMPUTE_TMPRING_SIZE on gfx12.
  void* gpu_scratch_cpu_ = nullptr;
  uint64_t gpu_scratch_gpu_ = 0;
  size_t gpu_scratch_size_ = 0;
  // Last scratch base/tmpring pushed into the queue MQD (avoid re-mapping the
  // queue when unchanged).
  uint64_t last_scratch_base_256_ = 0;
  uint32_t last_scratch_tmpring_ = 0;

  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_MACOS_AQL_QUEUE_H_
