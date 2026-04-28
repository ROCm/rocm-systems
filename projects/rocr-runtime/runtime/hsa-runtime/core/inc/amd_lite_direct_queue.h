////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_LITE_DIRECT_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_AMD_LITE_DIRECT_QUEUE_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "inc/hsa.h"

namespace rocr {
namespace AMD {
namespace lite {

constexpr uint32_t kMqdSize = 0x1000;
constexpr uint32_t kMqdDwordCount = kMqdSize / sizeof(uint32_t);
constexpr uint32_t kDirectComputeRingSize = 0x1000;
constexpr uint32_t kDirectComputeEopSize = 0x1000;
constexpr uint32_t kDirectComputeDoorbellBase = 0x20;
constexpr uint32_t kDirectComputeDoorbellStride = 2;

struct DirectQueueLayout {
  uint64_t base_offset = 0;
  uint64_t mqd_offset = 0;
  uint64_t ring_offset = 0;
  uint64_t eop_offset = 0;
  uint64_t rptr_offset = 0;
  uint64_t wptr_offset = 0;
  uint64_t mqd_gpu = 0;
  uint64_t ring_gpu = 0;
  uint64_t eop_gpu = 0;
  uint64_t rptr_gpu = 0;
  uint64_t wptr_gpu = 0;
};

using DirectQueueMqd = std::array<uint32_t, kMqdDwordCount>;

struct DirectQueueState {
  uint32_t queue_id = 0;
  uint32_t queue_index = 0;
  uint32_t doorbell_index = 0;
  uint32_t ring_size_bytes = 0;
  uint64_t ring_gpu = 0;
  uint64_t wptr = 0;
  volatile uint32_t* ring_cpu = nullptr;
  volatile uint64_t* wptr_cpu = nullptr;
  volatile uint64_t* rptr_cpu = nullptr;
  volatile uint64_t* doorbell_cpu = nullptr;
};

struct DirectQueueOptions {
  bool force_reclaim = false;
  bool use_firmware_dequeue = true;
  bool skip_destroy = false;
  bool trace = false;
  bool trace_verbose = false;
  uint32_t dequeue_settle_us = 100000;
  uint32_t activate_sleep_us = 10000;
  const char* trace_prefix = "ROCR lite direct queue";
};

class DirectQueuePlatform {
 public:
  virtual ~DirectQueuePlatform() = default;

  virtual hsa_status_t EnsureDoorbellAperture() const = 0;
  virtual hsa_status_t ReadMmio32(uint32_t base, uint32_t reg,
                                  uint32_t* value) const = 0;
  virtual hsa_status_t WriteMmio32(uint32_t base, uint32_t reg,
                                   uint32_t value) const = 0;
  virtual hsa_status_t ZeroGpuMemory(uint64_t offset, uint64_t size) const = 0;
  virtual hsa_status_t WriteGpuMemory32(uint64_t offset, uint32_t value) const = 0;
  virtual void* GpuMemoryCpuPointer(uint64_t offset) const = 0;
  virtual volatile uint64_t* DoorbellCpuPointer(uint32_t doorbell_index) const = 0;
  virtual void SleepUs(uint32_t usec) const = 0;
};

uint32_t DirectQueuePipe(uint32_t queue_index);
uint32_t DirectQueueHqd(uint32_t queue_index);
uint32_t DirectQueueDoorbell(uint32_t queue_index);

DirectQueueLayout BuildDirectQueueLayout(uint64_t framebuffer_base,
                                         uint32_t queue_index);
DirectQueueMqd BuildPm4DirectQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index);

hsa_status_t CreateDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState* queue,
                               uint32_t queue_index,
                               uint64_t framebuffer_base,
                               const DirectQueueOptions& options);
hsa_status_t DestroyDirectQueue(const DirectQueuePlatform& platform,
                                const DirectQueueState& queue,
                                const DirectQueueOptions& options);
hsa_status_t SubmitDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState& queue,
                               const uint32_t* pm4,
                               size_t dword_count,
                               const DirectQueueOptions& options);
hsa_status_t ReadDirectQueueRptr(const DirectQueuePlatform& platform,
                                 const DirectQueueState& queue,
                                 uint32_t* rptr);

}  // namespace lite
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_LITE_DIRECT_QUEUE_H_
