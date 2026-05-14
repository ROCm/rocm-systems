/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef NPKIT_H_
#define NPKIT_H_

#include <string>
#include <thread>

#include <hip/hip_runtime.h>

#include "npkit/npkit_event.h"
#include "npkit/npkit_struct.h"
#include "common.h"

#define NPKIT_GET_GPU_TIMESTAMP wall_clock64
#define NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK \
  __atomic_load_n(reinterpret_cast<uint64_t*>((uint8_t *)ncclShmem.comm.cpuTimestamp + 128*blockIdx.x), __ATOMIC_RELAXED)


class NpKit {
 public:
  static const uint64_t kNumGpuEventBuffers = 1024;

  static const uint64_t kNumCpuEventBuffers = 64;

  static ncclResult_t Init(int rank);

  static ncclResult_t Dump(const std::string& dump_dir);

  static ncclResult_t Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContexts();

  static inline __device__ void CollectGpuEvent(uint8_t type, int64_t size, uint32_t rsvd, uint64_t timestamp,
                                                NpKitEventCollectContext* ctx) {
    uint64_t event_buffer_head = ctx->event_buffer_head;
    if (event_buffer_head < kMaxNumGpuEventsPerBuffer) {
      NpKitEvent& event = ctx->event_buffer[event_buffer_head];
      event.fields.type = type;
      event.fields.size = size < 0 ? 0 : size;
      event.fields.rsvd = rsvd;
      event.fields.timestamp = timestamp;
      ctx->event_buffer_head++;
    }
  }

  static void CollectCpuEvent(uint8_t type, int64_t size, uint32_t rsvd, uint64_t timestamp, int channel_id);

  static uint64_t *GetCpuTimestamp();

 private:
  static void CpuTimestampUpdateThread();

  // 64K * 512 * 16B = 512MB per GPU
  static const uint64_t kMaxNumGpuEventsPerBuffer = 1ULL << 16;

  // 64K * 2 (send/recv) * (512/32) = 2M, 2M * 32 * 16B = 1GB per CPU
  static const uint64_t kMaxNumCpuEventsPerBuffer = 1ULL << 21;

  static NpKitEvent** gpu_event_buffers_;
  static NpKitEvent** cpu_event_buffers_;

  static NpKitEventCollectContext* gpu_collect_contexts_;
  static NpKitEventCollectContext* cpu_collect_contexts_;
  static uint64_t* cpu_timestamp_;

  static uint64_t rank_;

  static std::thread* cpu_timestamp_update_thread_;
  static volatile bool cpu_timestamp_update_thread_should_stop_;
};

#endif
