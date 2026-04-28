////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_lite_direct_queue.h"

#include <algorithm>
#include <atomic>
#include <cstdio>

namespace rocr {
namespace AMD {
namespace lite {
namespace {

constexpr uint32_t kGcBase0 = 0x1260;
constexpr uint32_t kGcBase1 = 0xA000;

constexpr uint32_t regGRBM_GFX_CNTL = 0x0900;
constexpr uint32_t regCP_MQD_BASE_ADDR = 0x1fa9;
constexpr uint32_t regCP_MQD_BASE_ADDR_HI = 0x1faa;
constexpr uint32_t regCP_HQD_ACTIVE = 0x1fab;
constexpr uint32_t regCP_HQD_VMID = 0x1fac;
constexpr uint32_t regCP_HQD_PERSISTENT_STATE = 0x1fad;
constexpr uint32_t regCP_HQD_PQ_BASE = 0x1fb1;
constexpr uint32_t regCP_HQD_PQ_BASE_HI = 0x1fb2;
constexpr uint32_t regCP_HQD_PQ_RPTR = 0x1fb3;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7;
constexpr uint32_t regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8;
constexpr uint32_t regCP_HQD_PQ_CONTROL = 0x1fba;
constexpr uint32_t regCP_HQD_DEQUEUE_REQUEST = 0x1fc1;
constexpr uint32_t regCP_MQD_CONTROL = 0x1fcb;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR = 0x1fce;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf;
constexpr uint32_t regCP_HQD_EOP_CONTROL = 0x1fd0;
constexpr uint32_t regCP_HQD_PQ_WPTR_LO = 0x1fdf;
constexpr uint32_t regCP_HQD_PQ_WPTR_HI = 0x1fe0;
constexpr uint32_t regCP_HQD_DEQUEUE_STATUS = 0x1fe8;

constexpr uint32_t kCpHqdPersistentStateDefault = 0x0be05501;

constexpr uint64_t kDirectComputeBaseOffset = 0x1900000;
constexpr uint64_t kDirectComputeStride = 0x40000;
constexpr uint64_t kDirectComputeMqdRelativeOffset = 0x00000;
constexpr uint64_t kDirectComputeRingRelativeOffset = 0x02000;
constexpr uint64_t kDirectComputeEopRelativeOffset = 0x10000;
constexpr uint64_t kDirectComputeRptrRelativeOffset = 0x20000;
constexpr uint64_t kDirectComputeWptrRelativeOffset = 0x21000;

const char* TracePrefix(const DirectQueueOptions& options) {
  return options.trace_prefix ? options.trace_prefix : "ROCR lite direct queue";
}

hsa_status_t SelectHqd(const DirectQueuePlatform& platform, uint32_t me,
                       uint32_t pipe, uint32_t queue) {
  return platform.WriteMmio32(kGcBase1, regGRBM_GFX_CNTL,
                              ((pipe & 0x3u) << 0) | ((me & 0x3u) << 2) |
                                  ((0u & 0xFu) << 4) | ((queue & 0x7u) << 8));
}

hsa_status_t DeselectHqd(const DirectQueuePlatform& platform) {
  return platform.WriteMmio32(kGcBase1, regGRBM_GFX_CNTL, 0);
}

hsa_status_t WaitForDirectHqdIdle(const DirectQueuePlatform& platform,
                                  uint32_t pipe, uint32_t queue,
                                  const char* phase,
                                  const DirectQueueOptions& options) {
  constexpr uint32_t kStepUs = 1000;
  const uint32_t timeout_us = options.dequeue_settle_us != 0 ? options.dequeue_settle_us : 100000;
  const uint32_t max_samples = std::max<uint32_t>(1, timeout_us / kStepUs);
  uint32_t active = 0;
  uint32_t pq_control = 0;
  uint32_t doorbell_control = 0;
  uint32_t dequeue_status = 0;
  for (uint32_t i = 0; i < max_samples; ++i) {
    hsa_status_t status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_DEQUEUE_STATUS, &dequeue_status);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (active == 0 && (doorbell_control & 0x40000000u) == 0) {
      if (options.trace && i > 0) {
        std::fprintf(stderr,
                     "%s hqd idle phase=%s pipe=%u hqd=%u samples=%u "
                     "active=0x%x pq_control=0x%x doorbell_control=0x%x "
                     "dequeue_status=0x%x\n",
                     TracePrefix(options), phase ? phase : "unknown", pipe, queue, i + 1,
                     active, pq_control, doorbell_control, dequeue_status);
      }
      return HSA_STATUS_SUCCESS;
    }
    platform.SleepUs(kStepUs);
  }
  if (options.trace) {
    std::fprintf(stderr,
                 "%s hqd idle timeout phase=%s pipe=%u hqd=%u active=0x%x "
                 "pq_control=0x%x doorbell_control=0x%x dequeue_status=0x%x "
                 "timeout_us=%u\n",
                 TracePrefix(options), phase ? phase : "unknown", pipe, queue, active,
                 pq_control, doorbell_control, dequeue_status, timeout_us);
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t ReclaimActiveHqd(const DirectQueuePlatform& platform,
                              const DirectQueueState& queue,
                              const DirectQueueOptions& options,
                              const char* phase) {
  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  uint32_t active = 0;
  hsa_status_t status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (active == 0) return HSA_STATUS_SUCCESS;

  if (options.use_firmware_dequeue) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s reclaim active HQD with dequeue index=%u pipe=%u "
                   "hqd=%u active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, active);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 1);
    uint32_t now_active = active;
    for (uint32_t i = 0; i < 1000; ++i) {
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &now_active);
      if (now_active == 0) break;
      platform.SleepUs(1000);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    status = WaitForDirectHqdIdle(platform, pipe, hqd_queue, phase, options);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (options.trace) {
      std::fprintf(stderr,
                   "%s dequeue reclaim complete index=%u pipe=%u hqd=%u "
                   "active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, now_active);
    }
  } else {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s reclaim active HQD without dequeue index=%u pipe=%u "
                   "hqd=%u active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, active);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0);
    uint32_t now_active = active;
    for (uint32_t i = 0; i < 1000; ++i) {
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &now_active);
      if (now_active == 0) break;
      platform.SleepUs(1000);
    }
    if (options.trace) {
      std::fprintf(stderr,
                   "%s direct-disable reclaim complete index=%u pipe=%u hqd=%u "
                   "active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, now_active);
    }
  }
  return HSA_STATUS_SUCCESS;
}

}  // namespace

uint32_t DirectQueuePipe(uint32_t queue_index) { return queue_index / 4; }

uint32_t DirectQueueHqd(uint32_t queue_index) { return queue_index % 4; }

uint32_t DirectQueueDoorbell(uint32_t queue_index) {
  return kDirectComputeDoorbellBase + queue_index * kDirectComputeDoorbellStride;
}

DirectQueueLayout BuildDirectQueueLayout(uint64_t framebuffer_base,
                                         uint32_t queue_index) {
  DirectQueueLayout layout;
  layout.base_offset = kDirectComputeBaseOffset + queue_index * kDirectComputeStride;
  layout.mqd_offset = layout.base_offset + kDirectComputeMqdRelativeOffset;
  layout.ring_offset = layout.base_offset + kDirectComputeRingRelativeOffset;
  layout.eop_offset = layout.base_offset + kDirectComputeEopRelativeOffset;
  layout.rptr_offset = layout.base_offset + kDirectComputeRptrRelativeOffset;
  layout.wptr_offset = layout.base_offset + kDirectComputeWptrRelativeOffset;
  layout.mqd_gpu = framebuffer_base + layout.mqd_offset;
  layout.ring_gpu = framebuffer_base + layout.ring_offset;
  layout.eop_gpu = framebuffer_base + layout.eop_offset;
  layout.rptr_gpu = framebuffer_base + layout.rptr_offset;
  layout.wptr_gpu = framebuffer_base + layout.wptr_offset;
  return layout;
}

DirectQueueMqd BuildPm4DirectQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index) {
  DirectQueueMqd mqd{};
  mqd[0] = 0xC0310800;
  mqd[1] = 1;
  constexpr uint32_t kStaticThreadMgmtDwords[] = {0x17u, 0x18u, 0x1Au, 0x1Bu};
  for (uint32_t dw : kStaticThreadMgmtDwords) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x2C] = 7;

  const uint64_t eop_base_shifted = layout.eop_gpu >> 8;
  mqd[0xA5] = static_cast<uint32_t>(eop_base_shifted);
  mqd[0xA6] = static_cast<uint32_t>(eop_base_shifted >> 32);
  mqd[0xA7] = 9;  // bit_length(4 KiB / 4) - 2.

  mqd[0x80] = static_cast<uint32_t>(layout.mqd_gpu) & 0xFFFFFFFCu;
  mqd[0x81] = static_cast<uint32_t>(layout.mqd_gpu >> 32);
  mqd[0x82] = 1;
  mqd[0x84] = (kCpHqdPersistentStateDefault & ~(0x3FFu << 8)) | (0x55u << 8);

  const uint64_t pq_base_shifted = layout.ring_gpu >> 8;
  mqd[0x88] = static_cast<uint32_t>(pq_base_shifted);
  mqd[0x89] = static_cast<uint32_t>(pq_base_shifted >> 32);
  mqd[0x8B] = static_cast<uint32_t>(layout.rptr_gpu) & 0xFFFFFFFCu;
  mqd[0x8C] = static_cast<uint32_t>(layout.rptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8D] = static_cast<uint32_t>(layout.wptr_gpu) & 0xFFFFFFF8u;
  mqd[0x8E] = static_cast<uint32_t>(layout.wptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8F] = ((doorbell_index & 0x03FFFFFFu) << 2) | (1u << 30);

  mqd[0x91] = 9u | (5u << 8) | (1u << 27) | (1u << 28) | (1u << 30) |
              (1u << 31) | 0x300000u | 0x8000u;
  mqd[0x95] = 0x00300000;
  mqd[0xA2] = 0x100;
  mqd[0xB8] = 1u << 15;
  return mqd;
}

hsa_status_t CreateDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState* queue,
                               uint32_t queue_index,
                               uint64_t framebuffer_base,
                               const DirectQueueOptions& options) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t status = platform.EnsureDoorbellAperture();
  if (status != HSA_STATUS_SUCCESS) return status;

  *queue = {};
  queue->queue_index = queue_index;
  queue->queue_id = queue_index + 1;
  queue->doorbell_index = DirectQueueDoorbell(queue_index);
  queue->ring_size_bytes = kDirectComputeRingSize;

  const uint32_t pipe = DirectQueuePipe(queue->queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue->queue_index);
  status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) {
    *queue = {};
    return status;
  }

  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    *queue = {};
    return status;
  }
  if (active != 0 && !options.force_reclaim) {
    DeselectHqd(platform);
    *queue = {};
    return HSA_STATUS_ERROR;
  }
  if (active != 0) {
    status = ReclaimActiveHqd(platform, *queue, options, "activate-reclaim");
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      *queue = {};
      return status;
    }
  }

  const DirectQueueLayout layout = BuildDirectQueueLayout(framebuffer_base, queue->queue_index);
  const DirectQueueMqd mqd = BuildPm4DirectQueueMqd(layout, queue->doorbell_index);

  status = platform.ZeroGpuMemory(layout.mqd_offset, kMqdSize);
  if (status == HSA_STATUS_SUCCESS) status = platform.ZeroGpuMemory(layout.ring_offset, kDirectComputeRingSize);
  if (status == HSA_STATUS_SUCCESS) status = platform.ZeroGpuMemory(layout.eop_offset, kDirectComputeEopSize);
  if (status == HSA_STATUS_SUCCESS) status = platform.ZeroGpuMemory(layout.rptr_offset, 0x20);
  if (status == HSA_STATUS_SUCCESS) status = platform.ZeroGpuMemory(layout.wptr_offset, 0x20);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  for (size_t i = 0; i < mqd.size(); ++i) {
    status = platform.WriteGpuMemory32(layout.mqd_offset + i * 4, mqd[i]);
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      *queue = {};
      return status;
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);

  auto* ring_cpu = static_cast<volatile uint32_t*>(
      platform.GpuMemoryCpuPointer(layout.ring_offset));
  auto* rptr_cpu = static_cast<volatile uint64_t*>(
      platform.GpuMemoryCpuPointer(layout.rptr_offset));
  auto* wptr_cpu = static_cast<volatile uint64_t*>(
      platform.GpuMemoryCpuPointer(layout.wptr_offset));
  volatile uint64_t* doorbell_cpu = platform.DoorbellCpuPointer(queue->doorbell_index);
  if (ring_cpu == nullptr || rptr_cpu == nullptr || wptr_cpu == nullptr ||
      doorbell_cpu == nullptr) {
    DeselectHqd(platform);
    *queue = {};
    return HSA_STATUS_ERROR;
  }

  platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0);
  uint32_t vmid = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &vmid);
  platform.WriteMmio32(kGcBase0, regCP_HQD_VMID, vmid & ~0xFu);
  uint32_t doorbell_ctl = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                       doorbell_ctl & ~0x40000000u);
  platform.WriteMmio32(kGcBase0, regCP_MQD_BASE_ADDR, mqd[0x80]);
  platform.WriteMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]);
  platform.WriteMmio32(kGcBase0, regCP_MQD_CONTROL, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_EOP_CONTROL, mqd[0xA7]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_BASE, mqd[0x88]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, mqd[0x89]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, mqd[0x91]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]);
  if (options.trace_verbose) {
    uint32_t mqd_base = 0;
    uint32_t mqd_base_hi = 0;
    uint32_t pq_base = 0;
    uint32_t pq_base_hi = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    uint32_t persistent = 0;
    uint32_t selected_vmid = 0;
    uint32_t active_before = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active_before);
    platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR, &mqd_base);
    platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, &persistent);
    platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &selected_vmid);
    std::fprintf(stderr,
                 "%s pre-active readback qid=%u active=0x%x "
                 "mqd=0x%08x:%08x pq=0x%08x:%08x pq_control=0x%08x "
                 "doorbell_control=0x%08x persistent=0x%08x vmid=0x%08x\n",
                 TracePrefix(options), queue->queue_id, active_before, mqd_base_hi,
                 mqd_base, pq_base_hi, pq_base, pq_control, doorbell_control,
                 persistent, selected_vmid);
  }
  platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 1);
  if (options.trace_verbose) {
    uint32_t active_immediate = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active_immediate);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    std::fprintf(stderr,
                 "%s post-active-write readback qid=%u active=0x%x "
                 "pq_control=0x%08x doorbell_control=0x%08x\n",
                 TracePrefix(options), queue->queue_id, active_immediate, pq_control,
                 doorbell_control);
  }

  platform.SleepUs(options.activate_sleep_us);
  uint32_t post_active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &post_active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    *queue = {};
    return status;
  }
  if (post_active == 0) {
    if (options.trace) {
      uint32_t mqd_base = 0;
      uint32_t mqd_base_hi = 0;
      uint32_t eop_base = 0;
      uint32_t eop_base_hi = 0;
      uint32_t eop_control = 0;
      uint32_t pq_base = 0;
      uint32_t pq_base_hi = 0;
      uint32_t pq_control = 0;
      uint32_t doorbell_control = 0;
      uint32_t rptr_report = 0;
      uint32_t rptr_report_hi = 0;
      uint32_t wptr_poll = 0;
      uint32_t wptr_poll_hi = 0;
      uint32_t persistent = 0;
      uint32_t selected_vmid = 0;
      uint32_t rptr = 0;
      uint32_t wptr = 0;
      uint32_t wptr_hi = 0;
      platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR, &mqd_base);
      platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR, &eop_base);
      platform.ReadMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, &eop_base_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_EOP_CONTROL, &eop_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, &rptr_report);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, &rptr_report_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, &wptr_poll);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, &wptr_poll_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, &persistent);
      platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &selected_vmid);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
      std::fprintf(stderr,
                   "%s activate failed qid=%u index=%u me=1 pipe=%u hqd=%u "
                   "doorbell=0x%x base_off=0x%llx ring=0x%llx active=0x0\n",
                   TracePrefix(options), queue->queue_id, queue->queue_index, pipe,
                   hqd_queue, queue->doorbell_index,
                   static_cast<unsigned long long>(layout.base_offset),
                   static_cast<unsigned long long>(layout.ring_gpu));
      std::fprintf(stderr,
                   "%s failed readback mqd=0x%08x:%08x eop=0x%08x:%08x "
                   "eop_ctl=0x%08x pq=0x%08x:%08x pq_ctl=0x%08x "
                   "doorbell_ctl=0x%08x rptr_report=0x%08x:%08x "
                   "wptr_poll=0x%08x:%08x persistent=0x%08x vmid=0x%08x "
                   "rptr=0x%x wptr=0x%08x:%08x\n",
                   TracePrefix(options), mqd_base_hi, mqd_base, eop_base_hi, eop_base,
                   eop_control, pq_base_hi, pq_base, pq_control, doorbell_control,
                   rptr_report_hi, rptr_report, wptr_poll_hi, wptr_poll, persistent,
                   selected_vmid, rptr, wptr_hi, wptr);
    }
    DeselectHqd(platform);
    *queue = {};
    return HSA_STATUS_ERROR;
  }

  if (options.trace) {
    uint32_t pq_base = 0;
    uint32_t pq_base_hi = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    uint32_t rptr = 0;
    uint32_t wptr = 0;
    uint32_t wptr_hi = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
    std::fprintf(stderr,
                 "%s activate qid=%u index=%u me=1 pipe=%u hqd=%u "
                 "doorbell=0x%x base_off=0x%llx ring=0x%llx active=0x%x "
                 "pq_base=0x%08x:%08x pq_control=0x%08x doorbell_control=0x%08x "
                 "rptr=0x%x wptr=0x%08x:%08x\n",
                 TracePrefix(options), queue->queue_id, queue->queue_index, pipe,
                 hqd_queue, queue->doorbell_index,
                 static_cast<unsigned long long>(layout.base_offset),
                 static_cast<unsigned long long>(layout.ring_gpu), post_active, pq_base_hi,
                 pq_base, pq_control, doorbell_control, rptr, wptr_hi, wptr);
  }
  DeselectHqd(platform);

  queue->ring_gpu = layout.ring_gpu;
  queue->wptr = 0;
  queue->ring_cpu = ring_cpu;
  queue->rptr_cpu = rptr_cpu;
  queue->wptr_cpu = wptr_cpu;
  queue->doorbell_cpu = doorbell_cpu;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t DestroyDirectQueue(const DirectQueuePlatform& platform,
                                const DirectQueueState& queue,
                                const DirectQueueOptions& options) {
  if (queue.queue_id == 0) return HSA_STATUS_SUCCESS;
  if (options.skip_destroy) {
    if (options.trace) {
      std::fprintf(stderr, "%s destroy skipped qid=%u index=%u\n",
                   TracePrefix(options), queue.queue_id, queue.queue_index);
    }
    return HSA_STATUS_SUCCESS;
  }

  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  hsa_status_t status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }

  if (options.use_firmware_dequeue) {
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 1);
    for (uint32_t i = 0; i < 100; ++i) {
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
      if (active == 0) break;
      platform.SleepUs(1000);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    status = WaitForDirectHqdIdle(platform, pipe, hqd_queue, "destroy", options);
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      return status;
    }
  } else {
    platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR, 0);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0);
    uint32_t doorbell_ctl = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                         doorbell_ctl & ~0x40000000u);
  }
  if (options.trace) {
    uint32_t post_active = 0;
    uint32_t rptr = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &post_active);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    std::fprintf(stderr,
                 "%s destroy qid=%u index=%u pipe=%u hqd=%u mode=%s "
                 "pre_active=0x%x post_active=0x%x rptr=0x%x\n",
                 TracePrefix(options), queue.queue_id, queue.queue_index, pipe,
                 hqd_queue, options.use_firmware_dequeue ? "dequeue" : "disable",
                 active, post_active, rptr);
  }
  DeselectHqd(platform);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t SubmitDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState& queue,
                               const uint32_t* pm4,
                               size_t dword_count,
                               const DirectQueueOptions& options) {
  if (queue.queue_id == 0 || queue.ring_cpu == nullptr || queue.wptr_cpu == nullptr ||
      queue.doorbell_cpu == nullptr || pm4 == nullptr || dword_count == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint64_t ring_dw = queue.ring_size_bytes / sizeof(uint32_t);
  const uint64_t wptr = queue.wptr;
  const uint64_t start = wptr % ring_dw;
  for (size_t i = 0; i < dword_count; ++i) {
    queue.ring_cpu[(start + i) % ring_dw] = pm4[i];
  }
  std::atomic_thread_fence(std::memory_order_release);
  const uint64_t new_wptr = wptr + dword_count;
  *queue.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);

  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  if (options.trace) {
    const size_t sample = std::min<size_t>(dword_count, 8);
    std::fprintf(stderr,
                 "%s submit qid=%u index=%u pipe=%u hqd=%u doorbell=0x%x "
                 "wptr=%llu new_wptr=%llu dwords=%zu first_pm4=",
                 TracePrefix(options), queue.queue_id, queue.queue_index, pipe,
                 hqd_queue, queue.doorbell_index,
                 static_cast<unsigned long long>(wptr),
                 static_cast<unsigned long long>(new_wptr), dword_count);
    for (size_t i = 0; i < sample; ++i) {
      std::fprintf(stderr, "%s0x%08x", i == 0 ? "" : ",", pm4[i]);
    }
    std::fprintf(stderr, "\n");
  }

  hsa_status_t status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  uint32_t selected_active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &selected_active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }
  if (selected_active == 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s submit rejected inactive HQD qid=%u index=%u pipe=%u "
                   "hqd=%u doorbell=0x%x\n",
                   TracePrefix(options), queue.queue_id, queue.queue_index, pipe,
                   hqd_queue, queue.doorbell_index);
    }
    DeselectHqd(platform);
    return HSA_STATUS_ERROR;
  }
  status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO,
                                static_cast<uint32_t>(new_wptr));
  if (status == HSA_STATUS_SUCCESS) {
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI,
                                  static_cast<uint32_t>(new_wptr >> 32));
  }
  if (options.trace) {
    uint32_t active = 0;
    uint32_t rptr = 0;
    uint32_t mmio_wptr = 0;
    uint32_t mmio_wptr_hi = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &mmio_wptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &mmio_wptr_hi);
    std::fprintf(stderr,
                 "%s after mmio-wptr qid=%u active=0x%x rptr=0x%x "
                 "wptr=0x%08x:%08x status=%u\n",
                 TracePrefix(options), queue.queue_id, active, rptr, mmio_wptr_hi,
                 mmio_wptr, status);
  }
  DeselectHqd(platform);
  if (status != HSA_STATUS_SUCCESS) return status;
  *queue.doorbell_cpu = new_wptr;
  queue.wptr = new_wptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ReadDirectQueueRptr(const DirectQueuePlatform& platform,
                                 const DirectQueueState& queue,
                                 uint32_t* rptr) {
  if (queue.queue_id == 0 || rptr == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  hsa_status_t status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, rptr);
  DeselectHqd(platform);
  return status;
}

}  // namespace lite
}  // namespace AMD
}  // namespace rocr
