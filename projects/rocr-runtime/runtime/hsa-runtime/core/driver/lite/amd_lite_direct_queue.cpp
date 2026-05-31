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
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <unordered_map>

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
constexpr uint32_t regCP_HQD_GFX_CONTROL = 0x1e9f;
constexpr uint32_t regCP_HQD_DEQUEUE_REQUEST = 0x1fc1;
// Pair RESET_WAVES dequeue with an SPI compute-queue reset (matches tinygrad's
// gfx12 _dequeue_hqds). GC BASE_IDX=0.
constexpr uint32_t regSPI_COMPUTE_QUEUE_RESET = 0x1f73;
constexpr uint32_t regCP_MQD_CONTROL = 0x1fcb;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR = 0x1fce;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf;
constexpr uint32_t regCP_HQD_EOP_CONTROL = 0x1fd0;
constexpr uint32_t regCP_HQD_PQ_WPTR_LO = 0x1fdf;
constexpr uint32_t regCP_HQD_PQ_WPTR_HI = 0x1fe0;
constexpr uint32_t regCP_HQD_DEQUEUE_STATUS = 0x1fe8;
constexpr uint32_t regCP_UNMAPPED_DOORBELL = 0x0880;
constexpr uint32_t regSCRATCH_REG0 = 0x2040;
constexpr uint32_t regCP_MES_CNTL = 0x2807;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL1 = 0x283c;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL2 = 0x283d;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL3 = 0x283e;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL4 = 0x283f;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL5 = 0x2840;
constexpr uint32_t regCP_MES_GP3_LO = 0x2849;
constexpr uint32_t regCP_PQ_WPTR_POLL_CNTL = 0x1e23;
constexpr uint32_t regCP_PQ_STATUS = 0x1e58;
constexpr uint32_t regRLC_CP_SCHEDULERS = 0x098a;

constexpr uint32_t kCpHqdPersistentStateDefault = 0x0be05501;
constexpr uint32_t kCpMqdControlDefault = 0x00000100;
constexpr uint32_t kCpHqdPqControlDefault = 0x00308509;
constexpr uint32_t kCpHqdPqControlPm4 =
    (kCpHqdPqControlDefault & ~0x00003f00u) | (9u << 8) | (1u << 28) |
    (1u << 30) | (1u << 31);
constexpr uint32_t kCpHqdPqControlMes = 0xd8300909;
constexpr uint32_t kCpHqdDequeueDrainPipe = 0x1;
constexpr uint32_t kCpHqdDequeueResetWaves = 0x2;
constexpr uint32_t kCpMesCntlPipe0Active = 1u << 26;
constexpr uint32_t kCpMesCntlPipe1Active = 1u << 27;
constexpr uint32_t kCpUnmappedDoorbellEnable = 1u << 0;
constexpr uint32_t kCpUnmappedDoorbellProcLsbMask = 0x00001f00u;
constexpr uint32_t kCpUnmappedDoorbellProcLsbShift = 8;
constexpr uint32_t kCpMesDoorbellOffsetMask = 0x0ffffffcu;
constexpr uint32_t kCpMesDoorbellEnable = 1u << 30;
constexpr uint32_t kCpMesDoorbellHit = 1u << 31;
constexpr uint32_t kCpHqdGfxControlDbUpdatedMsgEn = 1u << 15;

constexpr uint64_t kDirectComputeBaseOffset = 0x1900000;
constexpr uint64_t kDirectComputeStride = 0x40000;
constexpr uint64_t kDirectComputeMqdRelativeOffset = 0x00000;
constexpr uint64_t kDirectComputeRingRelativeOffset = 0x02000;
constexpr uint64_t kDirectComputeEopRelativeOffset = 0x10000;
constexpr uint64_t kDirectComputeRptrRelativeOffset = 0x20000;
constexpr uint64_t kDirectComputeWptrRelativeOffset = 0x21000;
constexpr uint32_t kMesRingSize = 0x8000;
constexpr uint64_t kMesSchedulerBaseOffset = 0x1800000;
constexpr uint64_t kMesKiqBaseOffset = 0x1840000;
[[maybe_unused]] constexpr uint64_t kLegacyKiqBaseOffset = 0x1880000;
constexpr uint64_t kMesSchedulerContextRelativeOffset = 0x22000;
constexpr uint64_t kMesQueryFenceRelativeOffset = 0x23000;
constexpr uint64_t kMesApiFenceRelativeOffset = 0x24000;
constexpr uint64_t kMesCleanerFenceRelativeOffset = 0x24020;

constexpr uint32_t kMesApiFrameDwords = 64;
constexpr uint32_t kMesApiTypeScheduler = 1;
constexpr uint32_t kMesOpcodeSetHwResources = 0;
constexpr uint32_t kMesOpcodeAddQueue = 2;
constexpr uint32_t kMesOpcodeRemoveQueue = 3;
constexpr uint32_t kMesOpcodeQuerySchedulerStatus = 11;
constexpr uint32_t kMesOpcodeSetHwResources1 = 19;
constexpr uint32_t kMesQueueTypeCompute = 1;
constexpr uint32_t kMesQueueTypeScheduler = 3;
constexpr uint32_t kMesSchedulerDoorbell = 0x00b << 1;
constexpr uint32_t kMesKiqDoorbell = 0x00c << 1;
constexpr uint32_t kMesKiqMe = 3;
constexpr uint32_t kMesKiqPipe = 1;
constexpr uint32_t kMesKiqHqd = 0;
[[maybe_unused]] constexpr uint32_t kLegacyKiqDoorbell = 0x000 << 1;
[[maybe_unused]] constexpr uint32_t kLegacyKiqMe = 1;
[[maybe_unused]] constexpr uint32_t kLegacyKiqPipe = 0;
[[maybe_unused]] constexpr uint32_t kLegacyKiqHqd = 0;
constexpr uint32_t kMesApiStatusSetHwResourcesDw = 50;
constexpr uint32_t kMesApiStatusSetHwResources1Dw = 2;
constexpr uint32_t kMesApiStatusAddQueueDw = 38;
constexpr uint32_t kMesApiStatusRemoveQueueDw = 6;
constexpr uint32_t kMesSetHwResourcesFlags = 0x00080447;
constexpr uint32_t kMesAddQueueMapLegacyKq = 1u << 13;
constexpr uint32_t kMesRemoveQueueUnmapLegacy = 1u << 3;
constexpr uint32_t kMesAggregatedDoorbellBase = 0x80;
constexpr uint32_t kPacketType3 = 3;
[[maybe_unused]] constexpr uint32_t kPacket3Nop = 0x10;
constexpr uint32_t kPacket3WriteData = 0x37;
constexpr uint32_t kPacket3MapQueues = 0xa2;

const char* TracePrefix(const DirectQueueOptions& options) {
  return options.trace_prefix ? options.trace_prefix : "ROCR lite direct queue";
}

uint32_t MesHeader(uint32_t opcode) {
  return kMesApiTypeScheduler | (opcode << 4) | (kMesApiFrameDwords << 12);
}

uint32_t Packet3(uint32_t opcode, uint32_t count) {
  return (kPacketType3 << 30) | ((opcode & 0xffu) << 8) |
         ((count & 0x3fffu) << 16);
}

void PutU64(uint32_t* frame, uint32_t dword_offset, uint64_t value) {
  frame[dword_offset] = static_cast<uint32_t>(value);
  frame[dword_offset + 1] = static_cast<uint32_t>(value >> 32);
}

DirectQueueLayout BuildQueueLayoutAt(uint64_t framebuffer_base,
                                     uint64_t base_offset) {
  DirectQueueLayout layout;
  layout.base_offset = base_offset;
  layout.base_gpu = framebuffer_base + layout.base_offset;
  layout.mqd_offset = layout.base_offset + kDirectComputeMqdRelativeOffset;
  layout.ring_offset = layout.base_offset + kDirectComputeRingRelativeOffset;
  layout.eop_offset = layout.base_offset + kDirectComputeEopRelativeOffset;
  layout.rptr_offset = layout.base_offset + kDirectComputeRptrRelativeOffset;
  layout.wptr_offset = layout.base_offset + kDirectComputeWptrRelativeOffset;
  layout.mqd_gpu = layout.base_gpu + kDirectComputeMqdRelativeOffset;
  layout.ring_gpu = layout.base_gpu + kDirectComputeRingRelativeOffset;
  layout.eop_gpu = layout.base_gpu + kDirectComputeEopRelativeOffset;
  layout.rptr_gpu = layout.base_gpu + kDirectComputeRptrRelativeOffset;
  layout.wptr_gpu = layout.base_gpu + kDirectComputeWptrRelativeOffset;
  return layout;
}

DirectQueueLayout BuildQueueLayoutFromMemory(const DirectQueueMemory& memory) {
  DirectQueueLayout layout;
  layout.base_offset = 0;
  layout.base_gpu = memory.gpu_addr;
  layout.cpu_base = memory.cpu;
  layout.cpu_size = memory.size;
  layout.mqd_offset = kDirectComputeMqdRelativeOffset;
  layout.ring_offset = kDirectComputeRingRelativeOffset;
  layout.eop_offset = kDirectComputeEopRelativeOffset;
  layout.rptr_offset = kDirectComputeRptrRelativeOffset;
  layout.wptr_offset = kDirectComputeWptrRelativeOffset;
  layout.mqd_gpu = layout.base_gpu + kDirectComputeMqdRelativeOffset;
  layout.ring_gpu = layout.base_gpu + kDirectComputeRingRelativeOffset;
  layout.eop_gpu = layout.base_gpu + kDirectComputeEopRelativeOffset;
  layout.rptr_gpu = layout.base_gpu + kDirectComputeRptrRelativeOffset;
  layout.wptr_gpu = layout.base_gpu + kDirectComputeWptrRelativeOffset;
  return layout;
}

uint64_t LayoutRelativeOffset(const DirectQueueLayout& layout,
                              uint64_t offset) {
  return offset - layout.base_offset;
}

void* LayoutCpuPointer(const DirectQueuePlatform& platform,
                       const DirectQueueLayout& layout,
                       uint64_t offset,
                       uint64_t size = 1) {
  if (layout.cpu_base != nullptr) {
    if (offset < layout.base_offset) return nullptr;
    const uint64_t relative = LayoutRelativeOffset(layout, offset);
    if (relative > layout.cpu_size || size > layout.cpu_size - relative) {
      return nullptr;
    }
    return static_cast<char*>(layout.cpu_base) + relative;
  }
  return platform.GpuMemoryCpuPointer(offset);
}

hsa_status_t ZeroLayoutMemory(const DirectQueuePlatform& platform,
                              const DirectQueueLayout& layout,
                              uint64_t offset,
                              uint64_t size) {
  if (layout.cpu_base != nullptr) {
    void* cpu = LayoutCpuPointer(platform, layout, offset, size);
    if (cpu == nullptr) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    std::memset(cpu, 0, static_cast<size_t>(size));
    std::atomic_thread_fence(std::memory_order_release);
    return HSA_STATUS_SUCCESS;
  }
  return platform.ZeroGpuMemory(offset, size);
}

hsa_status_t WriteLayoutMemory32(const DirectQueuePlatform& platform,
                                 const DirectQueueLayout& layout,
                                 uint64_t offset,
                                 uint32_t value) {
  if (layout.cpu_base != nullptr) {
    void* cpu = LayoutCpuPointer(platform, layout, offset, sizeof(uint32_t));
    if (cpu == nullptr) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    *reinterpret_cast<volatile uint32_t*>(cpu) = value;
    return HSA_STATUS_SUCCESS;
  }
  return platform.WriteGpuMemory32(offset, value);
}

hsa_status_t PrepareQueueLayout(const DirectQueuePlatform& platform,
                                uint64_t framebuffer_base,
                                uint64_t base_offset,
                                DirectQueueLayout* layout,
                                DirectQueueMemory* memory) {
  if (layout == nullptr || memory == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *memory = {};
  const char* force_vram = std::getenv("ROCR_AMDGPU_LITE_FORCE_VRAM_QUEUE_MEMORY");
  if (platform.PreferAllocatedQueueMemory() &&
      (force_vram == nullptr || force_vram[0] == '\0' ||
       force_vram[0] == '0')) {
    hsa_status_t status =
        platform.AllocateQueueMemory(kDirectComputeStride, memory);
    if (status != HSA_STATUS_SUCCESS) return status;
    *layout = BuildQueueLayoutFromMemory(*memory);
    return HSA_STATUS_SUCCESS;
  }
  *layout = BuildQueueLayoutAt(framebuffer_base, base_offset);
  return HSA_STATUS_SUCCESS;
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
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST,
                         kCpHqdDequeueResetWaves);
    platform.WriteMmio32(kGcBase0, regSPI_COMPUTE_QUEUE_RESET, 1);
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

hsa_status_t ResetSelectedHqdForProgramming(const DirectQueuePlatform& platform,
                                            uint32_t pipe,
                                            uint32_t hqd_queue,
                                            const DirectQueueOptions& options,
                                            const char* label) {
  constexpr uint32_t kStepUs = 1;
  const uint32_t timeout_us =
      options.dequeue_settle_us != 0 ? options.dequeue_settle_us : 100000;
  uint32_t active = 0;
  hsa_status_t status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) return status;

  if ((active & 1u) != 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s %s reset active HQD before program pipe=%u hqd=%u "
                   "active=0x%x\n",
                   TracePrefix(options), label ? label : "HQD", pipe,
                   hqd_queue, active);
    }
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST,
                                  kCpHqdDequeueDrainPipe);
    if (status != HSA_STATUS_SUCCESS) return status;

    for (uint32_t elapsed = 0; elapsed < timeout_us; elapsed += kStepUs) {
      status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
      if (status != HSA_STATUS_SUCCESS) return status;
      if ((active & 1u) == 0) break;
      platform.SleepUs(kStepUs);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    if ((active & 1u) != 0) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s %s reset dequeue timeout; clearing selected HQD "
                     "directly pipe=%u hqd=%u active=0x%x timeout_us=%u\n",
                     TracePrefix(options), label ? label : "HQD", pipe,
                     hqd_queue, active, timeout_us);
      }
    }
  } else {
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    if (status != HSA_STATUS_SUCCESS) return status;
  }

  uint32_t doorbell_ctl = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                               &doorbell_ctl);
  if (status != HSA_STATUS_SUCCESS) return status;
  doorbell_ctl &= ~kCpMesDoorbellEnable;
  doorbell_ctl |= kCpMesDoorbellHit;
  status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                                doorbell_ctl);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, 0);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR, 0) !=
          HSA_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }

  if (options.trace_verbose) {
    uint32_t rptr = 0;
    uint32_t wptr = 0;
    uint32_t wptr_hi = 0;
    uint32_t post_doorbell_ctl = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                        &post_doorbell_ctl);
    std::fprintf(stderr,
                 "%s %s reset complete pipe=%u hqd=%u rptr=0x%x "
                 "wptr=0x%08x:%08x doorbell_ctl=0x%08x\n",
                 TracePrefix(options), label ? label : "HQD", pipe, hqd_queue,
                 rptr, wptr_hi, wptr, post_doorbell_ctl);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ZeroQueueMemory(const DirectQueuePlatform& platform,
                             const DirectQueueLayout& layout,
                             uint32_t ring_size = kDirectComputeRingSize) {
  hsa_status_t status =
      ZeroLayoutMemory(platform, layout, layout.mqd_offset, kMqdSize);
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, layout, layout.ring_offset, ring_size);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status =
        ZeroLayoutMemory(platform, layout, layout.eop_offset, kDirectComputeEopSize);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, layout, layout.rptr_offset, 0x20);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, layout, layout.wptr_offset, 0x20);
  }
  if (status != HSA_STATUS_SUCCESS) return status;
  std::atomic_thread_fence(std::memory_order_release);
  return platform.FlushHdp();
}

hsa_status_t WriteMqdImage(const DirectQueuePlatform& platform,
                           const DirectQueueLayout& layout,
                           const DirectQueueMqd& mqd) {
  for (size_t i = 0; i < mqd.size(); ++i) {
    hsa_status_t status =
        WriteLayoutMemory32(platform, layout, layout.mqd_offset + i * 4, mqd[i]);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return platform.FlushHdp();
}

[[maybe_unused]] hsa_status_t ProgramHqdRegisters(
    const DirectQueuePlatform& platform,
    uint32_t me,
    uint32_t pipe,
    uint32_t hqd_queue,
    const DirectQueueMqd& mqd,
    const DirectQueueOptions& options,
    const char* label) {
  hsa_status_t status = SelectHqd(platform, me, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;

  auto write = [&](uint32_t base, uint32_t reg, uint32_t value) {
    status = platform.WriteMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };
  auto read = [&](uint32_t base, uint32_t reg, uint32_t* value) {
    status = platform.ReadMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };

  uint32_t vmid = 0;
  uint32_t doorbell_ctl = 0;
  if (!write(kGcBase0, regCP_HQD_ACTIVE, 0) ||
      !write(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0) ||
      !read(kGcBase0, regCP_HQD_VMID, &vmid) ||
      !write(kGcBase0, regCP_HQD_VMID, vmid & ~0xFu) ||
      !read(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
             doorbell_ctl & ~0x40000000u) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR, mqd[0x80]) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]) ||
      !write(kGcBase0, regCP_MQD_CONTROL, mqd[0xA2]) ||
      !write(kGcBase0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]) ||
      !write(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]) ||
      !write(kGcBase0, regCP_HQD_EOP_CONTROL, mqd[0xA7]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE, mqd[0x88]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE_HI, mqd[0x89]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]) ||
      !write(kGcBase0, regCP_HQD_PQ_CONTROL, mqd[0x91]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]) ||
      !write(kGcBase0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]) ||
      !write(kGcBase0, regCP_HQD_GFX_CONTROL,
             kCpHqdGfxControlDbUpdatedMsgEn) ||
      !write(kGcBase0, regCP_HQD_ACTIVE, 1)) {
    DeselectHqd(platform);
    return status;
  }
  uint32_t pq_status = 0;
  if (!read(kGcBase0, regCP_PQ_STATUS, &pq_status) ||
      !write(kGcBase0, regCP_PQ_STATUS, pq_status | (1u << 1))) {
    DeselectHqd(platform);
    return status;
  }

  platform.SleepUs(options.activate_sleep_us);
  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  DeselectHqd(platform);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (active == 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s %s HQD activation failed me=%u pipe=%u hqd=%u\n",
                   TracePrefix(options), label ? label : "MES", me, pipe,
                   hqd_queue);
    }
    return HSA_STATUS_ERROR;
  }
  if (options.trace) {
    std::fprintf(stderr, "%s %s HQD active me=%u pipe=%u hqd=%u\n",
                 TracePrefix(options), label ? label : "MES", me, pipe,
                 hqd_queue);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ProgramMesQueueRegisters(const DirectQueuePlatform& platform,
                                      uint32_t pipe,
                                      const DirectQueueMqd& mqd,
                                      const DirectQueueOptions& options,
                                      const char* label) {
  hsa_status_t status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
  if (status != HSA_STATUS_SUCCESS) return status;

  auto write = [&](uint32_t base, uint32_t reg, uint32_t value) {
    status = platform.WriteMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };
  auto read = [&](uint32_t base, uint32_t reg, uint32_t* value) {
    status = platform.ReadMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };

  status = ResetSelectedHqdForProgramming(platform, pipe, kMesKiqHqd, options,
                                          label ? label : "MES KIQ");
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }

  uint32_t vmid = 0;
  uint32_t doorbell_ctl = 0;
  if (!write(kGcBase0, regCP_HQD_ACTIVE, 0) ||
      !write(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0) ||
      !read(kGcBase0, regCP_HQD_VMID, &vmid) ||
      !write(kGcBase0, regCP_HQD_VMID, vmid & ~0xFu) ||
      !read(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
             doorbell_ctl & ~0x40000000u) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR, mqd[0x80]) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]) ||
      !write(kGcBase0, regCP_MQD_CONTROL, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE, mqd[0x88]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE_HI, mqd[0x89]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]) ||
      !write(kGcBase0, regCP_HQD_PQ_CONTROL, mqd[0x91]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]) ||
      !write(kGcBase0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]) ||
      !write(kGcBase0, regCP_HQD_ACTIVE, 1)) {
    DeselectHqd(platform);
    return status;
  }

  platform.SleepUs(options.activate_sleep_us);
  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  DeselectHqd(platform);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (active == 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s %s activation failed me=%u pipe=%u hqd=%u\n",
                   TracePrefix(options), label ? label : "MES KIQ", kMesKiqMe,
                   pipe, kMesKiqHqd);
    }
    return HSA_STATUS_ERROR;
  }
  if (options.trace) {
    std::fprintf(stderr, "%s %s active me=%u pipe=%u hqd=%u\n",
                 TracePrefix(options), label ? label : "MES KIQ", kMesKiqMe,
                 pipe, kMesKiqHqd);
  }
  if (options.trace_verbose) {
    hsa_status_t select_status =
        SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
    if (select_status == HSA_STATUS_SUCCESS) {
      uint32_t pq_control = 0;
      uint32_t rptr = 0;
      uint32_t wptr = 0;
      uint32_t wptr_hi = 0;
      uint32_t doorbell_control = 0;
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                          &doorbell_control);
      DeselectHqd(platform);
      std::fprintf(stderr,
                   "%s %s post-program pipe=%u hqd=%u pq_control=0x%08x "
                   "rptr=0x%x wptr=0x%08x:%08x doorbell_ctl=0x%08x\n",
                   TracePrefix(options), label ? label : "MES KIQ", pipe,
                   kMesKiqHqd, pq_control, rptr, wptr_hi, wptr,
                   doorbell_control);
    }
  }
  return HSA_STATUS_SUCCESS;
}

DirectQueueState MakeRingState(const DirectQueuePlatform& platform,
                               const DirectQueueLayout& layout,
                               uint32_t doorbell_index,
                               uint32_t ring_size = kDirectComputeRingSize,
                               uint32_t align_mask = 0,
                               uint32_t nop_packet = 0) {
  DirectQueueState ring{};
  ring.doorbell_index = doorbell_index;
  ring.ring_size_bytes = ring_size;
  ring.ring_align_mask = align_mask;
  ring.ring_nop = nop_packet;
  ring.ring_gpu = layout.ring_gpu;
  ring.ring_cpu = static_cast<volatile uint32_t*>(
      LayoutCpuPointer(platform, layout, layout.ring_offset,
                       ring_size));
  ring.rptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.rptr_offset,
                       sizeof(uint64_t)));
  ring.wptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.wptr_offset,
                       sizeof(uint64_t)));
  ring.doorbell_cpu = platform.DoorbellCpuPointer(doorbell_index);
  return ring;
}

hsa_status_t SubmitRingPm4(const DirectQueuePlatform& platform,
                           DirectQueueState& ring, const uint32_t* pm4,
                           size_t dword_count) {
  if (ring.ring_cpu == nullptr || ring.wptr_cpu == nullptr ||
      ring.doorbell_cpu == nullptr || pm4 == nullptr || dword_count == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint64_t ring_dw = ring.ring_size_bytes / sizeof(uint32_t);
  const uint64_t start = ring.wptr % ring_dw;
  for (size_t i = 0; i < dword_count; ++i) {
    ring.ring_cpu[(start + i) % ring_dw] = pm4[i];
  }
  size_t pad_count = 0;
  if (ring.ring_align_mask != 0) {
    const uint64_t unpadded_wptr = ring.wptr + dword_count;
    pad_count = (ring.ring_align_mask + 1 -
                 (unpadded_wptr & ring.ring_align_mask)) &
                ring.ring_align_mask;
    for (size_t i = 0; i < pad_count; ++i) {
      ring.ring_cpu[(start + dword_count + i) % ring_dw] = ring.ring_nop;
    }
  }
  std::atomic_thread_fence(std::memory_order_release);
  const uint64_t new_wptr = ring.wptr + dword_count + pad_count;
  *ring.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  hsa_status_t status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;
  *ring.doorbell_cpu = new_wptr;
  ring.wptr = new_wptr;
  return HSA_STATUS_SUCCESS;
}

struct MesSchedulerState {
  bool initialized = false;
  DirectQueueState ring;
  DirectQueueState kiq;
  DirectQueueMemory scheduler_memory;
  DirectQueueMemory kiq_memory;
  uint64_t sch_ctx_gpu = 0;
  uint64_t query_status_fence_gpu = 0;
  uint64_t api_fence_gpu = 0;
  volatile uint64_t* api_fence_cpu = nullptr;
  uint64_t cleaner_fence_gpu = 0;
  uint64_t kiq_sch_ctx_gpu = 0;
  uint64_t kiq_query_status_fence_gpu = 0;
  uint64_t kiq_api_fence_gpu = 0;
  volatile uint64_t* kiq_api_fence_cpu = nullptr;
  uint64_t kiq_cleaner_fence_gpu = 0;
  uint64_t next_fence_value = 0;
  std::array<uint32_t, 5> aggregated_doorbells{};
};

std::mutex& MesSchedulerMutex() {
  static std::mutex* m = new std::mutex();
  return *m;
}

std::unordered_map<const DirectQueuePlatform*, MesSchedulerState>& MesSchedulers() {
  static auto* states =
      new std::unordered_map<const DirectQueuePlatform*, MesSchedulerState>();
  return *states;
}

void ResetMesSchedulerState(const DirectQueuePlatform& platform,
                            MesSchedulerState& state) {
  platform.FreeQueueMemory(&state.scheduler_memory);
  platform.FreeQueueMemory(&state.kiq_memory);
  state = {};
}

uint32_t MesPipeForRing(const DirectQueueState& ring) {
  return ring.doorbell_index == kMesKiqDoorbell ? kMesKiqPipe : 0;
}

hsa_status_t SyncMesRingPointersFromHardware(const DirectQueuePlatform& platform,
                                             DirectQueueState& ring,
                                             const DirectQueueOptions& options,
                                             const char* label) {
  if (ring.wptr_cpu == nullptr || ring.rptr_cpu == nullptr) {
    return HSA_STATUS_ERROR;
  }

  const uint32_t pipe = MesPipeForRing(ring);
  hsa_status_t status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
  if (status != HSA_STATUS_SUCCESS) return status;

  uint32_t hw_wptr = 0;
  uint32_t hw_wptr_hi = 0;
  uint32_t hw_rptr = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &hw_wptr);
  if (status == HSA_STATUS_SUCCESS) {
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &hw_wptr_hi);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &hw_rptr);
  }
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }

  const uint64_t initial_wptr =
      (static_cast<uint64_t>(hw_wptr_hi) << 32) | hw_wptr;
  if (initial_wptr != ring.wptr ||
      (ring.rptr_cpu != nullptr && *ring.rptr_cpu != initial_wptr) ||
      (ring.wptr_cpu != nullptr && *ring.wptr_cpu != initial_wptr)) {
    ring.wptr = initial_wptr;
    *ring.wptr_cpu = initial_wptr;
    *ring.rptr_cpu = initial_wptr;
    std::atomic_thread_fence(std::memory_order_release);
    status = platform.FlushHdp();
    if (status == HSA_STATUS_SUCCESS) {
      status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR,
                                    static_cast<uint32_t>(initial_wptr));
    }
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      return status;
    }
  }

  uint32_t post_rptr = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &post_rptr);
  DeselectHqd(platform);

  if (options.trace_verbose) {
    std::fprintf(stderr,
                 "%s MES %s pointer sync pipe=%u doorbell=0x%x "
                 "hw_wptr=0x%08x:%08x hw_rptr=0x%x ring_wptr=%llu "
                 "post_rptr=0x%x\n",
                 TracePrefix(options), label ? label : "ring", pipe,
                 ring.doorbell_index, hw_wptr_hi, hw_wptr, hw_rptr,
                 static_cast<unsigned long long>(ring.wptr), post_rptr);
  }
  return HSA_STATUS_SUCCESS;
}

void DumpMesRingState(const DirectQueuePlatform& platform,
                      const DirectQueueState& ring,
                      const DirectQueueOptions& options,
                      const char* phase) {
  if (!options.trace_verbose) return;

  const uint32_t pipe = MesPipeForRing(ring);
  hsa_status_t status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s MES %s select failed pipe=%u status=%u\n",
                 TracePrefix(options), phase ? phase : "diagnostic", pipe,
                 status);
    return;
  }

  uint32_t mes_cntl = 0;
  uint32_t schedulers = 0;
  uint32_t version = 0;
  uint32_t active = 0;
  uint32_t vmid = 0;
  uint32_t mqd_base = 0;
  uint32_t mqd_base_hi = 0;
  uint32_t pq_base = 0;
  uint32_t pq_base_hi = 0;
  uint32_t pq_control = 0;
  uint32_t rptr = 0;
  uint32_t wptr = 0;
  uint32_t wptr_hi = 0;
  uint32_t rptr_report = 0;
  uint32_t rptr_report_hi = 0;
  uint32_t wptr_poll = 0;
  uint32_t wptr_poll_hi = 0;
  uint32_t doorbell_control = 0;
  uint32_t persistent = 0;
  uint32_t dequeue_status = 0;
  platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &mes_cntl);
  platform.ReadMmio32(kGcBase1, regRLC_CP_SCHEDULERS, &schedulers);
  platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version);
  platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &vmid);
  platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR, &mqd_base);
  platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, &rptr_report);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI,
                      &rptr_report_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, &wptr_poll);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI,
                      &wptr_poll_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                      &doorbell_control);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, &persistent);
  platform.ReadMmio32(kGcBase0, regCP_HQD_DEQUEUE_STATUS, &dequeue_status);
  DeselectHqd(platform);

  const uint64_t cpu_wptr = ring.wptr_cpu != nullptr ? *ring.wptr_cpu : 0;
  const uint64_t cpu_rptr = ring.rptr_cpu != nullptr ? *ring.rptr_cpu : 0;
  std::fprintf(stderr,
               "%s MES %s pipe=%u doorbell=0x%x ring_gpu=0x%llx "
               "wptr=%llu cpu_wptr=%llu cpu_rptr=%llu mes_cntl=0x%08x "
               "rlc_sched=0x%08x version=0x%08x active=0x%x vmid=0x%08x\n",
               TracePrefix(options), phase ? phase : "diagnostic", pipe,
               ring.doorbell_index,
               static_cast<unsigned long long>(ring.ring_gpu),
               static_cast<unsigned long long>(ring.wptr),
               static_cast<unsigned long long>(cpu_wptr),
               static_cast<unsigned long long>(cpu_rptr), mes_cntl,
               schedulers, version, active, vmid);
  std::fprintf(stderr,
               "%s MES %s regs mqd=0x%08x:%08x pq=0x%08x:%08x "
               "pq_ctl=0x%08x rptr=0x%x wptr=0x%08x:%08x "
               "rptr_report=0x%08x:%08x wptr_poll=0x%08x:%08x "
               "doorbell_ctl=0x%08x persistent=0x%08x dequeue=0x%08x\n",
               TracePrefix(options), phase ? phase : "diagnostic",
               mqd_base_hi, mqd_base, pq_base_hi, pq_base, pq_control, rptr,
               wptr_hi, wptr, rptr_report_hi, rptr_report, wptr_poll_hi,
               wptr_poll, doorbell_control, persistent, dequeue_status);

  if (ring.ring_cpu != nullptr) {
    std::fprintf(stderr, "%s MES %s ring[0..15]=", TracePrefix(options),
                 phase ? phase : "diagnostic");
    for (uint32_t i = 0; i < 16; ++i) {
      std::fprintf(stderr, "%s0x%08x", i == 0 ? "" : ",", ring.ring_cpu[i]);
    }
    std::fprintf(stderr, "\n");
  }
}

hsa_status_t SubmitMesApiFrameOnRing(
    const DirectQueuePlatform& platform,
    DirectQueueState& ring,
    volatile uint64_t* api_fence_cpu,
    uint64_t api_fence_gpu,
    uint64_t& next_fence_value,
    std::array<uint32_t, kMesApiFrameDwords>& frame,
    uint32_t api_status_dw,
    const DirectQueueOptions& options,
    const char* opcode_name) {
  if (ring.ring_cpu == nullptr || ring.wptr_cpu == nullptr ||
      ring.doorbell_cpu == nullptr || api_fence_cpu == nullptr) {
    return HSA_STATUS_ERROR;
  }
  const uint64_t fence_value = ++next_fence_value;
  api_fence_cpu[0] = 0;
  api_fence_cpu[1] = 0;
  PutU64(frame.data(), api_status_dw, api_fence_gpu);
  PutU64(frame.data(), api_status_dw + 2, fence_value);

  std::array<uint32_t, kMesApiFrameDwords> query{};
  query[0] = MesHeader(kMesOpcodeQuerySchedulerStatus);
  PutU64(query.data(), 2, api_fence_gpu + sizeof(uint64_t));
  PutU64(query.data(), 4, fence_value);

  const uint64_t ring_dw = ring.ring_size_bytes / sizeof(uint32_t);
  const uint64_t start = ring.wptr % ring_dw;
  for (uint32_t i = 0; i < kMesApiFrameDwords; ++i) {
    ring.ring_cpu[(start + i) % ring_dw] = frame[i];
  }
  for (uint32_t i = 0; i < kMesApiFrameDwords; ++i) {
    ring.ring_cpu[(start + kMesApiFrameDwords + i) % ring_dw] = query[i];
  }
  std::atomic_thread_fence(std::memory_order_release);

  const uint64_t new_wptr = ring.wptr + kMesApiFrameDwords * 2;
  *ring.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  hsa_status_t status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;
  if (options.trace_verbose) {
    std::fprintf(stderr,
                 "%s MES API %s submit start=%llu old_wptr=%llu "
                 "new_wptr=%llu api_fence=0x%llx status_dw=%u "
                 "frame0=0x%08x frame1=0x%08x frame22=0x%08x "
                 "frame24=0x%08x frame50=0x%08x query0=0x%08x\n",
                 TracePrefix(options), opcode_name ? opcode_name : "unknown",
                 static_cast<unsigned long long>(start),
                 static_cast<unsigned long long>(ring.wptr),
                 static_cast<unsigned long long>(new_wptr),
                 static_cast<unsigned long long>(api_fence_gpu),
                 api_status_dw, frame[0], frame[1], frame[22], frame[24],
                 frame[50], query[0]);
    DumpMesRingState(platform, ring, options, "pre-doorbell");
  }
  *ring.doorbell_cpu = new_wptr;
  ring.wptr = new_wptr;
  if (options.trace_verbose) {
    platform.SleepUs(100);
    DumpMesRingState(platform, ring, options, "post-doorbell");
  }

  constexpr uint32_t kStepUs = 1000;
  constexpr uint32_t kTimeoutUs = 5000000;
  uint64_t observed_api = 0;
  uint64_t observed_query = 0;
  for (uint32_t elapsed = 0; elapsed < kTimeoutUs; elapsed += kStepUs) {
    std::atomic_thread_fence(std::memory_order_acquire);
    observed_api = api_fence_cpu[0];
    observed_query = api_fence_cpu[1];
    if (observed_api == fence_value && observed_query == fence_value) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s MES API %s complete fence=%llu query=%llu "
                     "wptr=%llu\n",
                     TracePrefix(options), opcode_name ? opcode_name : "unknown",
                     static_cast<unsigned long long>(fence_value),
                     static_cast<unsigned long long>(observed_query),
                     static_cast<unsigned long long>(new_wptr));
      }
      return HSA_STATUS_SUCCESS;
    }
    if ((observed_api != 0 && observed_api != fence_value) ||
        (observed_query != 0 && observed_query != fence_value)) {
      break;
    }
    platform.SleepUs(kStepUs);
  }

  if (options.trace) {
    uint32_t rptr =
        ring.rptr_cpu != nullptr ? static_cast<uint32_t>(*ring.rptr_cpu) : 0;
    std::fprintf(stderr,
                 "%s MES API %s timeout/error observed_api=0x%llx "
                 "observed_query=0x%llx expected=%llu wptr=%llu rptr=%u\n",
                 TracePrefix(options), opcode_name ? opcode_name : "unknown",
                 static_cast<unsigned long long>(observed_api),
                 static_cast<unsigned long long>(observed_query),
                 static_cast<unsigned long long>(fence_value),
                 static_cast<unsigned long long>(ring.wptr), rptr);
    DumpMesRingState(platform, ring, options, "timeout");
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t SubmitMesApiFrame(const DirectQueuePlatform& platform,
                               MesSchedulerState& state,
                               std::array<uint32_t, kMesApiFrameDwords>& frame,
                               uint32_t api_status_dw,
                               const DirectQueueOptions& options,
                               const char* opcode_name) {
  return SubmitMesApiFrameOnRing(platform, state.ring, state.api_fence_cpu,
                                 state.api_fence_gpu, state.next_fence_value,
                                 frame, api_status_dw, options, opcode_name);
}

std::array<uint32_t, kMesApiFrameDwords> BuildMesSetHwResourcesFrame(
    const MesSchedulerState& state,
    uint64_t sch_ctx_gpu,
    uint64_t query_status_fence_gpu,
    bool include_scheduler_resources,
    uint32_t oversubscription_timer) {
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeSetHwResources);
  if (include_scheduler_resources) {
    frame[1] = 0x0000ff00u;
    frame[2] = 0x0000ffffu;
    frame[3] = 0;
    frame[4] = 0;
    for (uint32_t i = 0; i < 8; ++i) frame[5 + i] = 0xffu;
    frame[13] = 0xffu;
    frame[14] = 0xffu;
    frame[15] = 0xfcu;
    frame[16] = 0xfcu;
    for (uint32_t i = 0; i < state.aggregated_doorbells.size(); ++i) {
      frame[17 + i] = state.aggregated_doorbells[i];
    }
  }
  PutU64(frame.data(), 22, sch_ctx_gpu);
  PutU64(frame.data(), 24, query_status_fence_gpu);
  constexpr uint32_t kGcBases[] = {
      kGcBase0, kGcBase1, 0x0001c000u, 0x02402c00u, 0x02000112u};
  constexpr uint32_t kMmhubBases[] = {
      0x0001a000u, 0x02408800u, 0x0f0000ffu, 0x0003000eu,
      0x00016000u};
  constexpr uint32_t kOsssysBases[] = {
      0x000010a0u, 0x0240a000u, 0x03000046u, 0x00000106u,
      0x02411800u};
  for (uint32_t i = 0; i < 5; ++i) {
    frame[26 + i] = kGcBases[i];
    frame[34 + i] = kMmhubBases[i];
    frame[42 + i] = kOsssysBases[i];
  }
  frame[54] = kMesSetHwResourcesFlags;
  frame[55] = oversubscription_timer;
  return frame;
}

uint32_t MesOversubscriptionTimer(uint32_t version) {
  (void)version;
  return 50;
}

uint32_t ReadMesPipeVersion(const DirectQueuePlatform& platform,
                            uint32_t pipe) {
  uint32_t version = 0;
  if (SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd) ==
      HSA_STATUS_SUCCESS) {
    platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version);
    DeselectHqd(platform);
  }
  return version;
}

std::array<uint32_t, kMesApiFrameDwords> BuildMesSetHwResources1Frame(
    uint64_t cleaner_shader_fence_gpu) {
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeSetHwResources1);
  frame[13] = 0x0a;
  PutU64(frame.data(), 16, cleaner_shader_fence_gpu);
  return frame;
}

std::array<uint32_t, kMesApiFrameDwords> BuildMesMapLegacySchedulerFrame(
    const DirectQueueLayout& scheduler_layout) {
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeAddQueue);
  frame[18] = kMesSchedulerDoorbell;
  PutU64(frame.data(), 20, scheduler_layout.mqd_gpu);
  PutU64(frame.data(), 22, scheduler_layout.wptr_gpu);
  frame[28] = kMesQueueTypeScheduler;
  frame[37] = kMesAddQueueMapLegacyKq;
  frame[50] = 0;
  frame[51] = 0;
  return frame;
}

void InitMesAggregatedDoorbells(const DirectQueuePlatform& platform,
                                const MesSchedulerState& state) {
  constexpr uint32_t regs[] = {
      regCP_MES_DOORBELL_CONTROL1, regCP_MES_DOORBELL_CONTROL2,
      regCP_MES_DOORBELL_CONTROL3, regCP_MES_DOORBELL_CONTROL4,
      regCP_MES_DOORBELL_CONTROL5};
  for (uint32_t i = 0; i < state.aggregated_doorbells.size(); ++i) {
    uint32_t data = 0;
    if (platform.ReadMmio32(kGcBase1, regs[i], &data) != HSA_STATUS_SUCCESS) {
      continue;
    }
    data &= ~(kCpMesDoorbellOffsetMask | kCpMesDoorbellEnable |
              kCpMesDoorbellHit);
    data |= (state.aggregated_doorbells[i] << 2) | kCpMesDoorbellEnable;
    platform.WriteMmio32(kGcBase1, regs[i], data);
  }
  platform.WriteMmio32(kGcBase0, regCP_HQD_GFX_CONTROL,
                       kCpHqdGfxControlDbUpdatedMsgEn);
}

void EnableUnmappedDoorbellHandling(const DirectQueuePlatform& platform) {
  uint32_t data = 0;
  if (platform.ReadMmio32(kGcBase1, regCP_UNMAPPED_DOORBELL, &data) !=
      HSA_STATUS_SUCCESS) {
    return;
  }
  data &= ~kCpUnmappedDoorbellProcLsbMask;
  data |= 0xdu << kCpUnmappedDoorbellProcLsbShift;
  data |= kCpUnmappedDoorbellEnable;
  platform.WriteMmio32(kGcBase1, regCP_UNMAPPED_DOORBELL, data);
}

uint32_t QueueSizeField(uint32_t ring_size_bytes) {
  uint32_t dwords = ring_size_bytes / sizeof(uint32_t);
  uint32_t log2_dwords = 0;
  while (dwords > 1) {
    dwords >>= 1;
    ++log2_dwords;
  }
  return log2_dwords == 0 ? 0 : log2_dwords - 1;
}

uint32_t MesPqControl(uint32_t ring_size_bytes) {
  return (kCpHqdPqControlMes & ~0x3fu) |
         (QueueSizeField(ring_size_bytes) & 0x3fu);
}

DirectQueueMqd BuildMesKernelQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index,
                                      uint32_t ring_size) {
  DirectQueueMqd mqd{};
  mqd[0] = 0xC0310800;
  mqd[0x0B] = 1;
  constexpr uint32_t kStaticThreadMgmtDwords[] = {0x17u, 0x18u, 0x1Au, 0x1Bu};
  for (uint32_t dw : kStaticThreadMgmtDwords) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x20] = 7;

  const uint64_t eop_base_shifted = layout.eop_gpu >> 8;
  mqd[0xA5] = static_cast<uint32_t>(eop_base_shifted);
  mqd[0xA6] = static_cast<uint32_t>(eop_base_shifted >> 32);
  mqd[0xA7] = 8;  // Linux MES uses a 2 KiB EOP window.

  mqd[0x80] = static_cast<uint32_t>(layout.mqd_gpu) & 0xFFFFFFFCu;
  mqd[0x81] = static_cast<uint32_t>(layout.mqd_gpu >> 32);
  mqd[0x82] = 1;
  mqd[0x84] = (kCpHqdPersistentStateDefault & ~(0x3FFu << 8)) | (0x55u << 8);
  mqd[0x87] = 0x111;

  const uint64_t pq_base_shifted = layout.ring_gpu >> 8;
  mqd[0x88] = static_cast<uint32_t>(pq_base_shifted);
  mqd[0x89] = static_cast<uint32_t>(pq_base_shifted >> 32);
  mqd[0x8B] = static_cast<uint32_t>(layout.rptr_gpu) & 0xFFFFFFFCu;
  mqd[0x8C] = static_cast<uint32_t>(layout.rptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8D] = static_cast<uint32_t>(layout.wptr_gpu) & 0xFFFFFFF8u;
  mqd[0x8E] = static_cast<uint32_t>(layout.wptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8F] = ((doorbell_index & 0x03FFFFFFu) << 2) | (1u << 30);

  mqd[0x91] = MesPqControl(ring_size);
  mqd[0x95] = 0x00300000;
  mqd[0xA2] = kCpMqdControlDefault;
  mqd[0xB8] = 1u << 15;
  return mqd;
}

[[maybe_unused]] hsa_status_t SubmitKiqScratchTest(
    const DirectQueuePlatform& platform,
    MesSchedulerState& state,
    const DirectQueueOptions& options) {
  constexpr uint32_t kScratchValue = 0xdeadbeef;
  const uint32_t scratch_offset = kGcBase1 + regSCRATCH_REG0;
  platform.WriteMmio32(kGcBase1, regSCRATCH_REG0, 0xcafedead);
  const uint32_t pm4[] = {
      Packet3(kPacket3WriteData, 3),
      1u << 16,
      scratch_offset,
      0,
      kScratchValue,
  };
  hsa_status_t status =
      SubmitRingPm4(platform, state.kiq, pm4, sizeof(pm4) / sizeof(pm4[0]));
  if (status != HSA_STATUS_SUCCESS) return status;
  for (uint32_t i = 0; i < 100000; ++i) {
    uint32_t value = 0;
    platform.ReadMmio32(kGcBase1, regSCRATCH_REG0, &value);
    if (value == kScratchValue) {
      if (options.trace) {
        std::fprintf(stderr, "%s KIQ scratch test complete samples=%u\n",
                     TracePrefix(options), i + 1);
      }
      return HSA_STATUS_SUCCESS;
    }
    platform.SleepUs(1);
  }
  if (options.trace) {
    uint32_t rptr = state.kiq.rptr_cpu != nullptr
                        ? static_cast<uint32_t>(*state.kiq.rptr_cpu)
                        : 0;
    std::fprintf(stderr,
                 "%s KIQ scratch test timeout wptr=%llu rptr=%u\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(state.kiq.wptr), rptr);
  }
  return HSA_STATUS_ERROR;
}

[[maybe_unused]] hsa_status_t MapSchedulerWithKiq(
    const DirectQueuePlatform& platform,
    MesSchedulerState& state,
    const DirectQueueLayout& scheduler_layout,
    const DirectQueueOptions& options) {
  constexpr uint32_t kScratchValue = 0xdeadbeef;
  const uint32_t scratch_offset = kGcBase1 + regSCRATCH_REG0;
  platform.WriteMmio32(kGcBase1, regSCRATCH_REG0, 0xcafedead);
  const uint32_t map_control =
      (0u << 4) |   // queue select
      (0u << 8) |   // VMID
      (0u << 13) |  // queue
      (0u << 16) |  // pipe
      (2u << 18) |  // MES engine queue selector
      (0u << 21) |  // normal queue
      (0u << 24) |  // all on one pipe
      (5u << 26) |  // MES engine
      (1u << 29);
  const uint32_t pm4[] = {
      Packet3(kPacket3MapQueues, 5),
      map_control,
      kMesSchedulerDoorbell << 2,
      static_cast<uint32_t>(scheduler_layout.mqd_gpu),
      static_cast<uint32_t>(scheduler_layout.mqd_gpu >> 32),
      static_cast<uint32_t>(scheduler_layout.wptr_gpu),
      static_cast<uint32_t>(scheduler_layout.wptr_gpu >> 32),
      Packet3(kPacket3WriteData, 3),
      1u << 16,
      scratch_offset,
      0,
      kScratchValue,
  };
  hsa_status_t status =
      SubmitRingPm4(platform, state.kiq, pm4, sizeof(pm4) / sizeof(pm4[0]));
  if (status != HSA_STATUS_SUCCESS) return status;
  bool scratch_done = false;
  for (uint32_t i = 0; i < 100000; ++i) {
    uint32_t value = 0;
    platform.ReadMmio32(kGcBase1, regSCRATCH_REG0, &value);
    if (value == kScratchValue) {
      scratch_done = true;
      break;
    }
    platform.SleepUs(1);
  }
  if (options.trace) {
    const uint32_t kiq_rptr = state.kiq.rptr_cpu != nullptr
                                  ? static_cast<uint32_t>(*state.kiq.rptr_cpu)
                                  : 0;
    std::fprintf(stderr,
                 "%s MES scheduler MAP_QUEUES submitted by KIQ kiq_wptr=%llu "
                 "kiq_rptr=%u scratch=%s mqd=0x%llx wptr=0x%llx\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(state.kiq.wptr), kiq_rptr,
                 scratch_done ? "done" : "timeout",
                 static_cast<unsigned long long>(scheduler_layout.mqd_gpu),
                 static_cast<unsigned long long>(scheduler_layout.wptr_gpu));
  }
  return scratch_done ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t EnsureMesScheduler(const DirectQueuePlatform& platform,
                                uint64_t framebuffer_base,
                                const DirectQueueOptions& options) {
  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  MesSchedulerState& state = MesSchedulers()[&platform];
  if (state.initialized) return HSA_STATUS_SUCCESS;

  uint32_t mes_cntl = 0;
  hsa_status_t status = platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &mes_cntl);
  if (status != HSA_STATUS_SUCCESS) return status;
  if ((mes_cntl & (kCpMesCntlPipe0Active | kCpMesCntlPipe1Active)) !=
      (kCpMesCntlPipe0Active | kCpMesCntlPipe1Active)) {
    const uint32_t scheduler_version = ReadMesPipeVersion(platform, 0);
    const uint32_t kiq_version = ReadMesPipeVersion(platform, kMesKiqPipe);
    if (scheduler_version == 0 || kiq_version == 0) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s MES scheduler/KIQ pipe inactive CP_MES_CNTL=0x%08x "
                     "versions sched=0x%08x kiq=0x%08x; "
                     "run firmware bring-up first\n",
                     TracePrefix(options), mes_cntl, scheduler_version,
                     kiq_version);
      }
      return HSA_STATUS_ERROR;
    }
    if (options.trace) {
      std::fprintf(stderr,
                   "%s MES CP_MES_CNTL readback is 0x%08x but GP3 versions "
                   "are sched=0x%08x kiq=0x%08x; continuing\n",
                   TracePrefix(options), mes_cntl, scheduler_version,
                   kiq_version);
    }
  }

  hsa_status_t aperture_status = platform.EnsureDoorbellAperture();
  if (aperture_status != HSA_STATUS_SUCCESS) return aperture_status;

  DirectQueueLayout scheduler_layout{};
  DirectQueueLayout kiq_layout{};
  DirectQueueMemory scheduler_memory{};
  DirectQueueMemory kiq_memory{};
  status = PrepareQueueLayout(platform, framebuffer_base, kMesSchedulerBaseOffset,
                              &scheduler_layout, &scheduler_memory);
  if (status == HSA_STATUS_SUCCESS) {
    status = PrepareQueueLayout(platform, framebuffer_base, kMesKiqBaseOffset,
                                &kiq_layout, &kiq_memory);
  }
  if (status != HSA_STATUS_SUCCESS) {
    platform.FreeQueueMemory(&scheduler_memory);
    platform.FreeQueueMemory(&kiq_memory);
    return status;
  }

  DirectQueueMqd scheduler_mqd =
      BuildMesKernelQueueMqd(scheduler_layout, kMesSchedulerDoorbell,
                             kMesRingSize);
  DirectQueueMqd kiq_mqd =
      BuildMesKernelQueueMqd(kiq_layout, kMesKiqDoorbell, kMesRingSize);

  if (options.trace_verbose) {
    std::fprintf(stderr,
                 "%s MES layouts sched base=0x%llx mqd=0x%llx ring=0x%llx "
                 "rptr=0x%llx wptr=0x%llx doorbell=0x%x pq_ctl=0x%08x\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(scheduler_layout.base_gpu),
                 static_cast<unsigned long long>(scheduler_layout.mqd_gpu),
                 static_cast<unsigned long long>(scheduler_layout.ring_gpu),
                 static_cast<unsigned long long>(scheduler_layout.rptr_gpu),
                 static_cast<unsigned long long>(scheduler_layout.wptr_gpu),
                 kMesSchedulerDoorbell, scheduler_mqd[0x91]);
    std::fprintf(stderr,
                 "%s MES layouts kiq base=0x%llx mqd=0x%llx ring=0x%llx "
                 "rptr=0x%llx wptr=0x%llx doorbell=0x%x pq_ctl=0x%08x\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(kiq_layout.base_gpu),
                 static_cast<unsigned long long>(kiq_layout.mqd_gpu),
                 static_cast<unsigned long long>(kiq_layout.ring_gpu),
                 static_cast<unsigned long long>(kiq_layout.rptr_gpu),
                 static_cast<unsigned long long>(kiq_layout.wptr_gpu),
                 kMesKiqDoorbell, kiq_mqd[0x91]);
  }

  status = ZeroQueueMemory(platform, scheduler_layout, kMesRingSize);
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroQueueMemory(platform, kiq_layout, kMesRingSize);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(
        platform, scheduler_layout,
        scheduler_layout.base_offset + kMesSchedulerContextRelativeOffset,
        0x3000);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, kiq_layout,
                              kiq_layout.base_offset +
                                  kMesSchedulerContextRelativeOffset,
                              0x3000);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = WriteMqdImage(platform, scheduler_layout, scheduler_mqd);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = WriteMqdImage(platform, kiq_layout, kiq_mqd);
  }
  if (status != HSA_STATUS_SUCCESS) {
    platform.FreeQueueMemory(&scheduler_memory);
    platform.FreeQueueMemory(&kiq_memory);
    return status;
  }

  DirectQueueState ring = MakeRingState(platform, scheduler_layout,
                                        kMesSchedulerDoorbell, kMesRingSize);
  DirectQueueState kiq =
      MakeRingState(platform, kiq_layout, kMesKiqDoorbell, kMesRingSize);
  if (ring.ring_cpu == nullptr || ring.rptr_cpu == nullptr ||
      ring.wptr_cpu == nullptr || ring.doorbell_cpu == nullptr ||
      kiq.ring_cpu == nullptr || kiq.rptr_cpu == nullptr ||
      kiq.wptr_cpu == nullptr || kiq.doorbell_cpu == nullptr) {
    platform.FreeQueueMemory(&scheduler_memory);
    platform.FreeQueueMemory(&kiq_memory);
    return HSA_STATUS_ERROR;
  }

  state = {};
  state.ring = ring;
  state.kiq = kiq;
  state.scheduler_memory = scheduler_memory;
  state.kiq_memory = kiq_memory;
  scheduler_memory = {};
  kiq_memory = {};
  state.sch_ctx_gpu =
      scheduler_layout.base_gpu + kMesSchedulerContextRelativeOffset;
  state.query_status_fence_gpu =
      scheduler_layout.base_gpu + kMesQueryFenceRelativeOffset;
  state.api_fence_gpu = scheduler_layout.base_gpu + kMesApiFenceRelativeOffset;
  state.api_fence_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, scheduler_layout,
                       scheduler_layout.base_offset +
                           kMesApiFenceRelativeOffset,
                       sizeof(uint64_t) * 2));
  state.cleaner_fence_gpu =
      scheduler_layout.base_gpu + kMesCleanerFenceRelativeOffset;
  state.kiq_sch_ctx_gpu =
      kiq_layout.base_gpu + kMesSchedulerContextRelativeOffset;
  state.kiq_query_status_fence_gpu =
      kiq_layout.base_gpu + kMesQueryFenceRelativeOffset;
  state.kiq_api_fence_gpu = kiq_layout.base_gpu + kMesApiFenceRelativeOffset;
  state.kiq_api_fence_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, kiq_layout,
                       kiq_layout.base_offset + kMesApiFenceRelativeOffset,
                       sizeof(uint64_t) * 2));
  state.kiq_cleaner_fence_gpu =
      kiq_layout.base_gpu + kMesCleanerFenceRelativeOffset;
  if (state.api_fence_cpu == nullptr || state.kiq_api_fence_cpu == nullptr) {
    ResetMesSchedulerState(platform, state);
    return HSA_STATUS_ERROR;
  }
  for (uint32_t i = 0; i < state.aggregated_doorbells.size(); ++i) {
    state.aggregated_doorbells[i] = kMesAggregatedDoorbellBase + i * 2;
  }

  uint32_t schedulers = 0;
  if (platform.ReadMmio32(kGcBase1, regRLC_CP_SCHEDULERS, &schedulers) ==
      HSA_STATUS_SUCCESS) {
    schedulers &= 0xffffff00u;
    schedulers |= (kMesKiqMe << 5) | (kMesKiqPipe << 3) | kMesKiqHqd |
                  0x80u;
    platform.WriteMmio32(kGcBase1, regRLC_CP_SCHEDULERS, schedulers);
  }

  status = ProgramMesQueueRegisters(platform, kMesKiqPipe, kiq_mqd, options,
                                    "MES KIQ");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }
  status = SyncMesRingPointersFromHardware(platform, state.kiq, options,
                                           "KIQ");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }

  const uint32_t kiq_version = ReadMesPipeVersion(platform, kMesKiqPipe);
  const uint32_t scheduler_version = ReadMesPipeVersion(platform, 0);

  if (std::getenv("ROCR_AMDGPU_LITE_MES_QUERY_FIRST") != nullptr) {
    std::array<uint32_t, kMesApiFrameDwords> kiq_query_first{};
    kiq_query_first[0] = MesHeader(kMesOpcodeQuerySchedulerStatus);
    status = SubmitMesApiFrameOnRing(
        platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
        state.next_fence_value, kiq_query_first, 2, options,
        "KIQ QUERY_FIRST");
    if (status != HSA_STATUS_SUCCESS) {
      ResetMesSchedulerState(platform, state);
      return status;
    }
  }

  EnableUnmappedDoorbellHandling(platform);
  auto kiq_set_hw = BuildMesSetHwResourcesFrame(
      state, state.kiq_sch_ctx_gpu, state.kiq_query_status_fence_gpu, false,
      MesOversubscriptionTimer(kiq_version));
  status = SubmitMesApiFrameOnRing(
      platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
      state.next_fence_value, kiq_set_hw, kMesApiStatusSetHwResourcesDw,
      options, "KIQ SET_HW_RESOURCES");
  if (status == HSA_STATUS_SUCCESS) {
    auto kiq_set_hw1 =
        BuildMesSetHwResources1Frame(state.kiq_cleaner_fence_gpu);
    status = SubmitMesApiFrameOnRing(
        platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
        state.next_fence_value, kiq_set_hw1, kMesApiStatusSetHwResources1Dw,
        options,
        "KIQ SET_HW_RESOURCES_1");
  }
  if (status == HSA_STATUS_SUCCESS) {
    auto map_scheduler = BuildMesMapLegacySchedulerFrame(scheduler_layout);
    status = SubmitMesApiFrameOnRing(
        platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
        state.next_fence_value, map_scheduler, kMesApiStatusAddQueueDw,
        options, "KIQ MAP_SCHEDULER");
  }
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }

  auto set_hw = BuildMesSetHwResourcesFrame(
      state, state.sch_ctx_gpu, state.query_status_fence_gpu, true,
      MesOversubscriptionTimer(scheduler_version));
  status = SubmitMesApiFrame(platform, state, set_hw,
                             kMesApiStatusSetHwResourcesDw, options,
                             "SET_HW_RESOURCES");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }
  auto set_hw1 = BuildMesSetHwResources1Frame(state.cleaner_fence_gpu);
  status = SubmitMesApiFrame(platform, state, set_hw1,
                             kMesApiStatusSetHwResources1Dw, options,
                             "SET_HW_RESOURCES_1");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }
  InitMesAggregatedDoorbells(platform, state);

  state.initialized = true;
  if (options.trace) {
    uint32_t version = 0;
    platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version);
    std::fprintf(stderr,
                 "%s MES scheduler initialized doorbell=0x%x ring=0x%llx "
                 "version=0x%08x\n",
                 TracePrefix(options), kMesSchedulerDoorbell,
                 static_cast<unsigned long long>(scheduler_layout.ring_gpu),
                 version);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MapLegacyQueueWithMes(const DirectQueuePlatform& platform,
                                   DirectQueueState& queue,
                                   const DirectQueueLayout& layout,
                                   uint64_t framebuffer_base,
                                   const DirectQueueOptions& options) {
  hsa_status_t status = EnsureMesScheduler(platform, framebuffer_base, options);
  if (status != HSA_STATUS_SUCCESS) return status;

  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  auto it = MesSchedulers().find(&platform);
  if (it == MesSchedulers().end() || !it->second.initialized) {
    return HSA_STATUS_ERROR;
  }
  MesSchedulerState& state = it->second;
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeAddQueue);
  frame[18] = queue.doorbell_index;
  PutU64(frame.data(), 20, layout.mqd_gpu);
  PutU64(frame.data(), 22, layout.wptr_gpu);
  frame[28] = kMesQueueTypeCompute;
  frame[37] = kMesAddQueueMapLegacyKq;
  frame[50] = DirectQueuePipe(queue.queue_index);
  frame[51] = DirectQueueHqd(queue.queue_index);
  status = SubmitMesApiFrame(platform, state, frame, kMesApiStatusAddQueueDw,
                             options, "ADD_QUEUE");
  if (status != HSA_STATUS_SUCCESS) return status;

  queue.mes_backed = true;
  if (options.trace) {
    std::fprintf(stderr,
                 "%s MES mapped compute queue qid=%u index=%u pipe=%u hqd=%u "
                 "doorbell=0x%x mqd=0x%llx wptr=0x%llx\n",
                 TracePrefix(options), queue.queue_id, queue.queue_index,
                 DirectQueuePipe(queue.queue_index),
                 DirectQueueHqd(queue.queue_index), queue.doorbell_index,
                 static_cast<unsigned long long>(layout.mqd_gpu),
                 static_cast<unsigned long long>(layout.wptr_gpu));
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t UnmapLegacyQueueWithMes(const DirectQueuePlatform& platform,
                                     const DirectQueueState& queue,
                                     const DirectQueueOptions& options) {
  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  auto it = MesSchedulers().find(&platform);
  if (it == MesSchedulers().end() || !it->second.initialized) {
    return HSA_STATUS_ERROR;
  }
  MesSchedulerState& state = it->second;
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeRemoveQueue);
  frame[1] = queue.doorbell_index;
  frame[4] = kMesRemoveQueueUnmapLegacy;
  frame[10] = DirectQueuePipe(queue.queue_index);
  frame[11] = DirectQueueHqd(queue.queue_index);
  frame[15] = kMesQueueTypeCompute;
  hsa_status_t status = SubmitMesApiFrame(platform, state, frame,
                                          kMesApiStatusRemoveQueueDw, options,
                                          "REMOVE_QUEUE");
  if (options.trace) {
    std::fprintf(stderr,
                 "%s MES unmap compute queue qid=%u index=%u status=%u\n",
                 TracePrefix(options), queue.queue_id, queue.queue_index,
                 status);
  }
  return status;
}

}  // namespace

uint32_t DirectQueuePipe(uint32_t queue_index) { return queue_index / 4; }

uint32_t DirectQueueHqd(uint32_t queue_index) { return queue_index % 4; }

uint32_t DirectQueueDoorbell(uint32_t queue_index) {
  return kDirectComputeDoorbellBase + queue_index * kDirectComputeDoorbellStride;
}

DirectQueueLayout BuildDirectQueueLayout(uint64_t framebuffer_base,
                                         uint32_t queue_index) {
  return BuildQueueLayoutAt(framebuffer_base,
                            kDirectComputeBaseOffset +
                                queue_index * kDirectComputeStride);
}

DirectQueueMqd BuildPm4DirectQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index) {
  DirectQueueMqd mqd{};
  mqd[0] = 0xC0310800;
  mqd[1] = 1;
  constexpr uint32_t kStaticThreadMgmtDwords[] = {0x17u, 0x18u, 0x1Au, 0x1Bu};
  for (uint32_t dw : kStaticThreadMgmtDwords) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x20] = 7;  // compute_misc_reserved (dword 0x2C was a static_thread_mgmt_se4 slip)

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

  mqd[0x91] = kCpHqdPqControlPm4;
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
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr, "%s create failed: doorbell aperture status=%u\n",
                   TracePrefix(options), status);
    }
    return status;
  }

  *queue = {};
  queue->queue_index = queue_index;
  queue->queue_id = queue_index + 1;
  queue->doorbell_index = DirectQueueDoorbell(queue_index);
  queue->ring_size_bytes = kDirectComputeRingSize;
  if (options.trace) {
    std::fprintf(stderr,
                 "%s create qid=%u index=%u doorbell=0x%x mode=%s\n",
                 TracePrefix(options), queue->queue_id, queue->queue_index,
                 queue->doorbell_index,
                 options.use_mes_queue ? "mes" : "direct");
  }

  const uint32_t pipe = DirectQueuePipe(queue->queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue->queue_index);
  status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: select compute HQD pipe=%u hqd=%u "
                   "status=%u\n",
                   TracePrefix(options), pipe, hqd_queue, status);
    }
    *queue = {};
    return status;
  }

  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: read active pipe=%u hqd=%u status=%u\n",
                   TracePrefix(options), pipe, hqd_queue, status);
    }
    DeselectHqd(platform);
    *queue = {};
    return status;
  }
  if (active != 0 && !options.force_reclaim && !options.use_mes_queue) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: active HQD pipe=%u hqd=%u active=0x%x\n",
                   TracePrefix(options), pipe, hqd_queue, active);
    }
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

  DirectQueueLayout layout{};
  DirectQueueMemory queue_memory{};
  status = PrepareQueueLayout(
      platform, framebuffer_base,
      kDirectComputeBaseOffset + queue->queue_index * kDirectComputeStride,
      &layout, &queue_memory);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: allocate queue memory qid=%u index=%u "
                   "status=%u\n",
                   TracePrefix(options), queue->queue_id, queue->queue_index,
                   status);
    }
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  const DirectQueueMqd mqd = BuildPm4DirectQueueMqd(layout, queue->doorbell_index);

  status = ZeroQueueMemory(platform, layout);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: zero queue memory base_off=0x%llx "
                   "status=%u\n",
                   TracePrefix(options),
                   static_cast<unsigned long long>(layout.base_offset), status);
    }
    platform.FreeQueueMemory(&queue_memory);
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  status = WriteMqdImage(platform, layout, mqd);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: write MQD base_off=0x%llx status=%u\n",
                   TracePrefix(options),
                   static_cast<unsigned long long>(layout.base_offset), status);
    }
    platform.FreeQueueMemory(&queue_memory);
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  auto* ring_cpu = static_cast<volatile uint32_t*>(
      LayoutCpuPointer(platform, layout, layout.ring_offset,
                       kDirectComputeRingSize));
  auto* rptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.rptr_offset, sizeof(uint64_t)));
  auto* wptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.wptr_offset, sizeof(uint64_t)));
  volatile uint64_t* doorbell_cpu = platform.DoorbellCpuPointer(queue->doorbell_index);
  if (ring_cpu == nullptr || rptr_cpu == nullptr || wptr_cpu == nullptr ||
      doorbell_cpu == nullptr) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: CPU pointers ring=%p rptr=%p wptr=%p "
                   "doorbell=%p\n",
                   TracePrefix(options), const_cast<uint32_t*>(ring_cpu),
                   const_cast<uint64_t*>(rptr_cpu),
                   const_cast<uint64_t*>(wptr_cpu),
                   const_cast<uint64_t*>(doorbell_cpu));
    }
    platform.FreeQueueMemory(&queue_memory);
    DeselectHqd(platform);
    *queue = {};
    return HSA_STATUS_ERROR;
  }

  queue->ring_gpu = layout.ring_gpu;
  queue->wptr = 0;
  queue->memory = queue_memory;
  queue_memory = {};
  queue->ring_cpu = ring_cpu;
  queue->rptr_cpu = rptr_cpu;
  queue->wptr_cpu = wptr_cpu;
  queue->doorbell_cpu = doorbell_cpu;

  if (options.use_mes_queue) {
    DeselectHqd(platform);
    status = MapLegacyQueueWithMes(platform, *queue, layout, framebuffer_base,
                                   options);
    if (status != HSA_STATUS_SUCCESS) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s create failed: MES map qid=%u index=%u status=%u\n",
                     TracePrefix(options), queue->queue_id, queue->queue_index,
                     status);
      }
      platform.FreeQueueMemory(&queue->memory);
      *queue = {};
      return status;
    }
    return HSA_STATUS_SUCCESS;
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
  platform.WriteMmio32(kGcBase0, regCP_MQD_CONTROL, mqd[0xA2]);
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
    platform.FreeQueueMemory(&queue->memory);
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
    platform.FreeQueueMemory(&queue->memory);
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

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DestroyDirectQueue(const DirectQueuePlatform& platform,
                                DirectQueueState& queue,
                                const DirectQueueOptions& options) {
  if (queue.queue_id == 0) return HSA_STATUS_SUCCESS;
  if (options.skip_destroy) {
    if (options.trace) {
      std::fprintf(stderr, "%s destroy skipped qid=%u index=%u\n",
                   TracePrefix(options), queue.queue_id, queue.queue_index);
    }
    return HSA_STATUS_SUCCESS;
  }
  if (queue.mes_backed) {
    hsa_status_t status = UnmapLegacyQueueWithMes(platform, queue, options);
    const hsa_status_t free_status = platform.FreeQueueMemory(&queue.memory);
    if (status == HSA_STATUS_SUCCESS) status = free_status;
    queue = {};
    return status;
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
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST,
                         kCpHqdDequeueResetWaves);
    platform.WriteMmio32(kGcBase0, regSPI_COMPUTE_QUEUE_RESET, 1);
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
  const hsa_status_t free_status = platform.FreeQueueMemory(&queue.memory);
  queue = {};
  return free_status;
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
  hsa_status_t status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;

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

  if (queue.mes_backed) {
    *queue.doorbell_cpu = new_wptr;
    queue.wptr = new_wptr;
    if (options.trace) {
      const uint64_t rptr =
          queue.rptr_cpu != nullptr ? *queue.rptr_cpu : 0;
      std::fprintf(stderr,
                   "%s submit MES-backed qid=%u doorbell=0x%x "
                   "wptr=%llu rptr=%llu\n",
                   TracePrefix(options), queue.queue_id, queue.doorbell_index,
                   static_cast<unsigned long long>(new_wptr),
                   static_cast<unsigned long long>(rptr));
    }
    return HSA_STATUS_SUCCESS;
  }

  status = SelectHqd(platform, 1, pipe, hqd_queue);
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
  // Optionally skip the MMIO wptr poke: tinygrad/Linux deliver wptr only via the
  // VRAM wptr + doorbell value, and writing CP_HQD_PQ_WPTR_* on a live HQD races
  // the CP's own use of the context window. Default preserves the existing poke;
  // ROCR_MACOS_DIRECT_QUEUE_MMIO_WPTR=0 skips it (doorbell value latches wptr).
  const char* mmio_wptr_env = std::getenv("ROCR_MACOS_DIRECT_QUEUE_MMIO_WPTR");
  const bool skip_mmio_wptr =
      mmio_wptr_env != nullptr && std::strcmp(mmio_wptr_env, "0") == 0;
  if (!skip_mmio_wptr) {
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO,
                                  static_cast<uint32_t>(new_wptr));
    if (status == HSA_STATUS_SUCCESS) {
      status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI,
                                    static_cast<uint32_t>(new_wptr >> 32));
    }
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
  if (queue.mes_backed) {
    if (queue.rptr_cpu == nullptr) return HSA_STATUS_ERROR;
    *rptr = static_cast<uint32_t>(*queue.rptr_cpu);
    return HSA_STATUS_SUCCESS;
  }
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
