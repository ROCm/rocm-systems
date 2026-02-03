////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <cinttypes>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <utility>
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "impl/wddm/gpu_memory.h"
#include "hsa-runtime/inc/hsa_ext_amd.h"
#include "hsa-runtime/inc/amd_hsa_queue.h"
#include "hsa-runtime/inc/amd_hsa_signal.h"
#include "impl/wddm/cmd_util.h"

namespace wsl {
namespace thunk {

class Queue;
class WDDMDevice;

class WDDMQueue {
public:
  WDDMQueue(WDDMDevice *device,
            uint64_t cmdbuf_addr,
            uint32_t cmdbuf_size,
            uint32_t engine,
            bool use_hws = true) :
            device(device),
            context(0),
            queue(0),
            syncobj(0),
            sync_addr(NULL),
            cmdbuf(0),
            cmdbuf_addr(cmdbuf_addr),
            cmdbuf_size(cmdbuf_size),
            queue_engine(engine),
            use_hws(use_hws),
            prio(thunk_proxy::kNormal) {}

  virtual ~WDDMQueue() { }

  virtual hsa_status_t Init(void) { return HSA_STATUS_SUCCESS; }
  virtual hsa_status_t Fini(void) { return HSA_STATUS_SUCCESS; }
  virtual void RingDoorbell(uint64_t value) { }
  virtual void* GetHsaQueueAddr(void) const { return reinterpret_cast<void*>(GetCmdbufAddr()); }

  hsa_status_t SwsInit(void);
  hsa_status_t SwsFini(void);
  hsa_status_t SwsSubmit(uint64_t command_addr,
                         uint64_t command_size,
                         uint64_t fence_value);

  hsa_status_t HwsInit(void);
  hsa_status_t HwsFini(void);
  hsa_status_t HwsSubmit(uint64_t command_addr,
                         uint64_t command_size,
                         uint64_t fence_value);
  hsa_status_t SetPriority(hsa_amd_queue_priority_t priority);

  uint64_t *GetSyncAddr(void) const { return sync_addr; }
  uint64_t GetCmdbufAddr(void) const { return cmdbuf_addr; }

  thunk_proxy::SchedLevel ConvertSchedLevel(hsa_amd_queue_priority_t prio) const {
    switch (prio) {
    case HSA_AMD_QUEUE_PRIORITY_LOW:
      return thunk_proxy::kLow;
    case HSA_AMD_QUEUE_PRIORITY_HIGH:
      return thunk_proxy::kHigh;
    case HSA_AMD_QUEUE_PRIORITY_NORMAL:
    default:
      return thunk_proxy::kNormal;
    }
  }

  WDDMDevice *device;

  D3DKMT_HANDLE context;
  D3DKMT_HANDLE queue;

  D3DKMT_HANDLE syncobj;
  uint64_t *sync_addr;

  GpuMemoryHandle cmdbuf;
  uint64_t cmdbuf_addr;
  uint32_t cmdbuf_size;

  GpuMemoryHandle queue_mem;
  uint64_t queue_addr;

  uint32_t queue_engine;

  bool use_hws;
  thunk_proxy::SchedLevel prio;

  std::atomic<uint64_t>* ring_wptr = nullptr;
  std::atomic<uint64_t>* ring_rptr = nullptr;
};

class ComputeQueue : public WDDMQueue {
public:
  ComputeQueue(WDDMDevice *device,
               void *ring,
               uint64_t ring_size,
               std::atomic<uint64_t> *ring_wptr,
               std::atomic<uint64_t> *ring_rptr,
               volatile int64_t *error_addr,
               uint32_t cmdbuf_size,
               uint32_t engine,
               bool use_hws = true);

  ~ComputeQueue();

  virtual hsa_status_t Init(void);
  virtual hsa_status_t Fini(void);
  virtual hsa_status_t Submit(void);

  void* GetRing(void) const { return ring; }
  uint64_t GetRingSize(void) const { return ring_size; }
  std::atomic<uint64_t>* GetRingWptr(void) const { return ring_wptr; }
  std::atomic<uint64_t>* GetRingRptr(void) const { return ring_rptr; }

  uint64_t GetAqlWriteIndex(void) const { return cmdbuf_aql_frame_write_index; }
  uint32_t GetAqlFrameSize(void) const { return cmdbuf_aql_frame_size; }
  void* GetHsaQueueAddr(void) const { return ring; }

  bool IsInvalidPacket(void) const {
    uint16_t *packet = (uint16_t *)((char *)ring +
                       (cmdbuf_aql_frame_write_index % ring_size) * 64);
    return ((*packet >> HSA_PACKET_HEADER_TYPE) & ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1))
           == HSA_PACKET_TYPE_INVALID;
  }

  hsa_status_t Process(void);
  uint64_t * GetDoorbellPtr() const { return (uint64_t *)&doorbell_signal_value_; }
  void RingDoorbell(uint64_t value);
  GpuMemory* GetAmdQueueMemory() const { return amd_queue_memory_; }

 private:
  hsa_status_t KernelDispatchAqlToPm4(char *cpu, hsa_kernel_dispatch_packet_t *packet);
  hsa_status_t BarrierGenericAqlToPm4(char *cpu, hsa_barrier_and_packet_t *packet, bool is_or = false);

  uint64_t CalcDispatchGroups(hsa_kernel_dispatch_packet_t *packet);
  uint64_t CalcDispatchWavesPerGroup(hsa_kernel_dispatch_packet_t *packet, bool wave32);

  struct amd_aql_pm4_ib {
      uint16_t header;
      uint16_t ven_hdr;
      uint32_t ib_jump_cmd[4];
      uint32_t dw_cnt_remain;
      uint32_t reserved[8];
      hsa_signal_t completion_signal;
  };
  hsa_status_t VendorSpecificAqlToPm4(char *cpu, amd_aql_pm4_ib *packet);
  hsa_status_t SwitchAql2PM4(void);

  hsa_status_t PreSubmit(void);
  hsa_status_t EndSubmit(void);

  void *ring;         //!< AQL queue, allocated in ROCR and points to the AQL packets
  uint64_t ring_size; //!< AQL queue size in packets

  // ib_start_addr is the current ib start address
  uint64_t ib_start_addr;

  // ib_size is the current ib size.
  uint64_t ib_size;

  // record the last submitted aql frame write index
  uint64_t sync_point;

  uint64_t cmdbuf_aql_frame_write_index;
  uint32_t cmdbuf_aql_frame_size;

  uint64_t  *signal_addr_;
  bool platform_atomic_support_;
  bool needs_barrier;
  bool ready_to_submit;

  CmdUtil cmd_util;

private:
  bool EnableProfiling() {
    return AMD_HSA_BITS_GET(amd_queue_rocr_->queue_properties, AMD_QUEUE_PROPERTIES_ENABLE_PROFILING);
  }
  void HandleError(hsa_status_t status);
  bool UpdateScratch(hsa_kernel_dispatch_packet_t *packet, bool wave32);

  uint32_t UpdateIndexStride(uint32_t srd, bool wave32);

  void *ScratchBase() { return scratch_base_; }

  void AppendCmdbufSratchBaseOffset(int offset) {
      scratch_base_offset_array_.push_back(offset);
  }

  bool RelocateCmdbufScratchBase(uint64_t addr);

  uint32_t ScratchSizePerWave() { return scratch_size_per_wave_; }
  uint64_t GetKernelObjAddr(uint64_t addr) const;
  void InitScratchSRD();
  GpuMemoryHandle amd_queue_mem_;
  GpuMemory* amd_queue_memory_;     //!< Memory object associated with amd_queue_t structure from ROCr
  amd_queue_v2_t *amd_queue_;
  amd_queue_v2_t *amd_queue_rocr_;  //!< AQL queue, allocated in rocr and pointing to the header
  uint64_t amd_queue_size_rocr_;    //!< Size of the AQL queue allocated in ROCR, including header
  uint64_t doorbell_signal_value_;
  volatile std::atomic<int64_t> *error_code_;
  std::thread aql_to_pm4_thread_;
  bool thread_stop_;
  std::mutex thread_cond_lock_;
  std::condition_variable thread_cond_;
  static void AqlToPm4Thread(ComputeQueue *queue);

  uint64_t max_scratch_waves_;
  uint64_t dispatch_waves_;
  uint64_t scratch_size_per_wave_;
  uint64_t scratch_size_;
  uint64_t total_scratch_size_;
  void *scratch_base_;
  uint32_t scratch_mem_alignment_size_;
  GpuMemoryHandle scratch_mem_;

  std::vector<int> scratch_base_offset_array_;
  bool aql_;  //!< The queue is configured to the AQL execution
};

class SDMAQueue : public WDDMQueue {
public:
  SDMAQueue(WDDMDevice *device,
            void *ring,
            uint64_t cmdbuf_size,
            uint32_t engine,
            bool use_hws = true);

  virtual ~SDMAQueue();

  hsa_status_t Init(void);
  hsa_status_t Fini(void);
  hsa_status_t Submit(void);

  int PreparePacket(uint32_t offset, uint64_t size);

  void WaitQueue(void) {
    device->CpuWait(&syncobj, &rptr_next, 1, false);
  }

  uint64_t * GetRingWptr(void) { return &wptr_next_; }
  uint64_t * GetRingRptr(void) { return WDDMQueue::GetSyncAddr(); }
  uint64_t * GetDoorbellPtr() { return &doorbell_; }
  void RingDoorbell(uint64_t value);
  void* GetHsaQueueAddr(void) const { return reinterpret_cast<void*>(GetCmdbufAddr()); }

private:
  uint64_t wptr_next_;
  uint64_t wptr_pre_;
  uint64_t rptr_next;
  uint64_t doorbell_;
  std::vector<std::pair<uint64_t, uint64_t>> wptr_queue_;
  uint64_t ib_size;
  uint64_t ib_start_addr;

  std::thread thread_;
  bool thread_stop_;
  std::mutex thread_cond_lock_;
  std::condition_variable thread_cond_;
  static void SdmaThread(SDMAQueue *queue);

  struct SDMA_PKT_POLL_REGMEM {
    union {
      struct {
        unsigned int op : 8;
        unsigned int sub_op : 8;
        unsigned int reserved_0 : 10;
        unsigned int hdp_flush : 1;
        unsigned int reserved_1 : 1;
        unsigned int func : 3;
        unsigned int mem_poll : 1;
      };
      unsigned int DW_0_DATA;
    } HEADER_UNION;

    union {
      struct {
        unsigned int addr_31_0 : 32;
      };
      unsigned int DW_1_DATA;
    } ADDR_LO_UNION;

    union {
      struct {
        unsigned int addr_63_32 : 32;
      };
      unsigned int DW_2_DATA;
    } ADDR_HI_UNION;

    union {
      struct {
        unsigned int value : 32;
      };
      unsigned int DW_3_DATA;
    } VALUE_UNION;

    union {
      struct {
        unsigned int mask : 32;
      };
      unsigned int DW_4_DATA;
    } MASK_UNION;

    union {
      struct {
        unsigned int interval : 16;
        unsigned int retry_count : 12;
        unsigned int reserved_0 : 4;
      };
      unsigned int DW_5_DATA;
    } DW5_UNION;
  };
  const unsigned int SDMA_OP_POLL_REGMEM = 8;
  bool IsPollPacket(SDMA_PKT_POLL_REGMEM* pkt) {
    return pkt->HEADER_UNION.op == SDMA_OP_POLL_REGMEM &&
          pkt->HEADER_UNION.mem_poll == 1 &&
          pkt->HEADER_UNION.func == 3;
  }
  uint32_t WrapIntoRocrRing(uint64_t idx) { return (idx & (cmdbuf_size - 1)); }
};

} // namespace thunk
} // namespace wsl

