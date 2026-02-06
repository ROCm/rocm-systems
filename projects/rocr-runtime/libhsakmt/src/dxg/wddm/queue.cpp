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

#include <cstring>
#include <cinttypes>
#include <cstddef>

#include "impl/wddm/queue.h"
#include "impl/registers.h"

#include "hsa-runtime/inc/hsa.h"
#include "hsa-runtime/inc/hsa_ven_amd_loader.h"

extern hsa_signal_value_t hsakmt_hsa_signal_load_relaxed(hsa_signal_t signal);
extern hsa_signal_value_t hsakmt_hsa_signal_wait_relaxed(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);
extern void hsakmt_hsa_signal_store_screlease(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value);
extern hsa_status_t hsakmt_hsa_ven_amd_loader_query_host_address(
    const void *device_address, const void **host_address);
extern wsl::thunk::GpuMemory* GetGpuMemoryFromAddress(void* memory_address);

namespace wsl {
namespace thunk {

hsa_status_t WDDMQueue::SwsInit(void) {
  if (!device->CreateSyncobj(&syncobj, &sync_addr))
    return HSA_STATUS_ERROR;

  if (device->AllocUserQueueMemFromUMD()) {

    GpuMemory *gpu_mem = nullptr;
    GpuMemoryCreateInfo create_info{};

    create_info.domain = thunk_proxy::kUserQueue;
    create_info.size = device->GetSwsQueueSize();
    // GetComputeEngine returns schedId instead of engine flag
    create_info.engine_flag = device->GetComputeEngine();

    auto code = device->CreateGpuMemory(create_info, &gpu_mem);
    if (code != ErrorCode::Success) {
      device->DestroySyncobj(syncobj);
      syncobj = 0;
      return HSA_STATUS_ERROR;
    }

    queue_mem = gpu_mem->GetGpuMemoryHandle();
    queue = gpu_mem->GetAllocationHandle(0);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMQueue::SwsFini(void) {
  if (device->AllocUserQueueMemFromUMD()) {
    auto gpu_mem = GpuMemory::Convert(queue_mem);
    delete gpu_mem;
  }

  if (syncobj) {
    device->DestroySyncobj(syncobj);
    syncobj = 0;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMQueue::SwsSubmit(uint64_t command_addr,
                                  uint64_t command_size,
                                  uint64_t fence_value) {
  if (!device->SubmitToSwQueue(this, command_addr, command_size, fence_value))
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMQueue::HwsInit(void) {
  if (!device->CreateHwQueue(this))
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMQueue::HwsFini(void) {
  if (!device->DestroyHwQueue(this))
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMQueue::HwsSubmit(uint64_t command_addr,
                                  uint64_t command_size,
                                  uint64_t fence_value) {
  if (!device->SubmitToHwQueue(this, command_addr, command_size, fence_value))
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMQueue::SetPriority(hsa_amd_queue_priority_t priority) {
  if (!use_hws)
    return HSA_STATUS_SUCCESS;

  thunk_proxy::SchedLevel new_prio = ConvertSchedLevel(priority);
  if (prio == new_prio)
    return HSA_STATUS_SUCCESS;

  pr_debug("set prio %d -> %d\n", prio, new_prio);
  device->DestroyHwQueue(this);

  prio = new_prio;
  return HwsInit();
}

void ComputeQueue::HandleError(hsa_status_t status) {
  hsa_signal_t sig = amd_queue_rocr_->queue_inactive_signal;
  hsa_signal_value_t val = -1;

  struct queue_error_t {
    uint32_t code;
    hsa_status_t status;
  };
  static const queue_error_t QueueErrors[] = {
    {2, HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS},
    {4, HSA_STATUS_ERROR_INVALID_ALLOCATION},
    {8, HSA_STATUS_ERROR_INVALID_CODE_OBJECT},
    //{16, HSA_STATUS_ERROR_INVALID_ARGUMENT},
    {32, HSA_STATUS_ERROR_INVALID_PACKET_FORMAT},
    {64, HSA_STATUS_ERROR_INVALID_ARGUMENT},
    //{128, HSA_STATUS_ERROR_OUT_OF_REGISTERS},
    //{0x20000000, HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION},
    //{0x40000000, HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION},
    {0x80000000, HSA_STATUS_ERROR_EXCEPTION},
  };
  for (std::size_t i = 0; i < sizeof(QueueErrors) / sizeof(QueueErrors[0]); ++i) {
    if (QueueErrors[i].status == status) {
      val = QueueErrors[i].code;
      pr_err("error %d, sig_val %ld\n", status, val);
      break;
    }
  }

  if (sig.handle) {
    hsakmt_hsa_signal_store_screlease(sig, val);
  }
  if (error_code_) {
    error_code_->store(val, std::memory_order_release);
  }
}

void ComputeQueue::AqlToPm4Thread(ComputeQueue *queue) {

  // This timing system is used for sleeping this Thread
  // when one packet is invalid for about 2 seconds.
  std::chrono::steady_clock::time_point start_time, time;
  // Set the polling timeout value for 2 seconds
  const std::chrono::milliseconds kMaxElapsed(2000);
  uint64_t current_position = queue->GetAqlWriteIndex();
  bool sleep = false;
  start_time = std::chrono::steady_clock::now();

  while (true) {
    if (!queue->IsInvalidPacket()) {
      hsa_status_t status = queue->Process();
      if (status != HSA_STATUS_SUCCESS) {
        pr_err("process compute queue fail status = %08x\n", status);
        queue->HandleError(status);
        break;
      }
      sleep = false;
    } else {
      if (current_position == queue->GetAqlWriteIndex()) {
        time = std::chrono::steady_clock::now();
        if (time - start_time > kMaxElapsed)
          sleep = true;
      } else {
        start_time = std::chrono::steady_clock::now();
        current_position = queue->GetAqlWriteIndex();
        sleep = false;
      }
    }

    if ((queue->GetRingWptr()->load() > queue->GetRingRptr()->load()) && !sleep)
      continue;

    std::unique_lock<std::mutex> lock(queue->thread_cond_lock_);
    // CPU wait for valid packet
    if (queue->GetRingWptr()->load() <= queue->GetRingRptr()->load() ||
        (sleep && queue->IsInvalidPacket())) {
      if (queue->thread_stop_)
        break;
      pr_debug("wait %p wptr=%" PRIx64 " rptr=%" PRIx64 "\n",
               queue->ring, queue->GetRingWptr()->load(), queue->GetRingRptr()->load());
      queue->thread_cond_.wait(lock);
    }
  }

  pr_debug("aql to pm4 thread %p exit\n", queue->ring);
}

ComputeQueue::ComputeQueue(WDDMDevice *device,
               void *ring,
               uint64_t ring_size,
               std::atomic<uint64_t>* _ring_wptr,
               std::atomic<uint64_t>* _ring_rptr,
               volatile int64_t *error_addr,
               uint32_t cmdbuf_size,
               uint32_t engine,
               bool use_hws)
    : WDDMQueue(
          device, device->IsAqlSupported() ? reinterpret_cast<uintptr_t>(ring) : 0,
               device->IsAqlSupported() ? ring_size * 64 : cmdbuf_size, engine, use_hws),
               ring(ring),
               ring_size(ring_size),
               error_code_(reinterpret_cast<volatile std::atomic<int64_t>*>(error_addr)),
               ib_start_addr(0),
               ib_size(0),
               sync_point(0),
               cmdbuf_aql_frame_write_index(0),
               cmdbuf_aql_frame_size(0),
               needs_barrier(true),
               ready_to_submit(false),
               platform_atomic_support_(false),
               signal_addr_(NULL),
               thread_stop_(false),
               max_scratch_waves_(device->MaxScratchSlotsPerCu() * device->ComputeUnitCount()),
               dispatch_waves_(0),
               scratch_size_per_wave_(0),
               scratch_size_(0),
               total_scratch_size_(0),
               scratch_base_(nullptr) {
  ring_wptr = _ring_wptr;
  ring_rptr = _ring_rptr;
  amd_queue_rocr_ = (amd_queue_v2_t*)((char*)ring_rptr - offsetof(amd_queue_t, read_dispatch_id));
  amd_queue_memory_ = GetGpuMemoryFromAddress(amd_queue_rocr_);
  aql_ = device->DeviceInfo().hwsInfo.hwsMask.aql_queue;
  bool ret = device->CreateQueue(this);
  assert(ret);

  GpuMemoryCreateInfo create_info{};
  create_info.size = dxg_runtime->page_size;
  create_info.domain = thunk_proxy::kSystem;
  GpuMemory *gpu_mem = nullptr;
  auto code = device->CreateGpuMemory(create_info, &gpu_mem);
  assert(code == ErrorCode::Success);
  amd_queue_mem_ = gpu_mem->GetGpuMemoryHandle();
  amd_queue_ = reinterpret_cast<amd_queue_v2_t*>(gpu_mem->GpuAddress());

  amd_queue_rocr_ = (amd_queue_v2_t*)((char*)ring_rptr - offsetof(amd_queue_v2_t, read_dispatch_id));
  // Don't start the PM4 thread for AQL queue
  if (!aql_) {
    aql_to_pm4_thread_ = std::thread(AqlToPm4Thread, this);
  }

  if (device->Major() >= 11)
    scratch_mem_alignment_size_ = 256;
  else
    scratch_mem_alignment_size_ = 1024;
}

ComputeQueue::~ComputeQueue() {
  if (!aql_) {
    thread_cond_lock_.lock();
    thread_stop_ = true;
    thread_cond_lock_.unlock();
    thread_cond_.notify_one();
    aql_to_pm4_thread_.join();
  }

  //doorbell_signal_->Release();

  device->DestroyQueue(this);

  if (scratch_base_) {
    auto scratch_gpu_mem = GpuMemory::Convert(scratch_mem_);
    delete scratch_gpu_mem;
  }

  auto amd_queue_gpu_mem = GpuMemory::Convert(amd_queue_mem_);
  delete amd_queue_gpu_mem;
}

void ComputeQueue::InitScratchSRD() {
  // Populate scratch resource descriptor
  SQ_BUF_RSRC_WORD0 srd0;

  uintptr_t scratch_base = uintptr_t(scratch_base_);
  srd0.bits.BASE_ADDRESS = scratch_base;

  uint32_t srd1_u32;

  if (device->Major() < 11) {
    SQ_BUF_RSRC_WORD1 srd1;

    srd1.bits.BASE_ADDRESS_HI = scratch_base >> 32;
    srd1.bits.STRIDE = 0;
    srd1.bits.CACHE_SWIZZLE = 0;
    srd1.bits.SWIZZLE_ENABLE = 1;

    srd1_u32 = srd1.u32All;
  } else {
    SQ_BUF_RSRC_WORD1_GFX11 srd1;

    srd1.bits.BASE_ADDRESS_HI = scratch_base >> 32;
    srd1.bits.STRIDE = 0;
    srd1.bits.SWIZZLE_ENABLE = 1;

    srd1_u32 = srd1.u32All;
  }

  SQ_BUF_RSRC_WORD2 srd2;

  srd2.bits.NUM_RECORDS = scratch_size_;

  uint32_t srd3_u32;

  if (device->Major() < 10) {
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
    srd3.bits.ATC__CI__VI = 0;
    srd3.bits.HASH_ENABLE = 0;
    srd3.bits.HEAP = 0;
    srd3.bits.MTYPE__CI__VI = 0;
    srd3.bits.TYPE = SQ_RSRC_BUF;

    srd3_u32 = srd3.u32All;
  } else if (device->Major() == 10) {
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

    srd3_u32 = srd3.u32All;
  } else if (device->Major() == 11) {
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

    srd3_u32 = srd3.u32All;
  } else {
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

    srd3_u32 = srd3.u32All;
  }

  // Update Queue's Scratch descriptor's property
  amd_queue_->scratch_resource_descriptor[0] = srd0.u32All;
  amd_queue_->scratch_resource_descriptor[1] = srd1_u32;
  amd_queue_->scratch_resource_descriptor[2] = srd2.u32All;
  amd_queue_->scratch_resource_descriptor[3] = srd3_u32;

  // Populate flat scratch parameters in amd_queue_.
  amd_queue_->scratch_backing_memory_location = scratch_base;

  // For backwards compatibility this field records the per-lane scratch
  // for a 64 lane wavefront. If scratch was allocated for 32 lane waves
  // then the effective size for a 64 lane wave is halved.
  amd_queue_->scratch_wave64_lane_byte_size = scratch_size_per_wave_ / 64;

  uint64_t num_waves;
  if (device->Major() < 11) {
    COMPUTE_TMPRING_SIZE tmpring_size;
    // Scratch Size per Wave is specified in terms of scratch_mem_alignment_size_
    tmpring_size.bits.WAVESIZE = scratch_size_per_wave_ / scratch_mem_alignment_size_;
    num_waves = scratch_size_ / scratch_size_per_wave_;
    tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves_);

    amd_queue_->compute_tmpring_size = tmpring_size.u32All;
  } else if (device->Major() == 11) {
    COMPUTE_TMPRING_SIZE_GFX11 tmpring_size;
    tmpring_size.bits.WAVESIZE = scratch_size_per_wave_ / scratch_mem_alignment_size_;
    // For GFX11 we specify number of waves per engine instead of total
    num_waves = scratch_size_ / scratch_size_per_wave_ / device->NumShaderEngine();
    tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves_);

    amd_queue_->compute_tmpring_size = tmpring_size.u32All;
  } else {
    COMPUTE_TMPRING_SIZE_GFX12 tmpring_size = {};
    tmpring_size.bits.WAVESIZE = scratch_size_per_wave_ / scratch_mem_alignment_size_;
    // For GFX12 we specify number of waves per engine instead of total
    num_waves = scratch_size_ / scratch_size_per_wave_ / device->NumShaderEngine();
    tmpring_size.bits.WAVES = std::min(num_waves, max_scratch_waves_);

    amd_queue_->compute_tmpring_size = tmpring_size.u32All;
  }

  return;
}

uint64_t ComputeQueue::CalcDispatchGroups(hsa_kernel_dispatch_packet_t *packet)
{
  const uint64_t lanes_per_group =
      (uint64_t(packet->workgroup_size_x) * packet->workgroup_size_y) * packet->workgroup_size_z;

  uint64_t groups = ((uint64_t(packet->grid_size_x) + packet->workgroup_size_x - 1) /
                      packet->workgroup_size_x) *
                      ((uint64_t(packet->grid_size_y) + packet->workgroup_size_y - 1) /
                      packet->workgroup_size_y) *
                      ((uint64_t(packet->grid_size_z) + packet->workgroup_size_z - 1) /
                      packet->workgroup_size_z);
  const uint32_t cu_count = device->ComputeUnitCount();
  const uint32_t engines = device->NumShaderEngine();

  const uint32_t symmetric_cus = rocr::AlignDown(cu_count, engines);
  const uint32_t asymmetryPerRound = cu_count - symmetric_cus;
  const uint64_t rounds = groups / cu_count;
  const uint64_t asymmetricGroups = rounds * asymmetryPerRound;
  const uint64_t symmetricGroups = groups - asymmetricGroups;

  uint64_t maxGroupsPerEngine =
        ((symmetricGroups + engines - 1) / engines) + (asymmetryPerRound ? rounds : 0);

  // For gfx10+ devices we must attempt to assign the smaller of 256 lanes or 16 groups to each
  // engine.
  if (device->Major() >= 10 &&
      maxGroupsPerEngine < 16 &&
      lanes_per_group * maxGroupsPerEngine < 256) {
    uint64_t groups_per_interleave = (256 + lanes_per_group - 1) / lanes_per_group;
    maxGroupsPerEngine = std::min(groups_per_interleave, uint64_t(16ul));
  }

  return maxGroupsPerEngine * engines;
}

uint64_t ComputeQueue::CalcDispatchWavesPerGroup(hsa_kernel_dispatch_packet_t *packet,
                                                  bool wave32)
{
  const uint32_t lanes_per_wave = wave32 ? 32 : 64;

  const uint64_t lanes_per_group =
      (uint64_t(packet->workgroup_size_x) * packet->workgroup_size_y) * packet->workgroup_size_z;

  return (lanes_per_group + lanes_per_wave - 1) / lanes_per_wave;
}

bool ComputeQueue::UpdateScratch(hsa_kernel_dispatch_packet_t *packet, bool wave32) {
  const uint32_t lanes_per_wave = wave32 ? 32 : 64;
  const uint64_t size_per_thread = rocr::AlignUp(packet->private_segment_size,
                                  scratch_mem_alignment_size_ / lanes_per_wave);

  uint64_t groups = CalcDispatchGroups(packet);
  uint64_t waves_per_group = CalcDispatchWavesPerGroup(packet, wave32);

  // For packet batching, the maximum value must be used to fit all packets.
  scratch_size_per_wave_ = std::max(size_per_thread * lanes_per_wave, scratch_size_per_wave_);
  dispatch_waves_ = std::max(groups * waves_per_group, dispatch_waves_);

  const uint64_t max_scratch_size = scratch_size_per_wave_ * max_scratch_waves_;
  const uint64_t dispatch_size = scratch_size_per_wave_ * dispatch_waves_;

  scratch_size_ = std::min(dispatch_size, max_scratch_size);

  if (total_scratch_size_ >= scratch_size_)
    return true;

  pr_debug("need realloc scratch buffer, size %x -> %x\n",
           total_scratch_size_, scratch_size_);

  GpuMemoryCreateInfo create_info{};
  create_info.size = scratch_size_;
  create_info.domain = thunk_proxy::kLocal;
  GpuMemory *gpu_mem = nullptr;
  auto code = device->CreateGpuMemory(create_info, &gpu_mem);
  if (code != ErrorCode::Success)
    return false;

  if (scratch_base_) {
    auto scratch_gpu_mem = GpuMemory::Convert(scratch_mem_);
    delete scratch_gpu_mem;
  }

  total_scratch_size_ = scratch_size_;
  scratch_base_ = reinterpret_cast<void *>(gpu_mem->GpuAddress());
  scratch_mem_ = gpu_mem->GetGpuMemoryHandle();

  InitScratchSRD();
  return true;
}

bool ComputeQueue::RelocateCmdbufScratchBase(uint64_t addr) {
  if (scratch_base_offset_array_.empty())
    return true;

  for (size_t i = 0; i < scratch_base_offset_array_.size(); i++) {
    uint32_t *p_compute_user_data =
      reinterpret_cast<uint32_t *>(addr + scratch_base_offset_array_[i]);
    if (device->Major() >= 11) {
      p_compute_user_data[0] = Ptr48Low32(scratch_base_);
      p_compute_user_data[1] = Ptr48High8(scratch_base_);
    } else {
      p_compute_user_data[0] = PtrLow32(scratch_base_);
      p_compute_user_data[1] = (p_compute_user_data[1] & 0xffff0000) | PtrHigh32(scratch_base_);
    }
  }
  scratch_base_offset_array_.clear();

  return true;
}

uint32_t ComputeQueue::UpdateIndexStride(uint32_t srd, bool wave32) {

  assert(device->Major() < 13);

  if (device->Major() == 10) {
    SQ_BUF_RSRC_WORD3_GFX10 srd3;

    srd3.u32All = srd;
    srd3.bits.INDEX_STRIDE = wave32 ? 2 : 3;

    return srd3.u32All;
  } else if (device->Major() == 11) {
    SQ_BUF_RSRC_WORD3_GFX11 srd3;

    srd3.u32All = srd;
    srd3.bits.INDEX_STRIDE = wave32 ? 2 : 3;

    return srd3.u32All;
  } else if (device->Major() == 12) {
    SQ_BUF_RSRC_WORD3_GFX12 srd3;

    srd3.u32All = srd;
    srd3.bits.INDEX_STRIDE = wave32 ? 2 : 3;

    return srd3.u32All;
  }

  return srd;
}

uint64_t ComputeQueue::GetKernelObjAddr(uint64_t addr) const {
  /* convert dev_addr to host_addr */
  auto code = get_gpu_mem((void*)addr);
  if (code && code->IsBlitKernelObject()) {
    return code->GpuAddress();
  }

  uint64_t host_addr = 0;
  auto ret = hsakmt_hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(addr),
                                           reinterpret_cast<const void **>(&host_addr));
  if (ret == HSA_STATUS_SUCCESS) {
    return host_addr;
  }
  pr_err("failed to query host address for kernel object %p, ret=%d\n", (void*)addr, ret);
  return 0;
}

void ComputeQueue::RingDoorbell(uint64_t value) {
  if (!aql_) {
    thread_cond_lock_.lock();
    thread_cond_lock_.unlock();
    pr_debug("notify %p wptr=%" PRIx64 " rptr=%" PRIx64 "\n", ring, GetRingWptr()->load(),
             GetRingRptr()->load());
    thread_cond_.notify_one();
  } else {
    constexpr uint32_t kSizeOfAqlPacket = 64;
    auto aql_addr = reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(ring) +
                                                (value % ring_size) * kSizeOfAqlPacket);
    if (!device->SubmitToAqlQueue(this, aql_addr, kSizeOfAqlPacket, value)) {
      assert(!"Doorbell failed!");
    }
  }
}

hsa_status_t ComputeQueue::Init(void) {
  hsa_status_t ret = use_hws ? HwsInit() : SwsInit();
  if (ret) {
    return ret;
  }

  ib_start_addr = cmdbuf_addr;
  cmdbuf_aql_frame_size = device->GetAqlFrameSize();
  platform_atomic_support_ = device->SupportPlatformAtomic();

  return ret;
}

hsa_status_t ComputeQueue::Fini(void) {
  return use_hws ? HwsFini() : SwsFini();
}

hsa_status_t ComputeQueue::PreSubmit(void) {
  if (!device->WaitPagingFence(this))
    return HSA_STATUS_ERROR;

  RelocateCmdbufScratchBase(ib_start_addr);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ComputeQueue::EndSubmit(void) {
  // record last submitted cmdbuf_aql_frame_write_index to see if GPU is hungry
  sync_point = cmdbuf_aql_frame_write_index;

  ib_start_addr = cmdbuf_addr +
                  (cmdbuf_aql_frame_write_index % WDDMDevice::GetAqlFrameNum()) *
                  cmdbuf_aql_frame_size;
  ib_size = 0;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ComputeQueue::Submit(void) {
  hsa_status_t ret = PreSubmit();
  if (ret)
    return HSA_STATUS_ERROR;

  ret = use_hws ?
        HwsSubmit(ib_start_addr, ib_size, cmdbuf_aql_frame_write_index) :
        SwsSubmit(ib_start_addr, ib_size, cmdbuf_aql_frame_write_index);
  if (ret)
    return HSA_STATUS_ERROR;

  ret = EndSubmit();
  if (ret)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
ComputeQueue::KernelDispatchAqlToPm4(char *cpu, hsa_kernel_dispatch_packet_t *packet) {
  pr_debug("queue %p kernel dispatch head=%x setup=%x wx=%x wy=%x wz=%x "
           "gx=%x gy=%x gz=%x ps=%x gs=%x ko=%" PRIx64 " ka=%p cs=%" PRIx64 "\n",
           ring, packet->header,
           packet->setup, packet->workgroup_size_x, packet->workgroup_size_y,
           packet->workgroup_size_z, packet->grid_size_x, packet->grid_size_y,
           packet->grid_size_z, packet->private_segment_size,
           packet->group_segment_size, packet->kernel_object, packet->kernarg_address,
           packet->completion_signal.handle);

  if (packet->workgroup_size_x > 1024 ||
      packet->workgroup_size_y > 1024 ||
      packet->workgroup_size_z > 1024)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  int major = device->Major();
  int i = ib_size;

  const amd_kernel_code_t* kernel_object =
    (const amd_kernel_code_t *)GetKernelObjAddr(packet->kernel_object);
  if (kernel_object == NULL) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  void* entry = (void*)(packet->kernel_object + kernel_object->kernel_code_entry_byte_offset);
  assert((size_t)entry % AMD_ISA_ALIGN_BYTES == 0);

  pr_debug("kernel object property=%x entry=%p lds=%x+%x\n",
           kernel_object->kernel_code_properties, entry,
           kernel_object->workgroup_group_segment_byte_size,
           packet->group_segment_size);

  if (packet->setup == 0 || packet->setup > 3)
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  if (packet->group_segment_size > device->LdsSize())
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;

  uint32_t lds_blks = device->LdsBlocks(packet);
  if (lds_blks > 128)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const bool wave32 =
    AMD_HSA_BITS_GET(kernel_object->kernel_code_properties,
		     AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32);

  assert(packet->private_segment_size >= kernel_object->workitem_private_segment_byte_size);

  if (packet->private_segment_size != 0)
    UpdateScratch(packet, wave32);

  amd_signal_t *signal = (amd_signal_t *)packet->completion_signal.handle;

  // Record start timestamp when enabling profiling
  if (signal && EnableProfiling())
    i += cmd_util.BuildCopyData(&signal->start_ts, cpu + i);

  // Build a barrier packet if it is requested
  const bool is_barrier_packet = (packet->header >> HSA_PACKET_HEADER_BARRIER) & 0x1;
  if (is_barrier_packet && needs_barrier)
    i += cmd_util.BuildBarrier(cpu + i);

  // flush cache
  i += cmd_util.BuildAcquireMem(major, cpu + i);

  if (major >= 11) {
    AppendCmdbufSratchBaseOffset(
      i + offsetof(struct SetScratchTemplate, scratch_lo));

    i += cmd_util.BuildScratch(ScratchBase(), cpu + i);
    i += cmd_util.BuildComputeShaderParams(cpu + i);
  }

  struct DispatchInfo info;
  info.major = major;
  info.pPacket = packet;
  info.pEntry = entry;
  info.pKernelObject = kernel_object;
  info.ldsBlks = lds_blks;
  info.pAmdQueue = amd_queue_;
  info.wave32 = wave32;
  info.srd = UpdateIndexStride(
    info.pAmdQueue->scratch_resource_descriptor[3], wave32);
  info.pScratchBase = ScratchBase();
  info.scratchSizePerWave = ScratchSizePerWave();
  memset(info.scratchBaseOffset, 0, sizeof(info.scratchBaseOffset));
  info.offsetCnt = 0;

  size_t size;
  size = cmd_util.BuildDispatch(&info, cpu + i);
  for (int j = 0; j < info.offsetCnt; j++)
    AppendCmdbufSratchBaseOffset(i + info.scratchBaseOffset[j]);
  i += size;

  needs_barrier = (packet->completion_signal.handle == 0);

  if (signal) {
    // wait cs done
    i += cmd_util.BuildBarrier(cpu + i);

    // Record end timestamp when enabling profiling
    if (EnableProfiling())
      i += cmd_util.BuildCopyData(&signal->end_ts, cpu + i);

    // flush cache
    i += cmd_util.BuildAcquireMem(major, cpu + i);

    assert(signal->kind == AMD_SIGNAL_KIND_USER);
    uint64_t *signal_addr = (uint64_t *)&signal->value;
    pr_debug("signal value=%" PRIx64 "\n", signal->value);

    if (platform_atomic_support_)
      i += cmd_util.BuildAtomicMem(signal_addr, TC_OP_ATOMIC_ADD_RTN_64, cpu + i, cache_policy__mec_atomic_mem__bypass, -1);
    else
      signal_addr_ = signal_addr;
  }

  // The ring_rptr is used to record pm4 queue rptr value,
  // dispatch readptr position, this is used to share rptr with
  // aql queue.
  if (platform_atomic_support_)
    i += cmd_util.BuildAtomicMem((uint64_t *)ring_rptr, TC_OP_ATOMIC_ADD_RTN_64, cpu + i);
  else
    i += cmd_util.BuildWriteData64Command(cpu + i, (uint64_t *)ring_rptr, cmdbuf_aql_frame_write_index + 1);

  // Check if we exceeded the frame size
  if ((i - ib_size) > cmdbuf_aql_frame_size) {
    pr_err("PM4 command buffer overflow in KernelDispatch: used %d bytes, limit %d bytes\n", i - ib_size, cmdbuf_aql_frame_size);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  ib_size = i;
  cmdbuf_aql_frame_write_index++;
  packet->header = HSA_PACKET_TYPE_INVALID;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t
ComputeQueue::BarrierGenericAqlToPm4(char *cpu, hsa_barrier_and_packet_t *packet, bool is_or) {
  pr_debug("queue %p %s head=%x dep %" PRIx64 " %" PRIx64 " %" PRIx64
           " %" PRIx64 " %" PRIx64 " cs=%" PRIx64"\n",
           ring, is_or ? "or" : "and",
           packet->header, packet->dep_signal[0].handle,
           packet->dep_signal[1].handle, packet->dep_signal[2].handle,
           packet->dep_signal[3].handle, packet->dep_signal[4].handle,
           packet->completion_signal.handle);
  // fix me: can we use gpu packet?
  if (is_or) {
    bool unsignaled = true;
    hsa_signal_t sig[5];
    int n = 0;
    for (int i = 0; i < 5; i++) {
        if (packet->dep_signal[i].handle)
          sig[n++] = packet->dep_signal[i];
    }

    while (n) {
      for (int i = 0; i < n; i++) {
        if (!hsakmt_hsa_signal_load_relaxed(sig[i])) {
          unsignaled = false;
          break;
        }
      }
      if (!unsignaled)
        break;

      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  } else {
    for (int i = 0; i < 5; i++) {
      if (!packet->dep_signal[i].handle)
        continue;

    hsa_signal_value_t value =
      hsakmt_hsa_signal_wait_relaxed(packet->dep_signal[i], HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    assert(value == 0);
    }
  }

  int major = device->Major();
  int i = ib_size;

  if (packet->completion_signal.handle != 0) {
    amd_signal_t *signal = (amd_signal_t *)packet->completion_signal.handle;
    assert(signal->kind == AMD_SIGNAL_KIND_USER);
    uint64_t *signal_addr = (uint64_t *)&signal->value;
    pr_debug("signal value=%" PRIx64 "\n", signal->value);

    // Record start timestamp when enabling profiling
    if (EnableProfiling())
      i += cmd_util.BuildCopyData(&signal->start_ts, cpu + i);

    if (needs_barrier)
      i += cmd_util.BuildBarrier(cpu + i);

    needs_barrier = false;

    // Record end timestamp when enabling profiling
    if (EnableProfiling())
      i += cmd_util.BuildCopyData(&signal->end_ts, cpu + i);

    // flush cache
    i += cmd_util.BuildAcquireMem(major, cpu + i);

    if (platform_atomic_support_)
      i += cmd_util.BuildAtomicMem(signal_addr, TC_OP_ATOMIC_ADD_RTN_64, cpu + i, cache_policy__mec_atomic_mem__bypass, -1);
    else
      signal_addr_ = signal_addr;
  }

  // The ring_rptr is used to record pm4 queue rptr value,
  // dispatch readptr position, this is used to share rptr with
  // aql queue.
  if (platform_atomic_support_)
    i += cmd_util.BuildAtomicMem((uint64_t *)ring_rptr, TC_OP_ATOMIC_ADD_RTN_64, cpu + i);
  else
    i += cmd_util.BuildWriteData64Command(cpu + i, (uint64_t *)ring_rptr, cmdbuf_aql_frame_write_index + 1);

  // Check if we exceeded the frame size
  if ((i - ib_size) > cmdbuf_aql_frame_size) {
    pr_err("PM4 command buffer overflow in BarrierGeneric: used %d bytes, limit %d bytes\n", i - ib_size, cmdbuf_aql_frame_size);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  ib_size = i;
  cmdbuf_aql_frame_write_index++;
  packet->header = HSA_PACKET_TYPE_INVALID;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ComputeQueue::VendorSpecificAqlToPm4(char *cpu, amd_aql_pm4_ib *packet) {
  constexpr uint32_t AMD_AQL_FORMAT_PM4_IB = 0x1;
  assert(packet->ven_hdr == AMD_AQL_FORMAT_PM4_IB);

  uint8_t op = (packet->ib_jump_cmd[0] >> PM4_OPCODE_SHIFT) & 0xff;
  assert(op == IT_INDIRECT_BUFFER);
  uint32_t* pm4_addr = reinterpret_cast<uint32_t*>((static_cast<uint64_t>(packet->ib_jump_cmd[2]) << 32) | (static_cast<uint64_t>(packet->ib_jump_cmd[1]) & ~3ull));
  uint32_t pm4_size = packet->ib_jump_cmd[3]&0xfffff;
  pr_debug("queue %p %s VENDOR_SPECIFIC pkt pm4_addr %p pm4_size %#x cs=%" PRIx64"\n",
           ring, dxg_runtime->vendor_packet_process ? "process" : "skip", pm4_addr, pm4_size,
           packet->completion_signal.handle);
  for (int i = 0; i < pm4_size; i++) {
    pr_debug("pm4_addr[%d]=%#x\n", i, pm4_addr[i]);
  }

  int i = ib_size;

  if (dxg_runtime->vendor_packet_process) {
    int major = device->Major();
    memcpy(cpu+i, pm4_addr, pm4_size * sizeof(uint32_t));
    i += pm4_size * sizeof(uint32_t);

    if (packet->completion_signal.handle != 0) {
      amd_signal_t *signal = (amd_signal_t *)packet->completion_signal.handle;
      assert(signal->kind == AMD_SIGNAL_KIND_USER);
      uint64_t *signal_addr = (uint64_t *)&signal->value;
      pr_debug("signal value=%" PRIx64 "\n", signal->value);

      // Record start timestamp when enabling profiling
      if (EnableProfiling())
        i += cmd_util.BuildCopyData(&signal->start_ts, cpu + i);

      //if (needs_barrier)
        i += cmd_util.BuildBarrier(cpu + i);

      //needs_barrier = false;

      // Record end timestamp when enabling profiling
      if (EnableProfiling())
        i += cmd_util.BuildCopyData(&signal->end_ts, cpu + i);

      // flush cache
      i += cmd_util.BuildAcquireMem(major, cpu + i);

      if (platform_atomic_support_)
        i += cmd_util.BuildAtomicMem(signal_addr, TC_OP_ATOMIC_ADD_RTN_64, cpu + i, cache_policy__mec_atomic_mem__bypass, -1);
      else
        signal_addr_ = signal_addr;
    }
  } else {
    if (packet->completion_signal.handle != 0) {
      hsakmt_hsa_signal_store_screlease(packet->completion_signal, 0);
    }
  }

  // The ring_rptr is used to record pm4 queue rptr value,
  // dispatch readptr position, this is used to share rptr with
  // aql queue.
  if (platform_atomic_support_)
    i += cmd_util.BuildAtomicMem((uint64_t *)ring_rptr, TC_OP_ATOMIC_ADD_RTN_64, cpu + i);
  else
    i += cmd_util.BuildWriteData64Command(cpu + i, (uint64_t *)ring_rptr, cmdbuf_aql_frame_write_index + 1);

  // Check if we exceeded the frame size
  if ((i - ib_size) > cmdbuf_aql_frame_size) {
    pr_err("PM4 command buffer overflow in VendorSpecific: used %d bytes, limit %d bytes\n", i - ib_size, cmdbuf_aql_frame_size);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  ib_size = i;
  cmdbuf_aql_frame_write_index++;
  packet->header = HSA_PACKET_TYPE_INVALID;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ComputeQueue::SwitchAql2PM4(void) {

  uint16_t *packet = (uint16_t *) ((char *)ring +
    (cmdbuf_aql_frame_write_index % ring_size) * 64);
  uint16_t header = (*packet >> HSA_PACKET_HEADER_TYPE);
  header &= (1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
  hsa_kernel_dispatch_packet_t *aql_packet =
    (hsa_kernel_dispatch_packet_t *)packet;
  hsa_status_t ret;

  switch (header) {
  case HSA_PACKET_TYPE_KERNEL_DISPATCH:
    ret = KernelDispatchAqlToPm4((char *)ib_start_addr, aql_packet);
    if (ret != HSA_STATUS_SUCCESS)
      return ret;

    // Stop merging packages util below conditions are met:
    // 1) The kernel with completion signal;
    // 2) The cmdbuf_aql_frame_write_index reaches the end of cmdbuf
    // 3) The HW queue is empty now, submit the packet right now.
    // 4) The AQL queue is empty now, submit the packet right now.
    if (!(aql_packet->completion_signal.handle) &&
        (cmdbuf_aql_frame_write_index % WDDMDevice::GetAqlFrameNum()) &&
        (*sync_addr != sync_point) &&
        (cmdbuf_aql_frame_write_index != GetRingWptr()->load()))
      return HSA_STATUS_SUCCESS;

    break;
  case HSA_PACKET_TYPE_BARRIER_AND:
    BarrierGenericAqlToPm4((char *)ib_start_addr, (hsa_barrier_and_packet_t *)aql_packet);
    break;
  case HSA_PACKET_TYPE_BARRIER_OR:
    BarrierGenericAqlToPm4((char *)ib_start_addr, (hsa_barrier_and_packet_t *)aql_packet, true);
    break;
  case HSA_PACKET_TYPE_VENDOR_SPECIFIC:
    VendorSpecificAqlToPm4((char *)ib_start_addr, (amd_aql_pm4_ib *)aql_packet);
    break;
  case HSA_PACKET_TYPE_INVALID:
    // When packets are submitted out of order, the format field of current AQL packet
    // may not have been updated yet and is still INVALID. Return HSA_STATUS_SUCCESS and
    // do not process AQL packets before the packet format field is updated.
    assert(false && "Should not reach here, HSA_PACKET_TYPE_INVALID has been filtered in upper layer");
    return HSA_STATUS_SUCCESS;
  default:
    return HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;
  }

  ready_to_submit = true;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ComputeQueue::Process(void) {

  while (cmdbuf_aql_frame_write_index < ring_wptr->load() &&
         !IsInvalidPacket()) {
    pr_debug("process %p wptr=%" PRIx64 " rptr=%" PRIx64 "\n",
             ring, ring_wptr->load(), ring_rptr->load());

    hsa_status_t ret;

    // wait for next few cmdbuf slots to be free
    // If wptr catch up the rptr in the cmdbuf, this needs wait for the rptr to free the cmdbuf.
    // Here the wptr comes from queue->cmdbuf_aql_frame_write_index, while rptr comes from *queue->sync_addr.
    if (*sync_addr + WDDMDevice::GetAqlFrameNum() <= cmdbuf_aql_frame_write_index) {
      uint64_t value = cmdbuf_aql_frame_write_index - WDDMDevice::GetAqlFrameNum() + 1;
      if (!device->CpuWait(&syncobj, &value, 1, false))
        return HSA_STATUS_ERROR;
    }

    ret = SwitchAql2PM4();
    if (ret != HSA_STATUS_SUCCESS)
      return ret;

    if (!ready_to_submit)
      continue;

    ret = Submit();
    if (ret != HSA_STATUS_SUCCESS)
      return ret;

    // CPU wait for GPU fence, and cpu update the signal.
    if (!platform_atomic_support_ && signal_addr_) {
      // CPU wait for GPU fence
      if (!device->CpuWait(&syncobj, &cmdbuf_aql_frame_write_index, 1, false))
        return HSA_STATUS_ERROR;
      //CPU update completional signal
      rocr::atomic::Decrement(signal_addr_);
      signal_addr_ = NULL;
    }

    ready_to_submit = false;

    pr_debug("done %p wptr=%" PRIx64 " rptr=%" PRIx64 "\n",
             ring, ring_wptr->load(), ring_rptr->load());
  }

  return HSA_STATUS_SUCCESS;
}

void SDMAQueue::SdmaThread(SDMAQueue *queue) {

  while (true) {
    decltype(queue->wptr_queue_) pendings;
    {
      std::unique_lock<std::mutex> lock(queue->thread_cond_lock_);
      while (queue->wptr_queue_.empty() && !queue->thread_stop_)
        queue->thread_cond_.wait(lock);

      if (queue->thread_stop_)
        break;

      pendings.swap(queue->wptr_queue_);
    }

    for (const auto [start, end] : pendings) {
      pr_debug("wptr %lx %lx\n", start, end);

      SDMA_PKT_POLL_REGMEM* poll_pkt = reinterpret_cast<SDMA_PKT_POLL_REGMEM*>(queue->cmdbuf_addr + queue->WrapIntoRocrRing(start));
      SDMA_PKT_POLL_REGMEM* poll_next_pkt = poll_pkt + 1;
      while (queue->IsPollPacket(poll_pkt)) {
        uint64_t poll_addr = poll_pkt->ADDR_LO_UNION.addr_31_0 |
                             (uint64_t)poll_pkt->ADDR_HI_UNION.addr_63_32 << 32;

        uint64_t poll_val = poll_pkt->VALUE_UNION.value;
        uint32_t skip = 1;

        if (queue->IsPollPacket(poll_next_pkt)) {
          uint64_t poll_next_addr = poll_next_pkt->ADDR_LO_UNION.addr_31_0 |
                             (uint64_t)poll_next_pkt->ADDR_HI_UNION.addr_63_32 << 32;

          if (poll_next_addr + sizeof(uint32_t) == poll_addr) {
            poll_addr = poll_next_addr;
            poll_val = poll_next_pkt->VALUE_UNION.value |
                            (uint64_t)poll_pkt->VALUE_UNION.value << 32;
            skip = 2;
          }
        }

        amd_signal_t* signal = (amd_signal_t*)((char*)poll_addr - offsetof(amd_signal_t, value));
        uint64_t signal_handle = reinterpret_cast<uint64_t>(signal);
        pr_debug("poll signal %#lx addr %#lx val %ld\n", signal_handle, poll_addr, poll_val);
        hsa_signal_t hsa_signal = {signal_handle};
        hsa_signal_value_t value =
          hsakmt_hsa_signal_wait_relaxed(hsa_signal, HSA_SIGNAL_CONDITION_EQ, poll_val, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        assert(value == poll_val);

        memset(poll_pkt, 0, skip * sizeof(*poll_pkt));
        poll_pkt += skip;
        poll_next_pkt += skip;
      }
      queue->PreparePacket(queue->WrapIntoRocrRing(start), end - start);
      std::atomic_thread_fence(std::memory_order_release);
      queue->Submit();
    }
  }
  pr_debug("sdma thread exit\n");
}

SDMAQueue::SDMAQueue(WDDMDevice *device,
          void *ring,
          uint64_t cmdbuf_size,
          uint32_t engine,
          bool use_hws) :
          WDDMQueue(device, reinterpret_cast<uint64_t>(ring), cmdbuf_size, engine, use_hws),
          wptr_next_(0),
          wptr_pre_(0),
          rptr_next(0),
          thread_stop_(false),
          ib_size(0),
          ib_start_addr(0) {
  bool ret = device->CreateQueue(this);
  assert(ret);

  thread_ = std::thread(SdmaThread, this);
}

SDMAQueue::~SDMAQueue() {
  thread_cond_lock_.lock();
  thread_stop_ = true;
  thread_cond_lock_.unlock();
  thread_cond_.notify_one();
  thread_.join();

  device->DestroyQueue(this);
}

void SDMAQueue::RingDoorbell(uint64_t value) {
  pr_debug("ringdoorbell %#lx %#lx\n", wptr_pre_, wptr_next_);
  thread_cond_lock_.lock();

  wptr_queue_.emplace_back(wptr_pre_, wptr_next_);
  thread_cond_.notify_one();

  thread_cond_lock_.unlock();
  wptr_pre_ = wptr_next_;
}

hsa_status_t SDMAQueue::Init(void) {
  hsa_status_t ret = use_hws ? HwsInit() : SwsInit();
  if (ret)
    return ret;

  std::memset((char *)cmdbuf_addr, 0, cmdbuf_size);

  return ret;
}

hsa_status_t SDMAQueue::Fini(void) {
  return use_hws ? HwsFini() : SwsFini();
}

int SDMAQueue::PreparePacket(uint32_t offset, uint64_t size) {
  ib_start_addr = cmdbuf_addr + offset;
  ib_size = size;
  rptr_next += ib_size;

  return STATUS_SUCCESS;
}

hsa_status_t SDMAQueue::Submit(void) {
  if (!device->WaitPagingFence(this))
    return HSA_STATUS_ERROR;

  int ret = use_hws ?
            HwsSubmit(ib_start_addr, ib_size, rptr_next) :
            SwsSubmit(ib_start_addr, ib_size, rptr_next);
  if (ret)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

} // namespace thunk
} // namespace wsl
