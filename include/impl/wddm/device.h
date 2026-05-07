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

#ifndef _WSL_INC_WDDM_DEVICE_H_
#define _WSL_INC_WDDM_DEVICE_H_

#include <cassert>
#include <ntstatus.h>

#include <atomic>
#include <memory>
#include <vector>

#include "shared/include/d3dkmt_types.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include "impl/wddm/va_mgr.h"
#include "shared/include/status.h"
#include "shared/include/d3dkmt_types.h"
#include "impl/wddm/gpu_memory.h"
#include "impl/wddm/cmd_util.h"

namespace wsl {
namespace thunk {

//class Queue;
class Device;
class WDDMQueue;

// WSL2 hyperv GPADL protocol limitation
#define MAX_USERPTR_BLOCK_SIZE 0xf0000000
#define START_NON_CANONICAL_ADDR (1ULL << 47)
#define END_NON_CANONICAL_ADDR (~0UL - (1UL << 47))
#define IS_OVERLAPPING(start1, size1, start2, size2) \
  ((start1 < (start2 + size2)) && (start2 < (start1 + size1)))

struct SegmentInfo {
  uint32_t segment_id;
  uint32_t segment_type;    // 0=aperture, 1=gpu memory, 2=system memory
  bool aperture;
  bool system_memory;
  uint64_t commit_limit;

  SegmentInfo()
      : segment_id(0), segment_type(0), aperture(false),
        system_memory(false), commit_limit(0) {}
};

class WDDMDevice {
public:
  static constexpr size_t GpuMemoryChunkSize = 2 * (1ULL << 30);   // 2 GB

  WDDMDevice(Device *shared_dev,
             D3DKMT_HANDLE adapter, LUID adapter_luid, uint32_t node_id);
  ~WDDMDevice();

  Device *SharedDevice() const { return shared_dev_; }

  int NodeId() const { return node_id_; }
  int Major() { return shared_dev_->DeviceInfo().major; }
  int Minor() { return shared_dev_->DeviceInfo().minor; }
  int Stepping() { return shared_dev_->DeviceInfo().stepping; }
  bool IsDgpu() { return shared_dev_->DeviceInfo().is_dgpu; }
  const char *ProductName() { return shared_dev_->DeviceInfo().product_name; }
  uint64_t Uuid() { return shared_dev_->DeviceInfo().uuid; }
  uint32_t GfxFamily() { return shared_dev_->DeviceInfo().family; }
  uint32_t DeviceId() { return shared_dev_->DeviceInfo().device_id; }
  uint32_t WavefrontSize() { return shared_dev_->DeviceInfo().wavefront_size; }
  uint32_t ComputeUnitCount() { return shared_dev_->DeviceInfo().compute_unit_count; }
  uint32_t MaxEngineClockMhz() { return shared_dev_->DeviceInfo().max_engine_clock_mhz; }
  uint32_t WatchPointsNum() { return shared_dev_->DeviceInfo().watch_points_num; }
  uint32_t PciBusAddr() { return shared_dev_->DeviceInfo().pci_bus_addr; }

  uint32_t MemoryBusWidth() { return shared_dev_->DeviceInfo().memory_bus_width; }
  uint32_t MaxMemoryClockMhz() { return shared_dev_->DeviceInfo().max_memory_clock_mhz; }
  uint32_t WavePerCu() { return shared_dev_->DeviceInfo().wave_per_cu; }
  uint32_t SimdPerCu() { return shared_dev_->DeviceInfo().simd_per_cu; }
  uint32_t MaxScratchSlotsPerCu() { return shared_dev_->DeviceInfo().max_scratch_slots_per_cu; }
  uint32_t NumShaderEngine() { return shared_dev_->DeviceInfo().num_shader_engine; }
  uint32_t ShaderArrayPerShaderEngine() { return shared_dev_->DeviceInfo().shader_array_per_shader_engine; }
  uint32_t NumSdmaEngine() { return shared_dev_->DeviceInfo().sdma_schedid.size(); }
  uint32_t Domain() { return shared_dev_->DeviceInfo().domain; }
  uint32_t NumGws() { return shared_dev_->DeviceInfo().num_gws; }
  uint32_t AsicRevision() { return shared_dev_->DeviceInfo().asic_revision; }
  uint64_t LocalHeapSize() { return shared_dev_->DeviceInfo().local_visible_heap_size + shared_dev_->DeviceInfo().local_invisible_heap_size; }
  uint64_t LocalVisibleHeapSize() { return shared_dev_->DeviceInfo().local_visible_heap_size; }
  uint64_t LocalInvisibleHeapSize() { return shared_dev_->DeviceInfo().local_invisible_heap_size; }
  uint64_t NonLocalHeapSize() { return shared_dev_->DeviceInfo().non_local_heap_size; }
  uint64_t PrivateApertureBase() { return shared_dev_->DeviceInfo().private_aperture_base; }
  uint64_t PrivateApertureSize() { return shared_dev_->DeviceInfo().private_aperture_size; }
  uint64_t SharedApertureBase() { return shared_dev_->DeviceInfo().shared_aperture_base; }
  uint64_t SharedApertureSize() { return shared_dev_->DeviceInfo().shared_aperture_size; }
  uint32_t LdsSize() { return shared_dev_->DeviceInfo().lds_size; }
  uint64_t GPUCounterFrequency() { return shared_dev_->DeviceInfo().gpu_counter_frequency; }
  uint32_t GetSwsQueueSize(void) const { return shared_dev_->DeviceInfo().user_queue_size; }
  uint32_t GetMecFwVersion() { return shared_dev_->DeviceInfo().mec_fw_version; }
  uint32_t GetSdmaFwVersion() { return shared_dev_->DeviceInfo().sdma_fw_version; }
  uint32_t GetL1CacheSize() { return shared_dev_->DeviceInfo().l1_cache_size; }
  uint32_t GetL2CacheSize() { return shared_dev_->DeviceInfo().l2_cache_size; }
  uint32_t GetL3CacheSize() { return shared_dev_->DeviceInfo().l3_cache_size; }
  uint32_t Gl2CacheLineSize() { return shared_dev_->DeviceInfo().gl2_cacheline_size; }
  bool SupportStateShadowingByCpFw(void) const { return shared_dev_->DeviceInfo().state_shadowing_by_cpfw; }
  bool SupportPlatformAtomic(void) const { return shared_dev_->DeviceInfo().platform_atomic_support; }
  uint32_t GetSdmaEngine(uint32_t idx) {
    assert(idx < NumSdmaEngine());
    return shared_dev_->DeviceInfo().sdma_schedid[idx];
  }
  uint32_t GetComputeEngine() { return shared_dev_->DeviceInfo().compute_schedid; }

  uint64_t VramAvail();

  void GetClockCounters(uint64_t *gpu, uint64_t *cpu);
  uint32_t GetNumCpQueues() { return shared_dev_->DeviceInfo().num_cp_queues; }

  bool CreateSyncobj(D3DKMT_HANDLE *handle, uint64_t **addr);
  void DestroySyncobj(D3DKMT_HANDLE handle);

  bool CreateQueue(WDDMQueue *queue);
  void DestroyQueue(WDDMQueue *queue);
  bool CreateHwQueue(WDDMQueue *queue);
  bool DestroyHwQueue(WDDMQueue *queue);
  bool SubmitToSwQueue(WDDMQueue *queue, uint64_t command_addr,
                      uint64_t command_size, uint64_t fence_value);
  bool SubmitToHwQueue(WDDMQueue *queue, uint64_t command_addr,
                      uint64_t command_size, uint64_t fence_value);

  bool WaitPagingFence(WDDMQueue *queue) {
    uint64_t value = page_fence_value_;

    if (*page_fence_addr_ < value &&
        !GpuWait(queue, &page_syncobj_, &value, 1))
      return false;

    return true;
  }

  bool GpuWait(WDDMQueue *queue, const D3DKMT_HANDLE *syncobjs,
	       uint64_t *values, int count);
  bool GpuSignal(D3DKMT_HANDLE context, const D3DKMT_HANDLE *syncobjs,
		  uint64_t *value, int count);
  bool CpuWait(const D3DKMT_HANDLE *syncobjs, uint64_t *value,
	       int count, bool wait_any);
  bool WaitOnPagingFenceFromCpu();

  uint32_t LdsBlocks(const hsa_kernel_dispatch_packet_t *pkt);
  uint32_t GetCmdbufSize(void) const { return cmdbuf_size_; }
  uint32_t GetAqlFrameSize(void) const { return cmdbuf_aql_frame_size_; }
  static uint32_t GetAqlFrameNum(void) { return cmdbuf_aql_frame_num_; }

  // Both legacy HWS and stage 1 HWS use KMD to alloc use queue memory,
  // return false by default
  bool AllocUserQueueMemFromUMD(void) const { return false; }

  bool IsHwsEnabled(int engine) {
    return shared_dev_->DeviceInfo().IsHwsEnabled(engine);
  }

  void UpdatePageFence(uint64_t fence_value);

  D3DKMT_HANDLE PagingQueue() const { return page_queue_; }
  D3DKMT_HANDLE PagingFence() const { return page_syncobj_; }
  D3DKMT_HANDLE DeviceHandle() const { return device_; }
  LUID GetLuid() const { return adapter_luid_; }
  D3DKMT_HANDLE GetAdapter() const { return adapter_; }

  const thunk_proxy::DeviceInfo& DeviceInfo() const { return shared_dev_->DeviceInfo(); }

  ErrorCode CreateGpuMemory(const GpuMemoryCreateInfo &create_info, GpuMemory **gpu_mem, gpusize *gpu_va = nullptr);

private:
  bool CreateDevice(void);
  bool DestroyDevice(void);
  bool CreatePagingQueue(void);
  bool DestroyPagingQueue(void);
  void *Lock(D3DKMT_HANDLE handle);
  bool Unlock(D3DKMT_HANDLE handle);
  bool CreateContext(int engine, D3DKMT_HANDLE *handle);
  bool DestroyContext(D3DKMT_HANDLE handle);

  void SetPowerOptimization(bool restore);
  void InitCmdbufInfo(void);

  bool QuerySegmentInfo();
  bool GetSegmentId(D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE segment_type, uint32_t &segment_id);

  D3DKMT_HANDLE adapter_;
  LUID adapter_luid_;
  Device *shared_dev_ = nullptr;
  D3DKMT_HANDLE device_;

  D3DKMT_HANDLE page_queue_;
  D3DKMT_HANDLE page_syncobj_;
  uint64_t *page_fence_addr_;
  std::atomic<uint64_t> page_fence_value_;

  uint32_t cmdbuf_size_;
  uint32_t cmdbuf_aql_frame_size_;
  static const uint32_t cmdbuf_aql_frame_num_;
  uint32_t node_id_;
  std::vector<struct SegmentInfo> segment_infos_;
  //CmdUtil cmd_util;
};

NTSTATUS WDDMCreateDevices(std::vector<WDDMDevice *> &devices);

} // namespace thunk
} // namespace wsl

#endif
