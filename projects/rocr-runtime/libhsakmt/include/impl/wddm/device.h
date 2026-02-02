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

#include <cassert>

#if !defined(__linux__)
#define WIN32_NO_STATUS
#include <Windows.h>
#undef WIN32_NO_STATUS
#endif

#include <ntstatus.h>

#include <atomic>
#include <memory>
#include <vector>
#include <bitset>

#include "impl/wddm/types.h"
#include "impl/thunk_proxy/thunk_proxy.h"
#include "impl/wddm/va_mgr.h"
#include "impl/wddm/status.h"
#include "impl/wddm/types.h"
#include "impl/wddm/gpu_memory.h"
#include "impl/wddm/cmd_util.h"

namespace wsl {
namespace thunk {

//class Queue;
class WDDMQueue;

// WSL2 hyperv GPADL protocol limitation
#define MAX_USERPTR_BLOCK_SIZE 0xf0000000
#define START_NON_CANONICAL_ADDR (1ULL << 47)
#define END_NON_CANONICAL_ADDR (~0ULL - (1ULL << 47))
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
  enum device_init_result {
    kDeviceSuccess = 0,
    kDeviceSkipped = 1,
    kDeviceFailed = 2,
  };

  static constexpr size_t GpuMemoryChunkSize = 2 * (1ULL << 30);   // 2 GB
  static constexpr uint32_t kNumberOfHsaEvents = 1024;   //!< Note: may change in the future to 8K, KMD should define it
  static constexpr uint32_t kAqlPayloadId = 1 << 24;

  WDDMDevice(D3DKMT_HANDLE adapter, LUID adapter_luid, uint32_t node_id);
  ~WDDMDevice();

  int NodeId() const { return node_id_; }
  int Major() { return device_info_.major; }
  int Minor() { return device_info_.minor; }
  int Stepping() { return device_info_.stepping; }
  bool IsDgpu() { return device_info_.is_dgpu; }
  const char *ProductName() { return device_info_.product_name; }
  uint64_t Uuid() { return device_info_.uuid; }
  uint32_t GfxFamily() { return device_info_.family; }
  uint32_t DeviceId() { return device_info_.device_id; }
  uint32_t WavefrontSize() { return device_info_.wavefront_size; }
  uint32_t ComputeUnitCount() { return device_info_.compute_unit_count; }
  uint32_t MaxEngineClockMhz() { return device_info_.max_engine_clock_mhz; }
  uint32_t WatchPointsNum() { return device_info_.watch_points_num; }
  uint32_t PciBusAddr() { return device_info_.pci_bus_addr; }

  uint32_t MemoryBusWidth() { return device_info_.memory_bus_width; }
  uint32_t MaxMemoryClockMhz() { return device_info_.max_memory_clock_mhz; }
  uint32_t WavePerCu() { return device_info_.wave_per_cu; }
  uint32_t SimdPerCu() { return device_info_.simd_per_cu; }
  uint32_t MaxScratchSlotsPerCu() { return device_info_.max_scratch_slots_per_cu; }
  uint32_t NumShaderEngine() { return device_info_.num_shader_engine; }
  uint32_t ShaderArrayPerShaderEngine() { return device_info_.shader_array_per_shader_engine; }
  uint32_t NumSdmaEngine() { return device_info_.sdma_schedid.size(); }
  uint32_t Domain() { return device_info_.domain; }
  uint32_t NumGws() { return device_info_.num_gws; }
  uint32_t AsicRevision() { return device_info_.asic_revision; }
  uint64_t LocalHeapSize() { return device_info_.local_visible_heap_size + device_info_.local_invisible_heap_size; }
  uint64_t LocalVisibleHeapSize() { return device_info_.local_visible_heap_size; }
  uint64_t LocalInvisibleHeapSize() { return device_info_.local_invisible_heap_size; }
  uint64_t NonLocalHeapSize() { return device_info_.non_local_heap_size; }
  uint64_t PrivateApertureBase() { return device_info_.private_aperture_base; }
  uint64_t PrivateApertureSize() { return device_info_.private_aperture_size; }
  uint64_t SharedApertureBase() { return device_info_.shared_aperture_base; }
  uint64_t SharedApertureSize() { return device_info_.shared_aperture_size; }
  uint32_t LdsSize() { return device_info_.lds_size; }
  uint64_t GPUCounterFrequency() { return device_info_.gpu_counter_frequency; }
  uint32_t GetSwsQueueSize(void) const { return device_info_.user_queue_size; }
  uint32_t GetMecFwVersion() { return device_info_.mec_fw_version; }
  uint32_t GetSdmaFwVersion() { return device_info_.sdma_fw_version; }
  uint32_t GetL1CacheSize() { return device_info_.l1_cache_size; }
  uint32_t GetL2CacheSize() { return device_info_.l2_cache_size; }
  uint32_t GetL3CacheSize() { return device_info_.l3_cache_size; }
  uint32_t Gl2CacheLineSize() { return device_info_.gl2_cacheline_size; }
  bool SupportStateShadowingByCpFw(void) const { return device_info_.state_shadowing_by_cpfw; }
  bool SupportPlatformAtomic(void) const { return device_info_.platform_atomic_support; }
  bool IsAqlSupported() const { return device_info_.hwsInfo.hwsMask.aql_queue != 0; }

  uint32_t GetSdmaEngine(uint32_t idx) {
    assert(idx < NumSdmaEngine());
    return device_info_.sdma_schedid[idx];
  }
  uint32_t GetComputeEngine() { return device_info_.compute_schedid; }

  uint64_t VramAvail();

  void GetClockCounters(uint64_t *gpu, uint64_t *cpu);
  uint32_t GetNumCpQueues() { return device_info_.num_cp_queues; }
  uint32_t NumXcc() const { return device_info_.num_xcc; }

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
  bool SubmitToAqlQueue(WDDMQueue* queue, uint64_t command_addr, uint64_t command_size,
                        uint64_t SubmitToAqlQueue);

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

  bool AllocUserQueueMemFromUMD(void) const {
    // stage 1 HWS queue memory is allocated by KMD.
    // GFX10 uses legacy HWS. UMD allocates user queue.
    return (device_info_.major == 10);
  }

  bool IsHwsEnabled(int engine) {
    return thunk_proxy::GetHwsEnabled(engine, &device_info_);
  }

  void UpdatePageFence(uint64_t fence_value);

  D3DKMT_HANDLE PagingQueue() const { return page_queue_; }
  D3DKMT_HANDLE PagingFence() const { return page_syncobj_; }
  D3DKMT_HANDLE DeviceHandle() const { return device_; }
  LUID GetLuid() const { return adapter_luid_; }
  D3DKMT_HANDLE GetAdapter() const { return adapter_; }

  const thunk_proxy::DeviceInfo& DeviceInfo() const { return device_info_; }

  ErrorCode CreateGpuMemory(const GpuMemoryCreateInfo &create_info, GpuMemory **gpu_mem, gpusize *gpu_va = nullptr);
  uint32_t RegisterEvent(uint32_t type, HANDLE event_handle, uint64_t* mailbox);
  bool UnregisterEvent(uint32_t event_id, HANDLE event_handle);
  static HSAKMT_STATUS WaitOnMultipleEvents(HsaEvent* events[], uint32_t num_elems, bool wait_all, uint32_t msec);
  device_init_result InitStatus() const { return init_status_; }
  uint32_t GbAddrConfig() const { return device_info_.gb_addr_config; }

private:
  bool Escape(void* priv_data, uint32_t priv_size, bool hw_access);
  NTSTATUS ParseDeviceInfo(void);
  void DestroyDeviceInfo(void);
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
  D3DKMT_HANDLE device_;

  D3DKMT_HANDLE page_queue_;
  D3DKMT_HANDLE page_syncobj_;
  uint64_t *page_fence_addr_;
  std::atomic<uint64_t> page_fence_value_;

  uint32_t cmdbuf_size_;
  uint32_t cmdbuf_aql_frame_size_;
  static const uint32_t cmdbuf_aql_frame_num_;
  uint32_t node_id_;
  // device info
  thunk_proxy::DeviceInfo device_info_;
  std::vector<struct SegmentInfo> segment_infos_;
  //CmdUtil cmd_util;
  device_init_result init_status_;

  // GPU events fields
  uint64_t base_mailbox_va_ = 0;  //!< GPU VA returned by KMD for all mailboxes
  std::bitset<kNumberOfHsaEvents> alloced_events_;  //!< The bit map of allocated events
};

NTSTATUS WDDMCreateDevices(std::vector<WDDMDevice *> &devices);

} // namespace thunk
} // namespace wsl

