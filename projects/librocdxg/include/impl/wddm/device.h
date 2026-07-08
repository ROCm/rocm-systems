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

#include "shared/include/d3dkmt_types.h"
#include "shared/include/device.h"
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

class WDDMDevice {
public:
  static constexpr size_t GpuMemoryChunkSize = 2 * (1ULL << 30);   // 2 GB

  WDDMDevice(Device *shared_dev,
             D3DKMT_HANDLE adapter, uint32_t node_id);
  ~WDDMDevice();

  Device *SharedDevice() const { return shared_dev_; }

  int NodeId() const { return node_id_; }
  int Major() { return shared_dev_->Major(); }
  int Minor() { return shared_dev_->Minor(); }
  int Stepping() { return shared_dev_->Stepping(); }
  bool IsDgpu() { return shared_dev_->IsDgpu(); }
  const char *ProductName() { return shared_dev_->ProductName(); }
  uint64_t Uuid() { return shared_dev_->Uuid(); }
  uint32_t GfxFamily() { return shared_dev_->Family(); }
  uint32_t DeviceId() { return shared_dev_->DeviceId(); }
  uint32_t WavefrontSize() { return shared_dev_->WavefrontSize(); }
  uint32_t ComputeUnitCount() { return shared_dev_->ComputeUnitCount(); }
  uint32_t MaxEngineClockMhz() { return shared_dev_->MaxEngineClockMhz(); }
  uint32_t WatchPointsNum() { return shared_dev_->WatchPointsNum(); }
  uint32_t PciBusAddr() { return shared_dev_->PciBusAddr(); }

  uint32_t MemoryBusWidth() { return shared_dev_->MemoryBusWidth(); }
  uint32_t MaxMemoryClockMhz() { return shared_dev_->MaxMemoryClockMhz(); }
  uint32_t WavePerCu() { return shared_dev_->WavePerCu(); }
  uint32_t SimdPerCu() { return shared_dev_->SimdPerCu(); }
  uint32_t MaxScratchSlotsPerCu() { return shared_dev_->MaxScratchSlotsPerCu(); }
  uint32_t NumShaderEngine() { return shared_dev_->NumShaderEngine(); }
  uint32_t ShaderArrayPerShaderEngine() { return shared_dev_->ShaderArrayPerShaderEngine(); }
  uint32_t NumSdmaEngine() { return shared_dev_->NumSdmaEngines(); }
  uint32_t Domain() { return shared_dev_->Domain(); }
  uint32_t NumGws() { return shared_dev_->NumGws(); }
  uint32_t AsicRevision() { return shared_dev_->AsicRevision(); }
  uint64_t LocalHeapSize() { return shared_dev_->LocalVisibleHeapSize() + shared_dev_->LocalInvisibleHeapSize(); }
  uint64_t LocalVisibleHeapSize() { return shared_dev_->LocalVisibleHeapSize(); }
  uint64_t LocalInvisibleHeapSize() { return shared_dev_->LocalInvisibleHeapSize(); }
  uint64_t NonLocalHeapSize() { return shared_dev_->NonLocalHeapSize(); }
  uint64_t PrivateApertureBase() { return shared_dev_->PrivateApertureBase(); }
  uint64_t PrivateApertureSize() { return shared_dev_->PrivateApertureSize(); }
  uint64_t SharedApertureBase() { return shared_dev_->SharedApertureBase(); }
  uint64_t SharedApertureSize() { return shared_dev_->SharedApertureSize(); }
  uint32_t LdsSize() { return shared_dev_->LdsSize(); }
  uint64_t GPUCounterFrequency() { return shared_dev_->GpuCounterFrequency(); }
  uint32_t GetSwsQueueSize(void) const { return shared_dev_->UserQueueSize(); }
  uint32_t GetMecFwVersion() { return shared_dev_->MecFwVersion(); }
  uint32_t GetSdmaFwVersion() { return shared_dev_->SdmaFwVersion(); }
  uint32_t GetL1CacheSize() { return shared_dev_->L1CacheSize(); }
  uint32_t GetL2CacheSize() { return shared_dev_->L2CacheSize(); }
  uint32_t GetL3CacheSize() { return shared_dev_->L3CacheSize(); }
  uint32_t Gl2CacheLineSize() { return shared_dev_->Gl2CacheLineSize(); }
  bool SupportStateShadowingByCpFw(void) const { return shared_dev_->SupportStateShadowingByCpFw(); }
  bool SupportPlatformAtomic(void) const { return shared_dev_->SupportPlatformAtomic(); }
  uint32_t GetSdmaEngine(uint32_t idx) {
    assert(idx < NumSdmaEngine());
    return shared_dev_->SdmaEngine(idx);
  }
  uint32_t GetComputeEngine() { return shared_dev_->ComputeEngine(); }

  ErrorCode VramAvail(uint64_t *avail);

  void GetClockCounters(uint64_t *gpu, uint64_t *cpu);
  uint32_t GetNumCpQueues() { return shared_dev_->NumCpQueues(); }

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
    return shared_dev_->IsHwsEnabled(engine);
  }

  void UpdatePageFence(uint64_t fence_value);

  D3DKMT_HANDLE PagingQueue() const { return page_queue_; }
  D3DKMT_HANDLE PagingFence() const { return page_syncobj_; }
  D3DKMT_HANDLE DeviceHandle() const { return shared_dev_->DeviceHandle(); }
  LUID GetLuid() const { return shared_dev_->AdapterLuid(); }
  D3DKMT_HANDLE GetAdapter() const { return adapter_; }

  ErrorCode CreateGpuMemory(const GpuMemoryCreateInfo &create_info, GpuMemory **gpu_mem, gpusize *gpu_va = nullptr);

private:
  bool CreatePagingQueue(void);
  bool DestroyPagingQueue(void);
  void *Lock(D3DKMT_HANDLE handle);
  bool Unlock(D3DKMT_HANDLE handle);
  bool CreateContext(int engine, D3DKMT_HANDLE *handle);
  bool DestroyContext(D3DKMT_HANDLE handle);

  void SetPowerOptimization(bool restore);
  void InitCmdbufInfo(void);

  D3DKMT_HANDLE adapter_;
  Device *shared_dev_ = nullptr;

  D3DKMT_HANDLE page_queue_;
  D3DKMT_HANDLE page_syncobj_;
  uint64_t *page_fence_addr_;
  std::atomic<uint64_t> page_fence_value_;

  uint32_t cmdbuf_size_;
  uint32_t cmdbuf_aql_frame_size_;
  static const uint32_t cmdbuf_aql_frame_num_;
  uint32_t node_id_;
  //CmdUtil cmd_util;
};

NTSTATUS WDDMCreateDevices(std::vector<WDDMDevice *> &devices);

} // namespace thunk
} // namespace wsl

#endif
