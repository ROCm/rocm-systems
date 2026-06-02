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

#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <linux/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "shared/include/status.h"
#include "shared/include/d3dkmt_types.h"
#include "shared/include/platform.h"
#include "shared/include/device.h"
#include "shared/include/lda_chain.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include "shared/include/thunks.h"
#include "impl/wddm/device.h"
#include "impl/wddm/queue.h"
#include "shared/include/utils.h"

namespace wsl {
namespace thunk {

namespace dx = wsl::thunk::d3dthunk;

const uint32_t WDDMDevice::cmdbuf_aql_frame_num_ = 0x1000;

WDDMDevice::WDDMDevice(Device *shared_dev,
                       D3DKMT_HANDLE adapter, uint32_t node_id)
  : adapter_(adapter), shared_dev_(shared_dev),
    node_id_(node_id) {
  SetPowerOptimization(false);
  CreatePagingQueue();
  InitCmdbufInfo();

  unsigned ver = static_cast<unsigned>(Platform::instance().WddmVersion());
  if (ver)
    pr_info("WDDM version %u.%u\n", ver / 1000, (ver % 1000) / 100);
  else
    pr_info("WDDM version: unknown\n");
}

WDDMDevice::~WDDMDevice() {
  DestroyPagingQueue();
  SetPowerOptimization(true);
}

/*Local heap(dedicated GPU memory) includes visiable heap and invisiable heap.
 *Non local heap refers to shared GPU memory and it is sytem memory.
 */
ErrorCode WDDMDevice::VramAvail(uint64_t *avail) {
  if (!avail)
    return ErrorCode::InvalidPointer;

  *avail = 0;

  // wait fence complete
  uint64_t value = page_fence_value_.load();
  if (!CpuWait(&page_syncobj_, &value, 1, false))
    return ErrorCode::Unknown;

  uint64_t used_local = 0;
  ErrorCode ret =
      shared_dev_->QueryVramSegmentUsage(VramSegmentKind::kLocal, &used_local);
  if (ret != ErrorCode::Success)
    return ret;

  if (IsDgpu()) {
    const uint64_t total = LocalHeapSize();
    *avail = used_local >= total ? 0 : total - used_local;
    return ErrorCode::Success;
  }

  uint64_t used_non_local = 0;
  ret = shared_dev_->QueryVramSegmentUsage(VramSegmentKind::kNonLocal,
                                           &used_non_local);
  if (ret != ErrorCode::Success)
    return ret;

  const uint64_t total = LocalHeapSize() + NonLocalHeapSize();
  const uint64_t used = used_local + used_non_local;
  *avail = used >= total ? 0 : total - used;
  return ErrorCode::Success;
}

bool WDDMDevice::CreatePagingQueue(void) {
  D3DKMT_CREATEPAGINGQUEUE args = {0};
  args.hDevice = DeviceHandle();
  args.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;

  ErrorCode ret = dx::CreatePagingQueue(&args);
  if (ret == ErrorCode::Success) {
    page_queue_ = args.hPagingQueue;
    page_syncobj_ = args.hSyncObject;
    page_fence_addr_ = (uint64_t *)args.FenceValueCPUVirtualAddress;
    page_fence_value_ = 0;
    return true;
  }

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::DestroyPagingQueue(void) {
  D3DDDI_DESTROYPAGINGQUEUE args = {0};
  args.hPagingQueue = page_queue_;

  ErrorCode ret = dx::DestroyPagingQueue(&args);
  if (ret == ErrorCode::Success)
    return true;

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

void WDDMDevice::SetPowerOptimization(bool restore) {
  auto priv = thunk_proxy::MakePowerOptPrivData(restore);

  D3DKMT_ESCAPE d3dkmt_escape;
  memset(&d3dkmt_escape, 0, sizeof(d3dkmt_escape));

  d3dkmt_escape.hAdapter              = adapter_;
  d3dkmt_escape.hDevice               = DeviceHandle();
  d3dkmt_escape.hContext              = 0; //KMD only use device to identify the process
  d3dkmt_escape.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
  d3dkmt_escape.pPrivateDriverData    = priv.data();
  d3dkmt_escape.PrivateDriverDataSize = priv.size();
  d3dkmt_escape.Flags.HardwareAccess  = true;

  ErrorCode status = dx::Escape(adapter_, DeviceHandle(), &d3dkmt_escape);
  pr_debug("status %d, restore %d\n", static_cast<int>(status), restore);
}

void WDDMDevice::UpdatePageFence(uint64_t fence_value) {
  uint64_t current = page_fence_value_.load();

  // atomically set fence value when target is bigger than current one
  do {
    if (current >= fence_value)
      break;
  } while (!page_fence_value_.compare_exchange_weak(current, fence_value));
}

ErrorCode WDDMDevice::CreateGpuMemory(const GpuMemoryCreateInfo &create_info,
                                        GpuMemory **gpu_mem, gpusize *gpu_va) {
  ErrorCode ret;

  *gpu_mem = nullptr;
  auto mem = new GpuMemory(this);
  if (create_info.dmabuf_fd > 0)
    ret = mem->ImportPhysicalHandle(create_info, gpu_va);
  else 
    ret = mem->Init(create_info);
  if (ret == ErrorCode::Success)
    *gpu_mem = mem;
  else
    delete mem;

  return ret;
}

void *WDDMDevice::Lock(D3DKMT_HANDLE handle) {
  D3DKMT_LOCK2 args = {0};
  args.hDevice = DeviceHandle();
  args.hAllocation = handle;

  ErrorCode ret = dx::Lock2(&args);
  if (ret == ErrorCode::Success)
    return args.pData;

  pr_err("fail %d\n", static_cast<int>(ret));
  return NULL;
}

bool WDDMDevice::Unlock(D3DKMT_HANDLE handle) {
  D3DKMT_UNLOCK2 args = {0};
  args.hDevice = DeviceHandle();
  args.hAllocation = handle;

  ErrorCode ret = dx::Unlock2(&args);
  if (ret == ErrorCode::Success)
    return true;

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::CreateContext(int engine, D3DKMT_HANDLE *handle) {
  int ordinal = shared_dev_->EngineOrdinal(engine);
  if (ordinal < 0)
    return false;

  auto priv = thunk_proxy::MakeContextPrivData(SupportStateShadowingByCpFw());

  D3DKMT_CREATECONTEXTVIRTUAL args = {0};
  args.hDevice = DeviceHandle();
  args.EngineAffinity = 1 << 0;
  args.NodeOrdinal = ordinal;
  args.pPrivateDriverData = priv.data();
  args.PrivateDriverDataSize = priv.size();
  args.ClientHint = D3DKMT_CLIENTHINT_OPENCL;

  if (IsHwsEnabled(engine))
    args.Flags.HwQueueSupported = 1;
  else
    args.Flags.DisableGpuTimeout = shared_dev_->IsGpuTimeoutDisabled(engine);

  ErrorCode ret = dx::CreateContextVirtual(&args);
  if (ret == ErrorCode::Success) {
    *handle = args.hContext;
    return true;
  }

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::DestroyContext(D3DKMT_HANDLE handle) {
  D3DKMT_DESTROYCONTEXT args = {0};
  args.hContext = handle;

  ErrorCode ret = dx::DestroyContext(&args);
  if (ret == ErrorCode::Success)
    return true;

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::GpuWait(WDDMQueue *queue, const D3DKMT_HANDLE *syncobjs,
			 uint64_t *values, int count) {

  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU args = {0};
  args.hContext = queue->context;
  args.ObjectCount = count;
  args.ObjectHandleArray = syncobjs;
  args.MonitoredFenceValueArray = values;

  ErrorCode ret = dx::WaitForSynchronizationObjectFromGpu(&args);
  if (ret == ErrorCode::Success)
      return true;

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::GpuSignal(D3DKMT_HANDLE context, const D3DKMT_HANDLE *syncobjs,
			   uint64_t *value, int count) {
  D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU args = {0};
  args.hContext = context;
  args.ObjectCount = count;
  args.ObjectHandleArray = syncobjs;
  args.MonitoredFenceValueArray = value;

  ErrorCode ret = dx::SignalSynchronizationObjectFromGpu(&args);
  if (ret == ErrorCode::Success)
    return true;

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::CpuWait(const D3DKMT_HANDLE *syncobjs, uint64_t *value,
			 int count, bool wait_any) {
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU args = {0};
  args.hDevice = DeviceHandle();
  args.ObjectCount = count;
  args.ObjectHandleArray = syncobjs;
  args.FenceValueArray = value;
  args.Flags.WaitAny = wait_any;

  ErrorCode ret = dx::WaitForSynchronizationObjectFromCpu(&args);
  if (ret == ErrorCode::Success)
    return true;

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

bool WDDMDevice::WaitOnPagingFenceFromCpu() {
  uint64_t page_fence_value = 0;

  page_fence_value = page_fence_value_.load();
  if (CpuWait(&page_syncobj_, &page_fence_value, 1, false))
    return true;

  return false;
}

bool WDDMDevice::CreateSyncobj(D3DKMT_HANDLE *handle, uint64_t **addr) {
  D3DKMT_CREATESYNCHRONIZATIONOBJECT2 args = {0};
  args.hDevice = DeviceHandle();
  args.Info.Type = D3DDDI_MONITORED_FENCE;
  args.Info.MonitoredFence.EngineAffinity = 1 << 0;

  ErrorCode ret = dx::CreateSynchronizationObject2(&args);
  if (ret == ErrorCode::Success) {
    *handle = args.hSyncObject;
    *addr = (uint64_t *)args.Info.MonitoredFence.FenceValueCPUVirtualAddress;
    pr_debug("create syncobj cpu addr=%p gpu addr=%" PRIx64 "\n",
             args.Info.MonitoredFence.FenceValueCPUVirtualAddress,
             args.Info.MonitoredFence.FenceValueGPUVirtualAddress);

    return true;
  }

  pr_err("fail %d\n", static_cast<int>(ret));
  return false;
}

void WDDMDevice::DestroySyncobj(D3DKMT_HANDLE handle) {
  D3DKMT_DESTROYSYNCHRONIZATIONOBJECT args = {0};
  args.hSyncObject = handle;

  ErrorCode ret = dx::DestroySynchronizationObject(&args);
  if (ret != ErrorCode::Success)
    pr_err("fail %d\n", static_cast<int>(ret));
}

void WDDMDevice::InitCmdbufInfo(void) {
  if (Major() == 9) {
    cmdbuf_aql_frame_size_ = 2 * sizeof(gfx9::AcquireMemTemplate);
  } else if (Major() >= 10) {
    cmdbuf_aql_frame_size_ = 2 * sizeof(gfx10::AcquireMemTemplate);
  }

  if (Major() >= 11) {
    cmdbuf_aql_frame_size_ += sizeof(SetScratchTemplate);
    cmdbuf_aql_frame_size_ += sizeof(DispatchProgramResourceRegs); // BuildComputeShaderParams
  }

  cmdbuf_aql_frame_size_ +=
    sizeof(PM4MEC_COPY_DATA) * 2 +
    sizeof(BarrierTemplate) * 2 +
    sizeof(DispatchTemplate) +
    sizeof(AtomicTemplate) * 2;

  // Add safety margin to account for alignment and future additions
  cmdbuf_aql_frame_size_ += 128;

  cmdbuf_aql_frame_size_ = AlignUp(cmdbuf_aql_frame_size_, 0x10);

  cmdbuf_size_ = AlignUp(cmdbuf_aql_frame_num_ * cmdbuf_aql_frame_size_, 0x1000);
}

uint32_t WDDMDevice::LdsBlocks(const hsa_kernel_dispatch_packet_t *pkt) {
  static const uint32_t blk_sz = 512;
  uint32_t total_sz = pkt->group_segment_size;
  uint32_t blk_num = (total_sz + blk_sz - 1) / blk_sz;
  return blk_num;
}

NTSTATUS WDDMCreateDevices(std::vector<WDDMDevice *> &devices)
{
  auto &platform = Platform::instance();
  std::vector<Device *> shared_devices;
  ErrorCode code = platform.EnumerateDevices(shared_devices);
  if (code != ErrorCode::Success && shared_devices.empty())
    return STATUS_SUCCESS;

  for (auto *sdev : shared_devices) {
    auto *chain = sdev->GetLdaChain();
    D3DKMT_HANDLE adapter = chain->AdapterHandle();

    auto device = new WDDMDevice(sdev, adapter, devices.size() + 1);
    if (!device)
      continue;
    devices.push_back(device);
  }

  return STATUS_SUCCESS;
}


void WDDMDevice::GetClockCounters(uint64_t *gpu, uint64_t *cpu) {

  uint32_t engine = GetComputeEngine();
  int ordinal = shared_dev_->EngineOrdinal(engine);

  D3DKMT_QUERYCLOCKCALIBRATION args = {0};

 /* LDA(Linked Display Adapter)
  * In the LDA design multiple physical GPUs are linked together to be controlled
  * as a single object from the point of view of power manager, GPU scheduler and
  * GPU memory manager. The physical GPUs are represented by a signal logical adapter
  * object. There is a single DXGADAPTER objects, a single KMD adapter object.
  *
  * Set PhysicalAdapterIndex to 0 by default with None LDA mode.
  */
  args.hAdapter = adapter_;
  args.NodeOrdinal = ordinal;
  args.PhysicalAdapterIndex = 0;

  ErrorCode status = dx::QueryClockCalibration(&args);
  if (status != ErrorCode::Success) {
    pr_debug("status %d \n", static_cast<int>(status));
  } else {
    if (gpu)
      *gpu = args.ClockData.GpuClockCounter;

    if (cpu)
      *cpu = args.ClockData.CpuClockCounter;
  }
}

bool WDDMDevice::CreateQueue(WDDMQueue *queue) {
  if (!CreateContext(queue->queue_engine, &queue->context))
    return false;

  GpuMemory *gpu_mem = nullptr;
  if (queue->cmdbuf_addr == 0) {
    GpuMemoryCreateInfo create_info{};
    create_info.size = queue->cmdbuf_size;
    create_info.domain = thunk_proxy::kSystem;

    auto code = CreateGpuMemory(create_info, &gpu_mem);
    if (code != ErrorCode::Success)
        goto err_out0;

    queue->cmdbuf = gpu_mem->GetGpuMemoryHandle();
    queue->cmdbuf_addr = gpu_mem->GpuAddress();
  }

  if (queue->Init())
     goto err_out1;

  return true;

err_out1:
  delete gpu_mem;
err_out0:
  DestroyContext(queue->context);

  return false;
}

void WDDMDevice::DestroyQueue(WDDMQueue *queue) {

  queue->Fini();

  auto cmdbuf_mem = GpuMemory::Convert(queue->cmdbuf);
  delete cmdbuf_mem;

  DestroyContext(queue->context);
}

bool WDDMDevice::SubmitToSwQueue(WDDMQueue *queue, uint64_t command_addr,
                                uint64_t command_size, uint64_t fence_value) {
  auto priv = thunk_proxy::MakeSubmitPrivData(queue->queue, command_addr, command_size, false);

  D3DKMT_SUBMITCOMMAND args = {0};
  args.Commands = command_addr;
  args.CommandLength = command_size;
  args.BroadcastContextCount = 1;
  args.BroadcastContext[0] = queue->context;
  args.pPrivateDriverData = priv.data();
  args.PrivateDriverDataSize = priv.size();

  ErrorCode ret = dx::SubmitCommand(&args);
  if (ret != ErrorCode::Success) {
    pr_err("fail %d\n", static_cast<int>(ret));
    return false;
  }

  if (!GpuSignal(queue->context, &queue->syncobj, &fence_value, 1))
    return false;

  return true;
}

bool WDDMDevice::CreateHwQueue(WDDMQueue *queue) {
  auto priv = thunk_proxy::MakeHwQueuePrivData(SupportStateShadowingByCpFw(), queue->prio);

  D3DKMT_CREATEHWQUEUE createHwQueue = {0};
  createHwQueue.hHwContext = queue->context;
  createHwQueue.Flags.DisableGpuTimeout =
      shared_dev_->IsGpuTimeoutDisabled(queue->queue_engine);
  createHwQueue.pPrivateDriverData = priv.data();
  createHwQueue.PrivateDriverDataSize = priv.size();

  ErrorCode ret = dx::CreateHwQueue(&createHwQueue);
  if (ret != ErrorCode::Success) {
    pr_err("fail %d\n", static_cast<int>(ret));
    return false;
  }

  queue->queue = createHwQueue.hHwQueue;
  queue->syncobj = createHwQueue.hHwQueueProgressFence;
  queue->sync_addr = (uint64_t *)createHwQueue.HwQueueProgressFenceCPUVirtualAddress;

  return true;
}

bool WDDMDevice::DestroyHwQueue(WDDMQueue *queue) {
   D3DKMT_DESTROYHWQUEUE DestroyHwQueue = {
    .hHwQueue = queue->queue,
  };

  ErrorCode ret = dx::DestroyHwQueue(&DestroyHwQueue);
  if (ret != ErrorCode::Success) {
    pr_err("fail %d\n", static_cast<int>(ret));
    return false;
  }

  return true;
}

bool WDDMDevice::SubmitToHwQueue(WDDMQueue *queue, uint64_t command_addr,
                                uint64_t command_size, uint64_t fence_value) {
  auto priv = thunk_proxy::MakeSubmitPrivData(queue->queue, command_addr, command_size, true);

  D3DKMT_SUBMITCOMMANDTOHWQUEUE args = {0};
  args.hHwQueue = queue->queue;
  args.HwQueueProgressFenceId = fence_value;
  args.CommandBuffer = command_addr;
  args.CommandLength = command_size;
  args.pPrivateDriverData = priv.data();
  args.PrivateDriverDataSize = priv.size();

  ErrorCode ret = dx::SubmitCommandToHwQueue(&args);
  if (ret != ErrorCode::Success) {
    pr_err("fail %d\n", static_cast<int>(ret));
    return false;
  }

  return true;
}

} // namespace thunk
} // namespace wsl
