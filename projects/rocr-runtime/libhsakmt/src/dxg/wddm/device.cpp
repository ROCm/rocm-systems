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

#include <cinttypes>
#include <bitset>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <linux/mman.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include "impl/wddm/status.h"
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "impl/wddm/queue.h"
#include "impl/wddm/event.h"
#include "util/os.h"

namespace wsl {
namespace thunk {

const uint32_t WDDMDevice::cmdbuf_aql_frame_num_ = 0x1000;

WDDMDevice::WDDMDevice(D3DKMT_HANDLE adapter, LUID adapter_luid, uint32_t node_id)
  : adapter_(adapter), adapter_luid_(adapter_luid), node_id_(node_id), init_status_(kDeviceSuccess) {
  memset(&device_info_, 0, sizeof(device_info_));

  NTSTATUS ret = ParseDeviceInfo();

  if (ret == STATUS_OBJECT_NAME_NOT_FOUND || ret == STATUS_REVISION_MISMATCH) {
    // Skip adapter
    // Registry info not found (adapter may not support AMD GPU),
    // Or GFX IP version is not supported.
    init_status_ = kDeviceSkipped;
    return;
  }
  if (ret != STATUS_SUCCESS) {
    init_status_ = kDeviceFailed;
    return;
  }

  CreateDevice();
  SetPowerOptimization(false);
  CreatePagingQueue();
  InitCmdbufInfo();
  QuerySegmentInfo();
}

WDDMDevice::~WDDMDevice() {
  if (init_status_ == kDeviceSuccess ) {
    DestroyPagingQueue();
    SetPowerOptimization(true);
    DestroyDevice();
  }

  DestroyDeviceInfo();
}

static NTSTATUS WDDMQueryAdapter(D3DKMT_HANDLE adapter, KMTQUERYADAPTERINFOTYPE type,
void *data, int size) {
  D3DKMT_QUERYADAPTERINFO args = {0};

  args.hAdapter = adapter;
  args.Type = type;
  args.pPrivateDriverData = data;
  args.PrivateDriverDataSize = size;

  return DXCORE_CALL(D3DKMTQueryAdapterInfo(&args));
}

bool WDDMDevice::QuerySegmentInfo()
{
  uint32_t segmentCount = 0;
  segment_infos_.clear();

  // Get the number of segments
  D3DKMT_QUERYSTATISTICS adapterQuery = {};
  adapterQuery.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
  adapterQuery.AdapterLuid = adapter_luid_;

  NTSTATUS ret = DXCORE_CALL(D3DKMTQueryStatistics(&adapterQuery));
  if (ret == STATUS_SUCCESS) {
    segmentCount = adapterQuery.QueryResult.AdapterInformation.NbSegments;
    pr_debug("Total Segments: %u\n", segmentCount);
  } else {
    pr_err("Failed to query adapter info\n");
    return false;
  }

  for (uint32_t i = 0; i < segmentCount; i++) {

    D3DKMT_QUERYSTATISTICS segQuery = {};
    segQuery.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
    segQuery.AdapterLuid = adapter_luid_;
    segQuery.QuerySegment.SegmentId = i;

    ret = DXCORE_CALL(D3DKMTQueryStatistics(&segQuery));
    if (ret != STATUS_SUCCESS) {
      pr_err("Failed to query segment %u info\n", i);
      return false;
    }

    auto& seg = segQuery.QueryResult.SegmentInformation;

    SegmentInfo info;
    info.segment_id = i;
    info.segment_type = seg.SegmentProperties.SegmentType;
    info.system_memory = seg.SegmentProperties.SystemMemory;
    info.aperture = seg.Aperture;
    info.commit_limit = seg.CommitLimit;

    segment_infos_.push_back(info);
  }

  return true;
}

bool WDDMDevice::GetSegmentId(D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE segment_type,
                              uint32_t &segment_id)
{
  for (const auto& seg_info : segment_infos_) {
    if (seg_info.segment_type == segment_type) {
      segment_id = seg_info.segment_id;
      return true;
    }
  }
  pr_err("Failed to get segment id for type %u\n", segment_type);
  return false;
}

/*Local heap(dedicated GPU memory) includes visiable heap and invisiable heap.
 *Non local heap refers to shared GPU memory and it is sytem memory.
 */
uint64_t WDDMDevice::VramAvail(void) {
  D3DKMT_QUERYSTATISTICS stats;
  NTSTATUS ret;
  uint64_t usedVis = 0;
  uint64_t usedInv = 0;
  uint64_t usedNonLocal = 0;
  uint32_t segmentId = 0;

  // wait fence complete
  uint64_t value = page_fence_value_.load();
  if(!CpuWait(&page_syncobj_, &value, 1, false))
    return HSA_STATUS_ERROR;

  if (IsDgpu()) {
    // local cpu-visible memory
    if(!GetSegmentId(D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE_MEMORY, segmentId))
      return HSA_STATUS_ERROR;

    memset(&stats, 0, sizeof(D3DKMT_QUERYSTATISTICS));
    stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
    stats.AdapterLuid = adapter_luid_;
    stats.QuerySegment.SegmentId = segmentId;
    ret = DXCORE_CALL(D3DKMTQueryStatistics(&stats));
    if (ret == 0)
      usedVis = stats.QueryResult.SegmentInformation.BytesResident;

    // local invisible memory
    if (device_info_.local_invisible_heap_size) {
      segmentId++;
      memset(&stats, 0, sizeof(D3DKMT_QUERYSTATISTICS));
      stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
      stats.AdapterLuid = adapter_luid_;
      stats.QuerySegment.SegmentId = 1;

      ret = DXCORE_CALL(D3DKMTQueryStatistics(&stats));
      if (ret == 0)
        usedInv = stats.QueryResult.SegmentInformation.BytesResident;
    }

    return LocalHeapSize() - usedVis - usedInv;
  } else {
    // APU - NonLocal memory
    if(!GetSegmentId(D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE_SYSMEM, segmentId))
      return HSA_STATUS_ERROR;

    memset(&stats, 0, sizeof(D3DKMT_QUERYSTATISTICS));
    stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
    stats.AdapterLuid = adapter_luid_;
    stats.QuerySegment.SegmentId = segmentId;
    ret = DXCORE_CALL(D3DKMTQueryStatistics(&stats));
    if (ret == 0)
      usedNonLocal = stats.QueryResult.SegmentInformation.BytesResident;

    return NonLocalHeapSize() - usedNonLocal;
  }
}

bool WDDMDevice::CreateDevice(void) {
  D3DKMT_CREATEDEVICE args = {0};
  args.hAdapter = adapter_;

  NTSTATUS ret = DXCORE_CALL(D3DKMTCreateDevice(&args));
  if (ret == STATUS_SUCCESS) {
    device_ = args.hDevice;
    return true;
  }

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::DestroyDevice(void) {
  D3DKMT_DESTROYDEVICE args = {0};
  args.hDevice = device_;

  NTSTATUS ret = DXCORE_CALL(D3DKMTDestroyDevice(&args));
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::CreatePagingQueue(void) {
  D3DKMT_CREATEPAGINGQUEUE args = {0};
  args.hDevice = device_;
  args.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;

  NTSTATUS ret = DXCORE_CALL(D3DKMTCreatePagingQueue(&args));
  if (ret == STATUS_SUCCESS) {
    page_queue_ = args.hPagingQueue;
    page_syncobj_ = args.hSyncObject;
    page_fence_addr_ = (uint64_t *)args.FenceValueCPUVirtualAddress;
    page_fence_value_ = 0;
    return true;
  }

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::DestroyPagingQueue(void) {
  D3DDDI_DESTROYPAGINGQUEUE args = {0};
  args.hPagingQueue = page_queue_;

  NTSTATUS ret = DXCORE_CALL(D3DKMTDestroyPagingQueue(&args));
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

void WDDMDevice::SetPowerOptimization(bool restore) {
  void *priv_data;
  int priv_size;

  priv_size = thunk_proxy::GetPowerOptPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  thunk_proxy::FillinPowerOptPrivData(priv_data, restore);

  D3DKMT_ESCAPE d3dkmt_escape;
  memset(&d3dkmt_escape, 0, sizeof(d3dkmt_escape));

  d3dkmt_escape.hAdapter              = adapter_;
  d3dkmt_escape.hDevice               = device_;
  d3dkmt_escape.hContext              = 0; //KMD only use device to identify the process
  d3dkmt_escape.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
  d3dkmt_escape.pPrivateDriverData    = priv_data;
  d3dkmt_escape.PrivateDriverDataSize = priv_size;
  d3dkmt_escape.Flags.HardwareAccess  = true;

  NTSTATUS status = DXCORE_CALL(D3DKMTEscape(&d3dkmt_escape));
  pr_debug("status %d, restore %d\n", status, restore);
  free(priv_data);
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
  if (create_info.dmabuf_fd != 0 && create_info.dmabuf_fd != INVALID_DMABUF_FD)
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
  args.hDevice = device_;
  args.hAllocation = handle;

  NTSTATUS ret = DXCORE_CALL(D3DKMTLock2(&args));
  if (ret == STATUS_SUCCESS)
    return args.pData;

  pr_err("fail %x\n", ret);
  return NULL;
}

bool WDDMDevice::Unlock(D3DKMT_HANDLE handle) {
  D3DKMT_UNLOCK2 args = {0};
  args.hDevice = device_;
  args.hAllocation = handle;

  NTSTATUS ret = DXCORE_CALL(D3DKMTUnlock2(&args));
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::CreateContext(int engine, D3DKMT_HANDLE *handle) {
  void *priv_data;
  int priv_size;

  int ordinal = EngineOrdinal(engine, &device_info_);
  if (ordinal < 0)
    return false;

  priv_size = thunk_proxy::GetContextPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
#if defined(__linux__)
  thunk_proxy::FillinContextPrivData(priv_data, SupportStateShadowingByCpFw());
#else
  thunk_proxy::FillinContextPrivData(
    priv_data,
    SupportStateShadowingByCpFw(),
    device_info_.compute_schedid);
#endif

  D3DKMT_CREATECONTEXTVIRTUAL args = {0};
  args.hDevice = device_;
  args.EngineAffinity = 1 << 0;
  args.NodeOrdinal = ordinal;
  args.pPrivateDriverData = priv_data;
  args.PrivateDriverDataSize = priv_size;
  args.ClientHint = D3DKMT_CLIENTHINT_OPENCL;

  if (IsHwsEnabled(engine))
    args.Flags.HwQueueSupported = 1;
  else
    args.Flags.DisableGpuTimeout = thunk_proxy::ShouldDisableGpuTimeout(engine, &device_info_);

  NTSTATUS ret = DXCORE_CALL(D3DKMTCreateContextVirtual(&args));
  if (ret == STATUS_SUCCESS) {
    *handle = args.hContext;
    free(priv_data);
    return true;
  }

  free(priv_data);

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::DestroyContext(D3DKMT_HANDLE handle) {
  D3DKMT_DESTROYCONTEXT args = {0};
  args.hContext = handle;

  NTSTATUS ret = DXCORE_CALL(D3DKMTDestroyContext(&args));
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::GpuWait(WDDMQueue *queue, const D3DKMT_HANDLE *syncobjs,
			 uint64_t *values, int count) {

  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU args = {0};
  args.hContext = queue->context;
  args.ObjectCount = count;
  args.ObjectHandleArray = syncobjs;
  args.MonitoredFenceValueArray = values;

  NTSTATUS ret = DXCORE_CALL(D3DKMTWaitForSynchronizationObjectFromGpu(&args));
  if (ret == STATUS_SUCCESS)
      return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::GpuSignal(D3DKMT_HANDLE context, const D3DKMT_HANDLE *syncobjs,
			   uint64_t *value, int count) {
  D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU args = {0};
  args.hContext = context;
  args.ObjectCount = count;
  args.ObjectHandleArray = syncobjs;
  args.MonitoredFenceValueArray = value;

  NTSTATUS ret = DXCORE_CALL(D3DKMTSignalSynchronizationObjectFromGpu(&args));
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::CpuWait(const D3DKMT_HANDLE *syncobjs, uint64_t *value,
			 int count, bool wait_any) {
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU args = {0};
  args.hDevice = device_;
  args.ObjectCount = count;
  args.ObjectHandleArray = syncobjs;
  args.FenceValueArray = value;
  args.Flags.WaitAny = wait_any;

  NTSTATUS ret = DXCORE_CALL(D3DKMTWaitForSynchronizationObjectFromCpu(&args));
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
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
  args.hDevice = device_;
  args.Info.Type = D3DDDI_MONITORED_FENCE;
  args.Info.MonitoredFence.EngineAffinity = 1 << 0;

  NTSTATUS ret = DXCORE_CALL(D3DKMTCreateSynchronizationObject2(&args));
  if (ret == STATUS_SUCCESS) {
    *handle = args.hSyncObject;
    *addr = (uint64_t *)args.Info.MonitoredFence.FenceValueCPUVirtualAddress;
    pr_debug("create syncobj cpu addr=%p gpu addr=%" PRIx64 "\n",
             args.Info.MonitoredFence.FenceValueCPUVirtualAddress,
             args.Info.MonitoredFence.FenceValueGPUVirtualAddress);

    return true;
  }

  pr_err("fail %x\n", ret);
  return false;
}

void WDDMDevice::DestroySyncobj(D3DKMT_HANDLE handle) {
  D3DKMT_DESTROYSYNCHRONIZATIONOBJECT args = {0};
  args.hSyncObject = handle;

  NTSTATUS ret = DXCORE_CALL(D3DKMTDestroySynchronizationObject(&args));
  if (ret != STATUS_SUCCESS)
    pr_err("fail %x\n", ret);
}

void WDDMDevice::InitCmdbufInfo(void) {
  if (device_info_.major == 9) {
    cmdbuf_aql_frame_size_ = 2 * sizeof(gfx9::AcquireMemTemplate);
  } else if (device_info_.major >= 10) {
    cmdbuf_aql_frame_size_ = 2 * sizeof(gfx10::AcquireMemTemplate);
  }

  if (device_info_.major >= 11) {
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

  cmdbuf_aql_frame_size_ = rocr::AlignUp(cmdbuf_aql_frame_size_, 0x10);

  cmdbuf_size_ = rocr::AlignUp(cmdbuf_aql_frame_num_ * cmdbuf_aql_frame_size_, 0x1000);
}

uint32_t WDDMDevice::LdsBlocks(const hsa_kernel_dispatch_packet_t *pkt) {
  static const uint32_t blk_sz = 512;
  uint32_t total_sz = pkt->group_segment_size;
  uint32_t blk_num = (total_sz + blk_sz - 1) / blk_sz;
  return blk_num;
}

NTSTATUS WDDMCreateDevices(std::vector<WDDMDevice *> &devices)
{
  bool supported = false;
  NTSTATUS ret = STATUS_SUCCESS;
  D3DKMT_ENUMADAPTERS3 args = {0};
  args.Filter.IncludeComputeOnly = true;
  ret = DXCORE_CALL(D3DKMTEnumAdapters3(&args));
  if (ret != STATUS_SUCCESS)
    return ret;

  if (!args.NumAdapters) {
    return STATUS_SUCCESS;
  }

  D3DKMT_ADAPTERINFO *info = new D3DKMT_ADAPTERINFO[args.NumAdapters];
  if (!info)
    return STATUS_NO_MEMORY;

  args.pAdapters = info;
  ret = DXCORE_CALL(D3DKMTEnumAdapters3(&args));
  if (ret != STATUS_SUCCESS)
    goto err_out0;

  for (int i = 0; i < args.NumAdapters; i++) {
    D3DKMT_QUERY_DEVICE_IDS query = {0};

    ret = WDDMQueryAdapter(info[i].hAdapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
			   &query, sizeof(query));
    if (ret != STATUS_SUCCESS)
      continue;

    if (query.DeviceIds.VendorID != 0x1002)
      continue;

    supported = thunk_proxy::QueryAdapterSupported(query.DeviceIds.DeviceID);

    if (supported) {
      auto device = new WDDMDevice(
        info[i].hAdapter, info[i].AdapterLuid, devices.size() + 1);
      if (!device)
        goto err_out1;

      // Check if device initialization succeeded
      if (device->InitStatus() != WDDMDevice::kDeviceSuccess) {
        if (device->InitStatus() == WDDMDevice::kDeviceSkipped) {
          delete device;
          continue;
        }
        // For other errors, fail
        pr_info("Failed to initialize device for adapter %d\n", i);
        delete device;
        goto err_out1;
      }
      pr_info("Adapter %d: device id 0x%04x supported\n",
              i, query.DeviceIds.DeviceID);
      devices.push_back(device);
    }
  }

  delete[] info;
  return STATUS_SUCCESS;

 err_out1:
  for (auto &device : devices)
    delete device;
 err_out0:
  delete[] info;
  return ret;
}

NTSTATUS WDDMDevice::ParseDeviceInfo() {
#if defined(__linux__)
  return (thunk_proxy::ParseAdapterInfo(adapter_, &device_info_)) ?
    STATUS_SUCCESS : STATUS_OBJECT_NAME_NOT_FOUND;
#else
  return thunk_proxy::ParseAdapterInfo(adapter_, &device_info_);
#endif
}

void WDDMDevice::DestroyDeviceInfo() {
  free(device_info_.adapter_info);
}

void WDDMDevice::GetClockCounters(uint64_t *gpu, uint64_t *cpu) {

  uint32_t engine = GetComputeEngine();
  int ordinal = EngineOrdinal(engine, &device_info_);

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

  NTSTATUS status = DXCORE_CALL(D3DKMTQueryClockCalibration(&args));
  if (status) {
    pr_debug("status %d \n", status);
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
  void *priv_data;
  int priv_size;

  priv_size = thunk_proxy::GetSubmitPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  thunk_proxy::FillinSubmitPrivData(priv_data, queue->queue, command_addr, command_size, false);

  D3DKMT_SUBMITCOMMAND args = {0};
  args.Commands = command_addr;
  args.CommandLength = command_size;
  args.BroadcastContextCount = 1;
  args.BroadcastContext[0] = queue->context;
  args.pPrivateDriverData = priv_data;
  args.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = DXCORE_CALL(D3DKMTSubmitCommand(&args));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    free(priv_data);
    return false;
  }

  free(priv_data);

  if (!GpuSignal(queue->context, &queue->syncobj, &fence_value, 1))
    return false;

  return true;
}

bool WDDMDevice::CreateHwQueue(WDDMQueue *queue) {
  void *priv_data;
  int priv_size;

  priv_size = thunk_proxy::GetHwQueuePrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  bool FwManagedGfxState = SupportStateShadowingByCpFw();
#if defined(__linux__)
  thunk_proxy::FillinHwQueuePrivData(priv_data, FwManagedGfxState, queue->prio);
#else
  auto queue_memory = static_cast<ComputeQueue*>(queue)->GetAmdQueueMemory();
  auto resource = queue_memory->KmtHandle();
  thunk_proxy::FillinHwQueuePrivData(priv_data, FwManagedGfxState, queue->prio, IsAqlSupported(),
      queue->cmdbuf_addr, queue->cmdbuf_size, reinterpret_cast<uintptr_t>(queue->ring_wptr),
      reinterpret_cast<uintptr_t>(queue->ring_rptr), resource);
#endif

  D3DKMT_CREATEHWQUEUE createHwQueue = {0};
  createHwQueue.hHwContext = queue->context;
  createHwQueue.Flags.DisableGpuTimeout = thunk_proxy::ShouldDisableGpuTimeout(queue->queue_engine, &device_info_);
  createHwQueue.pPrivateDriverData = priv_data;
  createHwQueue.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = DXCORE_CALL(D3DKMTCreateHwQueue(&createHwQueue));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    free(priv_data);
    return false;
  }

  free(priv_data);

  queue->queue = createHwQueue.hHwQueue;
  queue->syncobj = createHwQueue.hHwQueueProgressFence;
  queue->sync_addr = (uint64_t *)createHwQueue.HwQueueProgressFenceCPUVirtualAddress;

  return true;
}

bool WDDMDevice::DestroyHwQueue(WDDMQueue *queue) {
   D3DKMT_DESTROYHWQUEUE DestroyHwQueue = {
    .hHwQueue = queue->queue,
  };

  NTSTATUS ret = DXCORE_CALL(D3DKMTDestroyHwQueue(&DestroyHwQueue));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    return false;
  }

  return true;
}

bool WDDMDevice::SubmitToHwQueue(WDDMQueue *queue, uint64_t command_addr,
                                uint64_t command_size, uint64_t fence_value) {
  void *priv_data;
  int priv_size;

  priv_size = thunk_proxy::GetSubmitPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  thunk_proxy::FillinSubmitPrivData(priv_data, queue->queue, command_addr, command_size, true);

  D3DKMT_SUBMITCOMMANDTOHWQUEUE args = {0};
  args.hHwQueue = queue->queue;
  args.HwQueueProgressFenceId = fence_value;
  args.CommandBuffer = command_addr;
  args.CommandLength = command_size;
  args.pPrivateDriverData = priv_data;
  args.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = DXCORE_CALL(D3DKMTSubmitCommandToHwQueue(&args));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    free(priv_data);
    return false;
  }

  free(priv_data);

  return true;
}

// ================================================================================================
bool WDDMDevice::SubmitToAqlQueue(WDDMQueue* queue, uint64_t command_addr, uint64_t command_size,
                                  uint64_t fence_value) {
#if defined(WIN32)
  int priv_size = thunk_proxy::GetAqlSubmitPrivDataSize();
  void* priv_data = alloca(priv_size);
  memset(priv_data, 0, priv_size);
  thunk_proxy::FillinAqlSubmitPrivData(priv_data, fence_value);
  D3DKMT_SUBMITCOMMANDTOHWQUEUE args = {
      .hHwQueue = queue->queue,
      .HwQueueProgressFenceId = static_cast<ULONG>(fence_value + 1),
      .CommandBuffer = command_addr,
      .CommandLength = static_cast<UINT>(command_size),
      .PrivateDriverDataSize = static_cast<UINT>(priv_size),
      .pPrivateDriverData = priv_data};
  NTSTATUS ret = DXCORE_CALL(D3DKMTSubmitCommandToHwQueue(&args));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    return false;
  }
#endif
  return true;
}

// ================================================================================================
bool WDDMDevice::Escape(void* priv_data, uint32_t priv_size, bool hw_access) {
  D3DKMT_ESCAPE d3dkmt_escape = {.hAdapter = adapter_,
                                 .hDevice = device_,
                                 .Type = D3DKMT_ESCAPE_DRIVERPRIVATE,
                                 .Flags = {.HardwareAccess = hw_access},
                                 .pPrivateDriverData = priv_data,
                                 .PrivateDriverDataSize = priv_size,
                                 .hContext = 0};  // KMD only uses device to identify the process
  NTSTATUS status = DXCORE_CALL(D3DKMTEscape(&d3dkmt_escape));
  if (status != STATUS_SUCCESS) {
    pr_debug("Escape call failed\n");
    return false;
  }
  return true;
}

// ================================================================================================
uint32_t WDDMDevice::RegisterEvent(uint32_t type, HANDLE event_handle, uint64_t* mailbox) {
#if defined(WIN32)
  // Reset maibox locaiton to 0
  *mailbox = 0;
  // Start from 1, since 0 is the default state and can't be identified in KMD
  for (uint32_t event_id = 1; event_id < kNumberOfHsaEvents; event_id++) {
    // Check if the current slot is free and assing the mailbox
    if (!alloced_events_.test(event_id)) {
      // Fill private KMD data
      int priv_size = thunk_proxy::GetRegisterEventPrivDataSize();
      void* priv_data = alloca(priv_size);
      memset(priv_data, 0, priv_size);
      thunk_proxy::FillinRegisterEventPrivData(priv_data, reinterpret_cast<uint64_t>(event_handle),
                                               event_id);
      // Make the escape call to KMD to get the mailbox and assign event ID
      if (Escape(priv_data, priv_size, false)) {
        // Initialize the mailbox array if it's the first call
        if (base_mailbox_va_ == 0) {
          base_mailbox_va_ = thunk_proxy::GetRegisterEventMailbox(priv_data);
        }
        alloced_events_.set(event_id);
        *mailbox = base_mailbox_va_ + event_id * sizeof(uint32_t);
        return event_id | kAqlPayloadId;
      } else {
        pr_debug("Request HSA event failed\n");
        return 0;
      }
    }
  }
#endif
  return 0;
}

// ================================================================================================
bool WDDMDevice::UnregisterEvent(uint32_t event_id, HANDLE event_handle) {
#if defined(WIN32)
  // Find the actual event ID by masking the AQL payload bit
  event_id &= kAqlPayloadId - 1;
  if (alloced_events_.test(event_id)) {
    alloced_events_.reset(event_id);
    // Fill private KMD data
    int priv_size = thunk_proxy::GetUnregisterEventPrivDataSize();
    void* priv_data = alloca(priv_size);
    memset(priv_data, 0, priv_size);
    thunk_proxy::FillinUnregisterEventPrivData(priv_data, reinterpret_cast<uint64_t>(event_handle));
    // Make the escape call to KMD to remove event assignment
    if (!Escape(priv_data, priv_size, false)) {
      pr_debug("Unregister event failed\n");
      return false;
    }
  }
#endif
  return true;
}

// ================================================================================================
HSAKMT_STATUS WDDMDevice::WaitOnMultipleEvents(HsaEvent* events[], uint32_t num_elems,
                                               bool wait_all, uint32_t msec) {
#if defined(WIN32)
  HANDLE* event_handles_ = reinterpret_cast<HANDLE*>(_alloca(sizeof(HANDLE) * num_elems));
  for (uint32_t i = 0; i < num_elems; ++i) {
    event_handles_[i] = reinterpret_cast<Event*>(events[i])->GetHandle();
  }
  uint32_t time = 0;
  uint32_t kWaitTimeout = 6000;  // 6 seconds
  if (!dxg_runtime->disable_wait_timeout_ && (msec > kWaitTimeout)) {
    msec = kWaitTimeout;
  }
  auto wait_msec = (num_elems <= MAXIMUM_WAIT_OBJECTS) ? msec : 1;
  while (time < msec) {
    int32_t size_to_process = static_cast<int>(num_elems);
    // WaitForMultipleObjects can only handle MAXIMUM_WAIT_OBJECTS (64) events at a time.
    // For larger counts, loop through chunks with 1ms timeout per iteration for
    // responsiveness. Not efficient, but unavoidable given Windows API constraints.
    for (uint32_t i = 0; (i <= (num_elems / MAXIMUM_WAIT_OBJECTS)) && (size_to_process > 0); ++i) {
      auto events_limit = std::min(size_to_process, MAXIMUM_WAIT_OBJECTS);
      const DWORD ret_code = WaitForMultipleObjects(
          events_limit, &event_handles_[i * MAXIMUM_WAIT_OBJECTS], wait_all, wait_msec);
      if (ret_code >= WAIT_OBJECT_0 &&
          ret_code <= (WAIT_OBJECT_0 + events_limit - 1)) {
        return HSAKMT_STATUS_SUCCESS;
      } else if (ret_code == WAIT_TIMEOUT) {
        // Timeout occurred, continue to next chunk of events.
        time += wait_msec;
        if (time >= msec) {
          break;
        }
      } else {
        // Wait failed with an error.
        pr_err("WaitForMultipleObjects failed with code %d\n", ret_code);
        return HSAKMT_STATUS_WAIT_FAILURE;
      }
      size_to_process -= MAXIMUM_WAIT_OBJECTS;
    }
  }
#endif
  return HSAKMT_STATUS_WAIT_TIMEOUT;
}

} // namespace thunk
} // namespace wsl
