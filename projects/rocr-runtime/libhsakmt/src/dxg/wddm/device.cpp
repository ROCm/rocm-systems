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
#include "impl/wddm/thunks.h"
#include "impl/wddm/vram_budget.h"
#include "util/os.h"

namespace wsl {
namespace thunk {

const uint32_t WDDMDevice::cmdbuf_aql_frame_num_ = 0x1000;

WDDMDevice::WDDMDevice(D3DKMT_HANDLE adapter, LUID adapter_luid, uint32_t node_id)
  : adapter_(adapter), adapter_luid_(adapter_luid), node_id_(node_id), init_status_(kDeviceSuccess) {
  memset(&device_info_, 0, sizeof(device_info_));

  NTSTATUS ret = ParseDeviceInfo();
  pr_rocr_info("kmd_version:%" PRIu32 "\n", device_info_.kmd_version);
  device_info_.hwsInfo.hwsMask.aql_queue &= !dxg_runtime->use_pm4_;
  pr_rocr_info("hwsInfo: aql_queue=%d computeHwsEnabled=%d use_pm4_override=%" PRIu64 "\n",
           device_info_.hwsInfo.hwsMask.aql_queue,
           device_info_.hwsInfo.hwsMask.computeHwsEnabled,
           (uint64_t)dxg_runtime->use_pm4_);

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

  if (device_info_.max_scratch_slots_per_cu == 0)
    device_info_.max_scratch_slots_per_cu = 32;

  unsigned ver = static_cast<unsigned>(dxg_runtime->wddm_version);
  if (ver)
    pr_rocr_info("WDDM version %u.%u\n", ver / 1000, (ver % 1000) / 100);
  else
    pr_rocr_info("WDDM version: unknown\n");

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
    info.is_aperture = seg.Aperture;
    info.is_system_memory = seg.SegmentProperties.SystemMemory;

    if (seg.Aperture) {
      info.kind = SegmentKind::kAperture;
    } else {
      info.kind = seg.SegmentProperties.SystemMemory
                      ? SegmentKind::kSystemMemory
                      : SegmentKind::kLocalMemory;
    }

    segment_infos_.push_back(info);
  }

  return true;
}

bool WDDMDevice::FindSegmentId(SegmentKind segment_kind, uint32_t* segment_id)
{
  for (const auto& seg_info : segment_infos_) {
    if (seg_info.kind == segment_kind) {
      *segment_id = seg_info.segment_id;
      return true;
    }
  }

  return false;
}

hsa_status_t WDDMDevice::QuerySegmentBytesResident(
    uint32_t segment_id, uint64_t* bytes_resident) const {
  D3DKMT_QUERYSTATISTICS stats = {};
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
  stats.AdapterLuid = adapter_luid_;
  stats.QuerySegment.SegmentId = segment_id;

  NTSTATUS ret = DXCORE_CALL(D3DKMTQueryStatistics(&stats));
  if (ret != STATUS_SUCCESS) {
    *bytes_resident = 0;
    return HSA_STATUS_ERROR;
  }

  *bytes_resident = stats.QueryResult.SegmentInformation.BytesResident;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMDevice::QuerySegmentGroupUsage(
    uint32_t segment_group, uint64_t* bytes_allocated) const {
  *bytes_allocated = 0;

  if (dxg_runtime->wddm_version < KMT_DRIVERVERSION_WDDM_3_1)
    return HSA_STATUS_ERROR;

  D3DKMT_QUERYSTATISTICS stats = {};
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT_GROUP_USAGE;
  stats.AdapterLuid = adapter_luid_;
  stats.QuerySegmentGroupUsage.PhysicalAdapterIndex = 0;
  stats.QuerySegmentGroupUsage.SegmentGroup =
      static_cast<UINT16>(segment_group);

  NTSTATUS ret = DXCORE_CALL(D3DKMTQueryStatistics(&stats));
  if (ret != STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  *bytes_allocated =
      stats.QueryResult.SegmentGroupUsageInformation.AllocatedBytes;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMDevice::QueryLocalVramUsage(uint64_t* usage_bytes) {
  *usage_bytes = 0;

  if (dxg_runtime->wddm_version >= KMT_DRIVERVERSION_WDDM_3_1 &&
      QuerySegmentGroupUsage(D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL,
                             usage_bytes) == HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;

  uint32_t visible_segment_id = 0;
  if (!FindSegmentId(SegmentKind::kLocalMemory, &visible_segment_id))
    return HSA_STATUS_ERROR;

  hsa_status_t ret =
      QuerySegmentBytesResident(visible_segment_id, usage_bytes);
  if (ret != HSA_STATUS_SUCCESS)
    return ret;

  if (!LocalInvisibleHeapSize())
    return HSA_STATUS_SUCCESS;

  uint32_t invisible_segment_id = 0;
  bool found_invisible = false;
  for (const auto& seg_info : segment_infos_) {
    if (seg_info.kind == SegmentKind::kLocalMemory &&
        seg_info.segment_id > visible_segment_id) {
      invisible_segment_id = seg_info.segment_id;
      found_invisible = true;
      break;
    }
  }

  if (!found_invisible)
    return HSA_STATUS_ERROR;

  uint64_t invisible_usage = 0;
  ret = QuerySegmentBytesResident(invisible_segment_id, &invisible_usage);
  if (ret != HSA_STATUS_SUCCESS)
    return ret;

  *usage_bytes += invisible_usage;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WDDMDevice::QueryNonLocalVramUsage(
    uint64_t* usage_bytes) const {
  *usage_bytes = 0;

  if (dxg_runtime->wddm_version >= KMT_DRIVERVERSION_WDDM_3_1 &&
      QuerySegmentGroupUsage(D3DKMT_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                             usage_bytes) == HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;

  bool found_segment = false;
  for (const auto& seg_info : segment_infos_) {
    if (!seg_info.is_aperture || !seg_info.is_system_memory)
      continue;

    found_segment = true;
    uint64_t segment_usage = 0;
    hsa_status_t ret =
        QuerySegmentBytesResident(seg_info.segment_id, &segment_usage);
    if (ret != HSA_STATUS_SUCCESS)
      return ret;
    *usage_bytes += segment_usage;
  }

  return found_segment ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

uint64_t WDDMDevice::VramTotal() {
  uint64_t total = LocalHeapSize();
  if (!IsDgpu())
    total += NonLocalHeapSize();
  return total;
}

hsa_status_t WDDMDevice::QueryVramUsage(uint64_t* usage_bytes) {
  hsa_status_t ret = QueryLocalVramUsage(usage_bytes);
  if (ret != HSA_STATUS_SUCCESS)
    return ret;

  if (IsDgpu())
    return HSA_STATUS_SUCCESS;

  uint64_t used_non_local = 0;
  ret = QueryNonLocalVramUsage(&used_non_local);
  if (ret != HSA_STATUS_SUCCESS)
    return ret;

  *usage_bytes += used_non_local;
  return HSA_STATUS_SUCCESS;
}

/*Local heap(dedicated GPU memory) includes visible heap and invisible heap.
 *Non local heap refers to shared GPU memory and it is system memory.
 */
hsa_status_t WDDMDevice::VramAvail(uint64_t* available_bytes) {
  if (!available_bytes)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  *available_bytes = 0;

  // wait fence complete
  uint64_t value = page_fence_value_.load();
  if (!CpuWait(&page_syncobj_, &value, 1, false))
    return HSA_STATUS_ERROR;

  if (dxg_runtime->wddm_version != 0 && dxg_runtime->wddm_version < KMT_DRIVERVERSION_WDDM_3_1 &&
      IsDgpu() && d3dthunk::QueryVideoMemoryInfoAvailable()) {
    D3DKMT_QUERYVIDEOMEMORYINFO info = {};
    info.hAdapter = adapter_;
    info.MemorySegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL;
    info.PhysicalAdapterIndex = 0;

    if (d3dthunk::QueryVideoMemoryInfo(&info) == ErrorCode::Success) {
      *available_bytes = AvailableVramBudget(info.Budget, info.CurrentUsage, VramTotal());
      return HSA_STATUS_SUCCESS;
    }
  }

  uint64_t used = 0;
  hsa_status_t ret = QueryVramUsage(&used);
  if (ret != HSA_STATUS_SUCCESS)
    return ret;

  const uint64_t total = VramTotal();
  *available_bytes = used >= total ? 0 : total - used;
  return HSA_STATUS_SUCCESS;
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

  priv_size = Wkmi::GetPowerOptPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  Wkmi::FillinPowerOptPrivData(priv_data, restore);

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

bool WDDMDevice::CreateContext(int engine, D3DKMT_HANDLE* handle, uint64_t debugger_data) {
  void *priv_data;
  int priv_size;

  int ordinal = EngineOrdinal(engine, &device_info_);
  if (ordinal < 0)
    return false;

  priv_size = Wkmi::GetContextPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  Wkmi::FillinContextPrivData(priv_data, SupportStateShadowingByCpFw(),
                              device_info_.compute_schedid, debugger_data);

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
    args.Flags.DisableGpuTimeout = Wkmi::ShouldDisableGpuTimeout(engine, &device_info_);

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

bool WDDMDevice::DestroySyncobj(D3DKMT_HANDLE handle) {
  D3DKMT_DESTROYSYNCHRONIZATIONOBJECT args = {0};
  args.hSyncObject = handle;

  NTSTATUS ret = DXCORE_CALL(D3DKMTDestroySynchronizationObject(&args));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    return false;
  }
  return true;
}

bool WDDMDevice::OpenSyncobjFromNtHandle(void *nt_handle,
                                         D3DKMT_HANDLE *out_handle) {
  if (nt_handle == nullptr || out_handle == nullptr) return false;

  D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2 args = {0};
  args.hNtHandle = nt_handle;
  args.hDevice = device_;

  NTSTATUS ret = DXCORE_CALL(D3DKMTOpenSyncObjectFromNtHandle2(&args));
  if (ret != STATUS_SUCCESS) {
    pr_err("D3DKMTOpenSyncObjectFromNtHandle2 failed: 0x%x\n", ret);
    return false;
  }

  *out_handle = args.hSyncObject;
  return true;
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

static void QueryWddmVersion(D3DKMT_HANDLE adapter) {
  D3DKMT_DRIVERVERSION version = static_cast<D3DKMT_DRIVERVERSION>(0);

  if (WDDMQueryAdapter(adapter, KMTQAITYPE_DRIVERVERSION, &version,
                       sizeof(version)) == STATUS_SUCCESS)
    dxg_runtime->wddm_version = version;
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

  if (args.NumAdapters > 0)
    QueryWddmVersion(info[0].hAdapter);

  for (int i = 0; i < args.NumAdapters; i++) {
    D3DKMT_QUERY_DEVICE_IDS query = {0};

    ret = WDDMQueryAdapter(info[i].hAdapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
			   &query, sizeof(query));
    if (ret != STATUS_SUCCESS)
      continue;

    if (query.DeviceIds.VendorID != 0x1002)
      continue;

    supported = Wkmi::QueryAdapterSupported(query.DeviceIds.DeviceID);

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
  return Wkmi::ParseAdapterInfo(adapter_, &device_info_);
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

bool WDDMDevice::CreateQueue(WDDMQueue* queue, uint64_t debugger_data) {
  if (!CreateContext(queue->queue_engine, &queue->context, debugger_data)) return false;

  GpuMemory *gpu_mem = nullptr;
  if (queue->cmdbuf_addr == 0) {
    GpuMemoryCreateInfo create_info{};
    create_info.size = queue->cmdbuf_size;
    create_info.domain = Wkmi::kSystem;

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

  priv_size = Wkmi::GetSubmitPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  Wkmi::FillinSubmitPrivData(priv_data, queue->queue, command_addr, command_size, false);

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

// Compute the CWSR (Context Wave Save/Restore) region size for this device.
//
// Mirrors the Linux KFD calculation in update_ctx_save_restore_size() (queues.c).
// The region must hold, per XCC:
//   - HsaUserContextSaveAreaHeader  (save-area header)
//   - Control stack (wave_num * bytes_per_wave + 8), page-aligned
//   - WG data (VGPR + SGPR + LDS + HW-regs per CU), page-aligned
// multiplied by num_xcc for multi-XCC devices.
//
// Constants ported from queues.c / dxg/queues.cpp:
//   CNTL_STACK_BYTES_PER_WAVE : gfx10+ = 12, older = 8
//   WG_CONTEXT_DATA_SIZE_PER_CU : vgpr_size + SGPR_SIZE_PER_CU + LDS + HWREG_SIZE_PER_CU
//   SGPR_SIZE_PER_CU: gfx10+ = 0x5000 (20 KB), older = 0x4000 (16 KB)  [matches KMD CalCwsrSaveAreaSize]
//   HWREG_SIZE_PER_CU: gfx10+ = 0x1400 (5 KB), older = 0x1000 (4 KB)  [matches KMD CalCwsrSaveAreaSize]
//   vgpr_size: all gfx10/gfx11/gfx12 = 256KB/CU (0x40000)             [matches KMD CalCwsrSaveAreaSize]
//   DEBUGGER_BYTES_PER_WAVE = 32, DEBUGGER_BYTES_ALIGN = 64
uint64_t WDDMDevice::AllocateCwsrSize(uint64_t* out_ctx_size, uint64_t* out_debug_size) const {
  const uint32_t page_size = 4096;
  const uint32_t debugger_bytes_per_wave = 32;
  const uint32_t debugger_bytes_align    = 64;

  const int major = device_info_.major;
  const uint32_t num_xcc         = device_info_.num_xcc ? device_info_.num_xcc : 1;
  // compute_unit_count in device_info_ is the total CU count across all XCCs.
  const uint32_t cu_num          = device_info_.compute_unit_count / num_xcc;
  // wave_per_cu from device_info_ is per-CU (40 for Navi10/12/14, 32 for Navi2x+).
  // KMD CalCwsrSaveAreaSize() split: gfx10+ (FAMILY_NV) vs pre-gfx10
  const bool is_gfx10_plus = (major >= 10);

  const uint32_t wave_per_cu     = device_info_.wave_per_cu ? device_info_.wave_per_cu : 32;
  uint32_t wave_num              = cu_num * wave_per_cu;
  // Pre-Navi (gfx9): Linux queues.c caps wave_num at NumShaderBanks/NumArrays*512.
  // Matches get_num_waves() in queues.c for gfxv < GFX_VERSION_NAVI10.
  // num_shader_engine is total across all XCCs (wkmi.cpp scales it by num_xcc for gfx9.4),
  // so divide by num_xcc to get per-XCC SE count, matching cu_num which is also per-XCC.
  if (!is_gfx10_plus) {
    const uint32_t num_se  = device_info_.num_shader_engine / num_xcc;
    const uint32_t num_sa  = device_info_.shader_array_per_shader_engine;
    if (num_se > 0 && num_sa > 0)
      wave_num = std::min(wave_num, (num_se / num_sa) * 512u);
  }

  // Control stack: gfx10+ = 12 bytes/wave, older = 8 bytes/wave
  const uint32_t bytes_per_wave  = is_gfx10_plus ? 12 : 8;
  uint32_t ctl_stack_size        = wave_num * bytes_per_wave + 8;
  // gfx10.x (Navi) HW control stack RAM is physically limited to 0x7000 bytes.
  // Matches Linux queues.c: if ((gfxv & 0x3f0000) == 0xA0000) ctl_stack_size = MIN(..., 0x7000)
  // major == 10 covers the full gfx10.x family (Navi10/12/14, Navi21/22/23/24).
  if (major == 10)
    ctl_stack_size = std::min(ctl_stack_size, 0x7000u);

  // Sizes from KMD kdx/src/RunListMgr.cpp CalCwsrSaveAreaSize():
  //   gfx10+: sgpr=0x5000, hwreg=0x1400, vgpr=0x40000
  //   pre-gfx10: sgpr=0x4000, hwreg=0x1000, vgpr=0x40000
  const uint32_t vgpr_size_per_cu  = 0x40000;
  const uint32_t sgpr_size_per_cu  = is_gfx10_plus ? 0x5000 : 0x4000;
  const uint32_t hwreg_size_per_cu = is_gfx10_plus ? 0x1400 : 0x1000;
  // LDS size stored in device_info_.lds_size (bytes)
  const uint32_t lds_size_bytes  = device_info_.lds_size;
  const uint32_t wg_size_per_cu  = vgpr_size_per_cu + sgpr_size_per_cu + lds_size_bytes + hwreg_size_per_cu;
  const uint32_t wg_data_size    = cu_num * wg_size_per_cu;

  // Align sizes to page boundary
  auto page_align_up = [&](uint64_t x) -> uint64_t {
    return (x + page_size - 1) & ~(uint64_t)(page_size - 1);
  };
  auto align_up = [&](uint64_t x, uint32_t align) -> uint64_t {
    return (x + align - 1) & ~(uint64_t)(align - 1);
  };

  // ctx_save_restore_size per XCC (header + ctl_stack + wg_data)
  uint64_t ctx_size = page_align_up(sizeof(HsaUserContextSaveAreaHeader) + ctl_stack_size)
                      + page_align_up(wg_data_size);

  // debug_memory_size per XCC
  uint64_t debug_size = align_up(wave_num * debugger_bytes_per_wave, debugger_bytes_align);

  if (out_ctx_size)   *out_ctx_size   = ctx_size;
  if (out_debug_size) *out_debug_size = debug_size;

  uint64_t total = page_align_up((ctx_size + debug_size) * num_xcc);

  return total;
}

// Initialize HsaUserContextSaveAreaHeader for each XCC in the CWSR region.
//
// Mirrors Linux's fill_cwsr_header() in queues.c.  The CWSR allocation is
// divided into equal-sized per-XCC slots of ctx_save_restore_size bytes, each
// beginning with an HsaUserContextSaveAreaHeader.  This must be called after
// the CPU-accessible system memory is allocated so the runtime and debugger can
// locate the control stack, wave state, and debug areas on context save.
//
// Layout per XCC slot (offsets relative to slot base):
//   [0]                 HsaUserContextSaveAreaHeader
//   [ctl_stack_offset]  Control stack  (ctl_stack_size bytes, page-aligned)
//   [wg_data_offset]    Wave/WG state  (wg_data_size bytes,  page-aligned)
//   [debug_offset]      Debugger area  (debug_memory_size bytes, 64-byte aligned)
//
// DebugOffset in each slot's header is relative to that slot's own base address,
// not the start of the allocation.  It points forward to the debug area at the end
// of the last XCC slot: (NumXcc - i) * ctx_save_restore_size from slot i's base.
void WDDMDevice::FillCwsrHeader(void* cpu_addr, uint64_t ctx_save_restore_size,
                                 uint64_t debug_memory_size, uint32_t num_xcc,
                                 volatile HSAint64* error_reason, HSAuint32 error_event_id) {
  for (uint32_t i = 0; i < num_xcc; i++) {
    auto* header = reinterpret_cast<HsaUserContextSaveAreaHeader*>(
        static_cast<uint8_t*>(cpu_addr) + i * ctx_save_restore_size);

    // ErrorEventId: the EventId of the HsaEvent passed at queue creation by the
    // runtime (HSA_EVENTTYPE_SIGNAL, shared across all queues on the agent).
    // Mirrors Linux fill_cwsr_header(): Event ? Event->EventId : 0.
    header->ErrorEventId = error_event_id;

    // ErrorReason mirrors Linux fill_cwsr_header(): pointer to the HSA signal
    // payload used by the runtime to report the error reason bitmask on
    // queue exception.  Sourced from QueueResource->ErrorReason, stored on the
    // queue as error_reason_ and passed through here.
    header->ErrorReason = error_reason;

    // DebugOffset is from this XCC's slot base to the debug area of the *last*
    // XCC slot, matching fill_cwsr_header():
    //   header->DebugOffset = (NumXcc - i) * ctx_save_restore_size
    header->DebugOffset = static_cast<HSAuint32>((num_xcc - i) * ctx_save_restore_size);
    header->DebugSize   = static_cast<HSAuint32>(debug_memory_size * num_xcc);

    // ControlStackOffset/Size and WaveStateOffset/Size describe where the
    // saved control stack and wave state ended up inside this XCC's slot.
    // They are written by the kernel (KFD/KMD) during AMDKFD_IOC_GET_QUEUE_WAVE_STATE
    // after preemption; see struct kfd_context_save_area_header::wave_state.
    // Zero is the correct initial value — no context has been saved yet.
    // rocdbgapi reads these fields (queue.cpp) only after a context save has
    // occurred, so zeroing here is safe and matches Linux fill_cwsr_header().
    header->ControlStackOffset = 0;
    header->ControlStackSize   = 0;
    header->WaveStateOffset    = 0;
    header->WaveStateSize      = 0;
    header->Reserved1          = 0;
  }
}

bool WDDMDevice::CreateHwQueue(WDDMQueue *queue) {
  void *priv_data;
  int priv_size;

  // Allocate CWSR (Context Wave Save/Restore) region in system (GTT) memory.
  // Matches Linux: anonymous mmap + register_svm_range(alwaysMapped=true).
  // locked keeps pages pinned (HSA_SVM_FLAG_GPU_ALWAYS_MAPPED equivalent).
  // The allocation handle is passed to KMD via UMDKMDIF_CREATEHWQUEUE_PRIVATE_DATA::CwsrMemHandle.
  // SDMA queues skip CWSR — mirrors Linux handle_concrete_asic() which returns
  // early for KFD_IOC_QUEUE_TYPE_SDMA/SDMA_XGMI before allocating ctx_save_restore.
  if (queue->cwsr_mem_ == nullptr && queue->needs_cwsr_) {
    uint64_t ctx_save_restore_size = 0;
    uint64_t debug_memory_size = 0;
    GpuMemoryCreateInfo cwsr_create_info{};
    cwsr_create_info.domain = Wkmi::kSystem;
    cwsr_create_info.size = AllocateCwsrSize(&ctx_save_restore_size, &debug_memory_size);

    GpuMemory *cwsr_gpu_mem = nullptr;
    ErrorCode cwsr_code = CreateGpuMemory(cwsr_create_info, &cwsr_gpu_mem);
    if (cwsr_code != ErrorCode::Success) {
      pr_err("CWSR memory allocation failed\n");
      return false;
    }
    queue->cwsr_mem_ = cwsr_gpu_mem->GetGpuMemoryHandle();
    queue->cwsr_mem_handle_ = cwsr_gpu_mem->KmtHandle();

    // Initialise the per-XCC HsaUserContextSaveAreaHeader in the CPU-visible
    // system memory, mirroring fill_cwsr_header() in Linux queues.c.
    const uint32_t num_xcc = device_info_.num_xcc ? device_info_.num_xcc : 1;
    FillCwsrHeader(cwsr_gpu_mem->CpuAddress(), ctx_save_restore_size,
                   debug_memory_size, num_xcc, queue->error_reason_, queue->error_event_id_);
  }

  priv_size = Wkmi::GetHwQueuePrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  bool FwManagedGfxState = SupportStateShadowingByCpFw();
  uint32_t* doorbell_loc = nullptr;
  // amd_queue_memory_ / KmtHandle and AQL parameters only apply when the queue
  // is an AQL ComputeQueue. SDMAQueue (and SwsCompute non-AQL queues) must not
  // be down-cast to ComputeQueue here -- doing so reads garbage and crashes.
  ComputeQueue* compute_queue = dynamic_cast<ComputeQueue*>(queue);
  D3DKMT_HANDLE resource = 0;
  bool is_aql = false;
  if (compute_queue != nullptr && IsAqlSupported()) {
    auto queue_memory = compute_queue->GetAmdQueueMemory();
    resource = queue_memory->KmtHandle();
    is_aql = true;
  }
  Wkmi::FillinHwQueuePrivData(priv_data, FwManagedGfxState, queue->prio, is_aql,
      queue->cmdbuf_addr, queue->cmdbuf_size, reinterpret_cast<uintptr_t>(queue->ring_wptr),
      reinterpret_cast<uintptr_t>(queue->ring_rptr), resource, &doorbell_loc,
      queue->cwsr_mem_handle_);

  D3DKMT_CREATEHWQUEUE createHwQueue = {0};
  createHwQueue.hHwContext = queue->context;
  createHwQueue.Flags.DisableGpuTimeout = Wkmi::ShouldDisableGpuTimeout(queue->queue_engine, &device_info_);
  createHwQueue.pPrivateDriverData = priv_data;
  createHwQueue.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = DXCORE_CALL(D3DKMTCreateHwQueue(&createHwQueue));
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    free(priv_data);
    if (queue->cwsr_mem_ != nullptr) {
      delete GpuMemory::Convert(queue->cwsr_mem_);
      queue->cwsr_mem_ = nullptr;
      queue->cwsr_mem_handle_ = 0;
    }
    return false;
  }
  if (doorbell_loc != nullptr) {
    queue->aql_doorbell_offset_ = *doorbell_loc;
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

  if (queue->cwsr_mem_ != nullptr) {
    auto cwsr_gpu_mem = GpuMemory::Convert(queue->cwsr_mem_);
    delete cwsr_gpu_mem;
    queue->cwsr_mem_ = nullptr;
    queue->cwsr_mem_handle_ = 0;
  }

  return true;
}

bool WDDMDevice::SubmitToHwQueue(WDDMQueue *queue, uint64_t command_addr,
                                uint64_t command_size, uint64_t fence_value) {
  void *priv_data;
  int priv_size;

  priv_size = Wkmi::GetSubmitPrivDataSize();
  priv_data = malloc(priv_size);
  assert(priv_data);
  memset(priv_data, 0, priv_size);
  Wkmi::FillinSubmitPrivData(priv_data, queue->queue, command_addr, command_size, true);

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
bool WDDMDevice::SetCuMask(uint32_t doorbell, uint32_t cu_mask_count,
                           const uint32_t* queue_cu_mask) {
#if defined(WIN32)
  pr_debug("set CU mask doorbell: %d -> %d\n", doorbell, cu_mask_count);
  // Fill private KMD data
  int priv_size = Wkmi::GetCuMaskPrivDataSize();
  void* priv_data = alloca(priv_size);
  memset(priv_data, 0, priv_size);
  Wkmi::FillinCuMaskPrivData(priv_data, doorbell, cu_mask_count, queue_cu_mask);
  // Update CU mask for the queue
  if (Escape(priv_data, priv_size, false)) {
    return true;
  } else {
    pr_debug("CU mask escape/update failed for doorbell %u\n", doorbell);
    return false;
  }
#endif
  return false;
}

// ================================================================================================
bool WDDMDevice::SubmitToAqlQueue(WDDMQueue* queue, uint64_t command_addr, uint64_t command_size,
                                  uint64_t fence_value) {
#if defined(WIN32)
  int priv_size = Wkmi::GetAqlSubmitPrivDataSize();
  void* priv_data = alloca(priv_size);
  memset(priv_data, 0, priv_size);
  Wkmi::FillinAqlSubmitPrivData(priv_data, fence_value);
  // HwQueueProgressFenceId is UINT64 in the DDI; drop the 32-bit
  // truncation so the full fence value reaches WDDM.
  D3DKMT_SUBMITCOMMANDTOHWQUEUE args = {
      .hHwQueue = queue->queue,
      .HwQueueProgressFenceId = fence_value + 1,
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
bool WDDMDevice::Escape(void* priv_data, uint32_t priv_size, bool hw_access) const {
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
      int priv_size = Wkmi::GetRegisterEventPrivDataSize();
      void* priv_data = alloca(priv_size);
      memset(priv_data, 0, priv_size);
      Wkmi::FillinRegisterEventPrivData(priv_data, reinterpret_cast<uint64_t>(event_handle),
                                               event_id);
      // Make the escape call to KMD to get the mailbox and assign event ID
      if (Escape(priv_data, priv_size, false)) {
        // Initialize the mailbox array if it's the first call
        if (base_mailbox_va_ == 0) {
          base_mailbox_va_ = Wkmi::GetRegisterEventMailbox(priv_data);
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
    int priv_size = Wkmi::GetUnregisterEventPrivDataSize();
    void* priv_data = alloca(priv_size);
    memset(priv_data, 0, priv_size);
    Wkmi::FillinUnregisterEventPrivData(priv_data, reinterpret_cast<uint64_t>(event_handle));
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

bool WDDMDevice::GetKmdDbgVersion(struct Wkmi::KmdDbgVersion* version) const {
  int priv_size = Wkmi::GetDebuggerCmdPrivDataSize();
  void* priv_data = alloca(priv_size);

  memset(priv_data, 0, priv_size);
  Wkmi::FillinKmdDbgVersionPrivData(priv_data);

  if (Escape(priv_data, priv_size, true)) {
    Wkmi::GetKmdDbgVersion(priv_data, version);
    return true;
  }

  return false;
}

bool WDDMDevice::RegisterRuntimeState(uint32_t runtime_state, const void* r_debug,
                                      bool ttmp_setup_hint) const {
  int priv_size = Wkmi::GetDebuggerCmdPrivDataSize();
  void* priv_data = alloca(priv_size);

#ifdef WIN32
  HANDLE init_event = CreateEvent(nullptr, true, false, TEXT("RuntimeInitEvent"));
  if (!init_event) {
    return false;
  }
#else   // !WIN32
  // It isn't clear yet how system events are going to be shared across OSes.
  HANDLE init_event = nullptr;
  pr_warn_once("not supported\n");
  return false;
#endif  // !WIN32

  memset(priv_data, 0, priv_size);
  Wkmi::FillinRegisterRuntimeStatePrivData(priv_data, runtime_state, r_debug, ttmp_setup_hint,
                                           init_event);

  bool ret = Escape(priv_data, priv_size, true);

#ifdef WIN32
  if (ret) {
    ret = (WaitForSingleObject(init_event, INFINITE) == WAIT_OBJECT_0);
  }

  CloseHandle(init_event);
#endif  // WIN32
  return ret;
}

bool WDDMDevice::SetTrapHandler(uint64_t tba, uint64_t tma) const {
  int priv_size = Wkmi::GetDebuggerCmdPrivDataSize();
  void* priv_data = alloca(priv_size);

  memset(priv_data, 0, priv_size);
  Wkmi::FillinTrapHandlerPrivData(priv_data, tba, tma);

  return Escape(priv_data, priv_size, true);
}


} // namespace thunk
} // namespace wsl
