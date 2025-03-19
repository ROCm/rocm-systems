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

#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <linux/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "impl/wddm/status.h"
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "impl/wddm/queue.h"

namespace wsl {
namespace thunk {

const uint32_t WDDMDevice::cmdbuf_aql_frame_num_ = 0x1000;

WDDMDevice::WDDMDevice(D3DKMT_HANDLE adapter, LUID adapter_luid)
  : adapter_(adapter), adapter_luid_(adapter_luid) {
  memset(&device_info_, 0, sizeof(device_info_));

  ParseDeviceInfo();
  CreateDevice();
  SetPowerOptimization(false);
  CreatePagingQueue();
  ReserveLocalHeapSpace();
  ReserveSystemHeapSpace();
  InitHandleApertureSpace();
  InitVaMgr();
  InitHandleApertureMgr();
  InitCmdbufInfo();
}

WDDMDevice::~WDDMDevice() {
  FreeLocalHeapSpace();
  FreeSystemHeapSpace();
  DestroyPagingQueue();
  SetPowerOptimization(true);
  DestroyDevice();

  DestroyDeviceInfo();
}

static NTSTATUS WDDMQueryAdapter(D3DKMT_HANDLE adapter, KMTQUERYADAPTERINFOTYPE type,
				 void *data, int size)
{
  D3DKMT_QUERYADAPTERINFO args = {0};

  args.hAdapter = adapter;
  args.Type = type;
  args.pPrivateDriverData = data;
  args.PrivateDriverDataSize = size;

  return D3DKMTQueryAdapterInfo(&args);
}

uint64_t WDDMDevice::VramAvail(void) {
  D3DKMT_QUERYSTATISTICS stats;
  NTSTATUS ret;
  uint64_t usedVis = 0;
  uint64_t usedInv = 0;

  // wait fence complete
  uint64_t value = page_fence_value_.load();
  if(!CpuWait(&page_syncobj_, &value, 1, false))
    return HSA_STATUS_ERROR;

  // local cpu-visible memory
  memset(&stats, 0, sizeof(D3DKMT_QUERYSTATISTICS));
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
  stats.AdapterLuid = adapter_luid_;
  stats.QuerySegment.SegmentId = 0;
  ret = D3DKMTQueryStatistics(&stats);
  if (ret == 0)
    usedVis = stats.QueryResult.SegmentInformation.BytesResident;

  // local invisible memory
  memset(&stats, 0, sizeof(D3DKMT_QUERYSTATISTICS));
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
  stats.AdapterLuid = adapter_luid_;
  stats.QuerySegment.SegmentId = 1;

  ret = D3DKMTQueryStatistics(&stats);
  if (ret == 0)
    usedInv = stats.QueryResult.SegmentInformation.BytesResident;

  return LocalHeapSize() - usedVis - usedInv;
}

bool WDDMDevice::CreateDevice(void) {
  D3DKMT_CREATEDEVICE args = {0};
  args.hAdapter = adapter_;

  NTSTATUS ret = D3DKMTCreateDevice(&args);
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

  NTSTATUS ret = D3DKMTDestroyDevice(&args);
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::CreatePagingQueue(void) {
  D3DKMT_CREATEPAGINGQUEUE args = {0};
  args.hDevice = device_;
  args.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;

  NTSTATUS ret = D3DKMTCreatePagingQueue(&args);
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

  NTSTATUS ret = D3DKMTDestroyPagingQueue(&args);
  if (ret == STATUS_SUCCESS)
    return true;

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::CommitSystemHeapSpace(void* addr, int64_t size, bool lock) {
  int32_t protFlags = PROT_READ | PROT_WRITE | PROT_EXEC;
  int32_t mapFlags = MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED|
                     MAP_NORESERVE|MAP_UNINITIALIZED;
  if (lock)
    mapFlags |= MAP_LOCKED;
  void* paddr = mmap(addr, size, protFlags, mapFlags, -1, 0);
  if (paddr == MAP_FAILED) {
    pr_err("fail to commit %s addr = %p, paddr = %p\n", (lock ? "locked" : ""), addr, paddr);
    return false;
  }
  assert(addr == paddr);

  /*if (!Runtime::runtime_singleton_->PinWARequired())
      return true;*/

  /*
   * Do not make the pages in this range available to the child
   * after a fork(2).  This is useful to prevent copy-on-write
   * semantics from changing the physical location of a page if
   * the parent writes to it after a fork(2).  (Such page
   * relocations cause problems for hardware that DMAs into the
   * page.)
   *
   * https://man7.org/linux/man-pages/man2/madvise.2.html
   */
  if (madvise(addr, size, MADV_DONTFORK))
    pr_err("fail to set MADV_DONTFORK for addr = %p\n", addr);

  return true;
}

bool WDDMDevice::DecommitSystemHeapSpace(void* addr, int64_t size) {
  int32_t protFlags = PROT_NONE;
  int32_t mapFlags = MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED|
                     MAP_NORESERVE|MAP_UNINITIALIZED;
  void* paddr = mmap(addr, size, protFlags, mapFlags, -1, 0);
  if (paddr == MAP_FAILED) {
    pr_err("fail to decommit addr = %p, paddr = %p\n", addr, paddr);
    return false;
  }
  assert(addr == paddr);
  return true;
}

bool WDDMDevice::CommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd, bool lock) {
  int fd = -1;

  if (memfd == -1) {
    fd = memfd_create("rocr4wsl_gtt", MFD_CLOEXEC);
    if (fd < 0) {
      pr_err("memfd_create failed\n");
      return false;
    }

    ftruncate(fd, size);
  } else {
    fd = memfd;
  }

  int32_t protFlags = PROT_READ | PROT_WRITE;
  int32_t mapFlags = MAP_SHARED | MAP_FIXED | MAP_NORESERVE |
      MAP_UNINITIALIZED | (lock ? MAP_LOCKED : 0);

  void* paddr = mmap(addr, size, protFlags, mapFlags, fd, 0);
  if (paddr == MAP_FAILED) {
    pr_err("fail to commit %s addr = %p, paddr = %p\n", (lock ? "locked" : ""), addr, paddr);
    if (memfd == -1)
      close(fd);
    return false;
  }
  assert(addr == paddr);

  memfd = fd;

  if (madvise(addr, size, MADV_DONTFORK))
    pr_err("fail to set MADV_DONTFORK for addr = %p\n", addr);

  return true;
}

bool WDDMDevice::DecommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd) {
  if (munmap(addr, size) != 0) {
    pr_err("fail to unmap = %p \n", addr);
    return false;
  }
  close(memfd);
  memfd = -1;
  return true;
}

bool WDDMDevice::ReserveSystemHeapSpace() {
  struct sysinfo info;
  int ret = sysinfo(&info);
  uint64_t max_ram = 0x10000000000;
  uint64_t alignment = 0x100000000;
  assert(!ret);

  int32_t protFlags = PROT_NONE;
  // minimum of reserve size is 8G, maximum of reserve size is 1T.
  system_heap_space_size_ = std::min(AlignUp(info.totalram, alignment) * 2, max_ram);
  void* cpu = mmap(NULL, system_heap_space_size_, protFlags,
              MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (cpu == MAP_FAILED) {
    pr_err("fail to reserve system_heap_space_size_ = %lx \n", system_heap_space_size_);
    return false;
  }

  system_heap_space_start_ = (uint64_t)cpu;
  return true;
}

bool WDDMDevice::FreeSystemHeapSpace(void) {
  void *cpu = (void *)system_heap_space_start_;
  if (munmap(cpu, system_heap_space_size_) != 0) {
    pr_err("fail to unmap = %p \n", cpu);
    return false;
  }
  return true;
}

/*
 * To find the avaliable same range for cpu
 * virtual space and gpu virtual space.
 * sys_va_size of cpu va range is larger 1G
 * than gpu va range, otherwise ReserveGPUVirtualAddress
 * will return error.
 */
bool WDDMDevice::ReserveLocalHeapSpace(void) {
  uint64_t sys_va[16] = {0};
  uint64_t local_va;
  uint64_t sys_va_size;
  int match_index = -1;
  uint64_t align = 0x40000000; /* 1G */
  void* ptr = NULL;

  local_heap_space_start_ = 0;
  local_heap_space_size_ = AlignUp(LocalHeapSize(), align) * 4;
  sys_va_size = local_heap_space_size_ + align;

  /* it will retry 16 times to find the avaliable range. */
  for (int i = 0; i < 16; i++) {
    local_va = 0;
    ptr = mmap(NULL, sys_va_size , PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      pr_err("fail to reserve cpu va in %d time!\n", i);
      break;
    }

    sys_va[i] = (uint64_t)ptr;

    if (d3dthunk::ReserveGpuVirtualAddress(
          adapter_, local_heap_space_size_,
          (uint64_t)ptr,
          (uint64_t)ptr + sys_va_size, &local_va) == ErrorCode::Success) {

      match_index = i;
      local_heap_space_start_ = local_va;
      pr_debug("success to reserve gpu va %lx and va cpu %p in %d time\n",
               local_va, ptr, i);
      break;
    } else {
      pr_err("%s fail to reserve gpu va for cpu va %p in %d time!\n",
              __FUNCTION__, ptr, i);
    }
  }

  if (match_index >= 0) {
    /* release cpu unused ranges*/
    uint64_t left_size = local_va - sys_va[match_index];
    uint64_t right_size = align - left_size;
    if ((left_size > 0) && munmap((void*)sys_va[match_index], left_size))
      pr_err("fail to unmap left %lx with size %lx\n", sys_va[match_index], left_size);
    if ((right_size > 0) && munmap((void*)(local_va + local_heap_space_size_), right_size))
      pr_err("fail to unmap right %lx with size %lx\n", (local_va + local_heap_space_size_), right_size);
  } else {
      pr_err("fail to reserve Local Heap Space!\n");
  }

  /* free match fail address for cpu va */
  int free = match_index >= 0 ? match_index : 16;
  for (int j = 0; j < free; j++) {
    if (sys_va[j] != 0 && munmap((void*)sys_va[j], sys_va_size)) {
      pr_err("fail to unmap %d %lx\n", j, sys_va[j]);
    }
  }

  return match_index >= 0;
}

bool WDDMDevice::FreeLocalHeapSpace(void) {
  d3dthunk::FreeGpuVirtualAddress(adapter_, local_heap_space_start_, local_heap_space_size_);
  void *cpu = (void *)local_heap_space_start_;
  return munmap(cpu, local_heap_space_size_) == 0;
}

void WDDMDevice::InitVaMgr() {
  local_va_mgr_ = std::make_unique<VaMgr>(local_heap_space_start_,
                                          local_heap_space_size_,
                                          DEFAULT_GPU_PAGE_SIZE);
}

void WDDMDevice::InitHandleApertureMgr() {
  handle_aperture_mgr_ = std::make_unique<VaMgr>(handle_aperture_start_,
                                                 handle_aperture_size_,
                                                 DEFAULT_GPU_PAGE_SIZE);
}

bool WDDMDevice::InitHandleApertureSpace(void) {
  handle_aperture_start_ = START_NON_CANONICAL_ADDR;
  handle_aperture_size_ = 1ULL << 47;

  while (handle_aperture_start_ < END_NON_CANONICAL_ADDR - 1) {
    if (device_info_.private_aperture_base &&
      IS_OVERLAPPING(device_info_.private_aperture_base,
                     device_info_.private_aperture_size,
                     handle_aperture_start_,
                     handle_aperture_size_)) {
      handle_aperture_start_ += (1ULL << 47);
      continue;
    }

    if (device_info_.shared_aperture_base &&
      IS_OVERLAPPING(device_info_.shared_aperture_base,
                     device_info_.shared_aperture_size,
                     handle_aperture_start_,
                     handle_aperture_size_)) {
      handle_aperture_start_ += (1ULL << 47);
      continue;
    }

    pr_debug("handle aperture start %lx, size %lx\n", handle_aperture_start_, handle_aperture_size_);
    return true;
  }

  handle_aperture_start_ = 0;
  pr_err("fail\n");

  return false;
}

void WDDMDevice::SetPowerOptimization(bool restore) {
  void *priv_data;
  int priv_size;

  priv_size = thunk_proxy::CreatePowerOptPrivData(&priv_data, restore);

  D3DKMT_ESCAPE d3dkmt_escape;
  memset(&d3dkmt_escape, 0, sizeof(d3dkmt_escape));

  d3dkmt_escape.hAdapter              = adapter_;
  d3dkmt_escape.hDevice               = device_;
  d3dkmt_escape.hContext              = 0; //KMD only use device to identify the process
  d3dkmt_escape.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
  d3dkmt_escape.pPrivateDriverData    = priv_data;
  d3dkmt_escape.PrivateDriverDataSize = priv_size;
  d3dkmt_escape.Flags.HardwareAccess  = true;

  NTSTATUS status = D3DKMTEscape(&d3dkmt_escape);
  pr_debug("status %d, restore %d\n", status, restore);
  thunk_proxy::DestroyPrivData(priv_data);
}

ErrorCode WDDMDevice::ReserveGpuVirtualAddress(const thunk_proxy::AllocDomain domain,
                                               gpusize hit_base_addr, gpusize size,
                                               gpusize *out_gpu_virt_addr, gpusize alignment, bool lock) {
  gpusize gpu_addr = 0;
  ErrorCode code = ErrorCode::Success;

  if (domain == thunk_proxy::kSystem) {

    code = d3dthunk::ReserveGpuVirtualAddress(adapter_, size,
                                          system_heap_space_start_,
                                          system_heap_space_start_ + system_heap_space_size_,
                                          &gpu_addr);
    if (code != ErrorCode::Success)
      return code;

    if (!CommitSystemHeapSpace((void*)gpu_addr, size, lock)) {
      d3dthunk::FreeGpuVirtualAddress(adapter_, gpu_addr, size);
      code = ErrorCode::SyscallFail;
    }
  } else {
    uint64_t align = alignment == 0 ? (64 * 1024) : alignment; // default 64K alignment
    if (domain == thunk_proxy::kLocal && size >= GPU_HUGE_PAGE_SIZE)
      align = GPU_HUGE_PAGE_SIZE;

    gpu_addr = local_va_mgr_->Alloc(size, align, hit_base_addr);
    if (gpu_addr == 0)
      code = ErrorCode::OutOfGpuMemory;

  }

  *out_gpu_virt_addr = (code == ErrorCode::Success) ? gpu_addr : 0;
  return code;
}

ErrorCode WDDMDevice::FreeGpuVirtualAddress(const thunk_proxy::AllocDomain domain,
                                            gpusize gpu_addr, gpusize size) {
  auto code = ErrorCode::Success;

  if (domain == thunk_proxy::kSystem) {

      DecommitSystemHeapSpace((void *)gpu_addr, size);

      d3dthunk::FreeGpuVirtualAddressArgs free_args{};
      free_args.hAdapter = adapter_;
      free_args.BaseAddress = gpu_addr;
      free_args.Size = size;

      code = d3dthunk::FreeGpuVirtualAddress(&free_args);
  } else {
    local_va_mgr_->Free(gpu_addr);
  }

  return code;
}

ErrorCode WDDMDevice::ReserveIPCSysMem(gpusize size,
                                       gpusize *out_gpu_virt_addr, gpusize alignment,
                                       int &memfd, bool lock) {
  gpusize gpu_addr = 0;
  ErrorCode code = ErrorCode::Success;

  code = d3dthunk::ReserveGpuVirtualAddress(adapter_, size,
                                            system_heap_space_start_,
                                            system_heap_space_start_ + system_heap_space_size_,
                                            &gpu_addr);
  if (code != ErrorCode::Success)
    return code;

  if (!CommitSystemHeapSpaceIPC((void*)gpu_addr, size, memfd, lock)) {
    d3dthunk::FreeGpuVirtualAddress(adapter_, gpu_addr, size);
    code = ErrorCode::SyscallFail;
  }

  *out_gpu_virt_addr = (code == ErrorCode::Success) ? gpu_addr : 0;
  return code;
}

ErrorCode WDDMDevice::FreeIPCSysMem(gpusize gpu_addr, gpusize size, int &memfd) {
  auto code = ErrorCode::Success;

  DecommitSystemHeapSpaceIPC((void *)gpu_addr, size, memfd);

  d3dthunk::FreeGpuVirtualAddressArgs free_args{};
  free_args.hAdapter = adapter_;
  free_args.BaseAddress = gpu_addr;
  free_args.Size = size;

  code = d3dthunk::FreeGpuVirtualAddress(&free_args);

  return code;
}

ErrorCode WDDMDevice::HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr) {
  uint64_t align = DEFAULT_GPU_PAGE_SIZE;

  if (size >= GPU_HUGE_PAGE_SIZE)
    align = GPU_HUGE_PAGE_SIZE;

  *out_gpu_virt_addr = handle_aperture_mgr_->Alloc(size, align);
  if (*out_gpu_virt_addr == 0)
    return ErrorCode::OutOfHandleApeMemory;

  return ErrorCode::Success;
}

void WDDMDevice::HandleApertureFree(gpusize gpu_addr) {
  handle_aperture_mgr_->Free(gpu_addr);
}

void WDDMDevice::UpdatePageFence(uint64_t fence_value) {
  uint64_t current = page_fence_value_.load();

  // atomically set fence value when target is bigger than current one
  do {
    if (current >= fence_value)
      break;
  } while (!page_fence_value_.compare_exchange_weak(current, fence_value));
}

ErrorCode WDDMDevice::CreateGpuMemory(const GpuMemoryCreateInfo &create_info, GpuMemory **gpu_mem) {
  ErrorCode ret;

  *gpu_mem = nullptr;
  auto mem = new GpuMemory(this);
  if (create_info.dmabuf_fd > 0)
    ret = mem->ImportPhysicalHandle(create_info);
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

  NTSTATUS ret = D3DKMTLock2(&args);
  if (ret == STATUS_SUCCESS)
    return args.pData;

  pr_err("fail %x\n", ret);
  return NULL;
}

bool WDDMDevice::Unlock(D3DKMT_HANDLE handle) {
  D3DKMT_UNLOCK2 args = {0};
  args.hDevice = device_;
  args.hAllocation = handle;

  NTSTATUS ret = D3DKMTUnlock2(&args);
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

  bool FwManagedGfxState = SupportStateShadowingByCpFw();
  priv_size = thunk_proxy::CreateContextPrivData(&priv_data, FwManagedGfxState);

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

  NTSTATUS ret = D3DKMTCreateContextVirtual(&args);
  if (ret == STATUS_SUCCESS) {
    *handle = args.hContext;
    thunk_proxy::DestroyPrivData(priv_data);
    return true;
  }

  thunk_proxy::DestroyPrivData(priv_data);

  pr_err("fail %x\n", ret);
  return false;
}

bool WDDMDevice::DestroyContext(D3DKMT_HANDLE handle) {
  D3DKMT_DESTROYCONTEXT args = {0};
  args.hContext = handle;

  NTSTATUS ret = D3DKMTDestroyContext(&args);
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

  NTSTATUS ret = D3DKMTWaitForSynchronizationObjectFromGpu(&args);
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

  NTSTATUS ret = D3DKMTSignalSynchronizationObjectFromGpu(&args);
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

  NTSTATUS ret = D3DKMTWaitForSynchronizationObjectFromCpu(&args);
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

  NTSTATUS ret = D3DKMTCreateSynchronizationObject2(&args);
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

  NTSTATUS ret = D3DKMTDestroySynchronizationObject(&args);
  if (ret != STATUS_SUCCESS)
    pr_err("fail %x\n", ret);
}

void WDDMDevice::InitCmdbufInfo(void) {
  if (device_info_.major == 9) {
    cmdbuf_aql_frame_size_ = 2 * sizeof(gfx9::AcquireMemTemplate);
  } else if (device_info_.major >= 10) {
    cmdbuf_aql_frame_size_ = 2 * sizeof(gfx10::AcquireMemTemplate);
  }

  if (device_info_.major >= 11)
    cmdbuf_aql_frame_size_ += sizeof(SetScratchTemplate);

  cmdbuf_aql_frame_size_ +=
    sizeof(PM4MEC_COPY_DATA) * 2 +
    sizeof(BarrierTemplate) * 2 +
    sizeof(DispatchTemplate) +
    sizeof(AtomicTemplate) * 2;
  cmdbuf_aql_frame_size_ = AlignUp(cmdbuf_aql_frame_size_, 0x10);

  cmdbuf_size_ = AlignUp(cmdbuf_aql_frame_num_ * cmdbuf_aql_frame_size_, 0x1000);
}

uint32_t WDDMDevice::LdsBlocks(const hsa_kernel_dispatch_packet_t *pkt) {
  static const uint32_t blk_sz = 512;
  uint32_t total_sz = pkt->group_segment_size;
  uint32_t blk_num = (total_sz + blk_sz - 1) / blk_sz;
  return blk_num;
}

NTSTATUS WDDMGetAdapters(D3DKMT_ADAPTERINFO *&adapters, int &num_adapters)
{
  bool supported = false;
  D3DKMT_ENUMADAPTERS2 args = {0};
  NTSTATUS ret = D3DKMTEnumAdapters2(&args);
  if (ret != STATUS_SUCCESS)
    return ret;

  if (!args.NumAdapters) {
    adapters = NULL;
    num_adapters = 0;
    return STATUS_SUCCESS;
  }

  D3DKMT_ADAPTERINFO *info = new D3DKMT_ADAPTERINFO[args.NumAdapters];
  if (!info)
    return STATUS_NO_MEMORY;

  args.pAdapters = info;
  ret = D3DKMTEnumAdapters2(&args);
  if (ret != STATUS_SUCCESS)
    goto err_out0;

  adapters = new D3DKMT_ADAPTERINFO[args.NumAdapters];
  if (!adapters)
    goto err_out0;

  num_adapters = 0;
  for (int i = 0; i < args.NumAdapters; i++) {
    D3DKMT_ADAPTERREGISTRYINFO query = {0};

    ret = WDDMQueryAdapter(info[i].hAdapter, KMTQAITYPE_ADAPTERREGISTRYINFO,
			   &query, sizeof(query));
    if (ret != STATUS_SUCCESS)
      goto err_out1;

    if (!std::wcsstr(query.ChipType, L"AMD"))
      continue;

    supported = thunk_proxy::QueryAdapterSupported(info[i].hAdapter);

    if (supported) {
      adapters[num_adapters++] = info[i];
    }
  }

  delete[] info;
  return STATUS_SUCCESS;

 err_out1:
  delete[] adapters;
  adapters = NULL;
 err_out0:
  delete[] info;
  return ret;
}

bool WDDMDevice::ParseDeviceInfo() {
  bool ret;

  memset(&device_info_, 0, sizeof(device_info_));
  ret = thunk_proxy::ParseAdapterInfo(adapter_, &device_info_);
  if (!ret)
    return false;

  return true;
}

void WDDMDevice::DestroyDeviceInfo() {
  free(device_info_.adapter_info);
}

void WDDMDevice::GetClockCounters(uint64_t *gpu, uint64_t *cpu) {
  void *priv_data;
  int priv_size;

  priv_size = thunk_proxy::CreateCalibratedTimestampsPrivData(&priv_data);

  D3DKMT_ESCAPE d3dkmt_escape;
  memset(&d3dkmt_escape, 0, sizeof(d3dkmt_escape));

  d3dkmt_escape.hAdapter              = adapter_;
  d3dkmt_escape.hDevice               = device_;
  d3dkmt_escape.hContext              = 0; //KMD only use device to identify the process
  d3dkmt_escape.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
  d3dkmt_escape.pPrivateDriverData    = priv_data;
  d3dkmt_escape.PrivateDriverDataSize = priv_size;
  d3dkmt_escape.Flags.HardwareAccess  = true;

  NTSTATUS status = D3DKMTEscape(&d3dkmt_escape);
  if (status) {
    pr_debug("status %d \n", status);
  } else {
    thunk_proxy::QueryCalibratedTimestamps(priv_data, gpu, cpu);
  }
  thunk_proxy::DestroyPrivData(priv_data);
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

  priv_size = thunk_proxy::CreateSubmitPrivData(&priv_data, queue->queue, command_addr, command_size, false);

  D3DKMT_SUBMITCOMMAND args = {0};
  args.Commands = command_addr;
  args.CommandLength = command_size;
  args.BroadcastContextCount = 1;
  args.BroadcastContext[0] = queue->context;
  args.pPrivateDriverData = priv_data;
  args.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = D3DKMTSubmitCommand(&args);
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    thunk_proxy::DestroyPrivData(priv_data);
    return false;
  }

  thunk_proxy::DestroyPrivData(priv_data);

  if (!GpuSignal(queue->context, &queue->syncobj, &fence_value, 1))
    return false;

  return true;
}

bool WDDMDevice::CreateHwQueue(WDDMQueue *queue) {
  void *priv_data;
  int priv_size;

  bool FwManagedGfxState = SupportStateShadowingByCpFw();
  priv_size = thunk_proxy::CreateHwQueuePrivData(&priv_data, queue->context,
                                                  FwManagedGfxState, queue->prio);

  D3DKMT_CREATEHWQUEUE createHwQueue = {0};
  createHwQueue.hHwContext = queue->context;
  createHwQueue.Flags.DisableGpuTimeout = thunk_proxy::ShouldDisableGpuTimeout(queue->queue_engine, &device_info_);
  createHwQueue.pPrivateDriverData = priv_data;
  createHwQueue.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = D3DKMTCreateHwQueue(&createHwQueue);
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    thunk_proxy::DestroyPrivData(priv_data);
    return false;
  }

  thunk_proxy::DestroyPrivData(priv_data);

  queue->queue = createHwQueue.hHwQueue;
  queue->syncobj = createHwQueue.hHwQueueProgressFence;
  queue->sync_addr = (uint64_t *)createHwQueue.HwQueueProgressFenceCPUVirtualAddress;

  return true;
}

bool WDDMDevice::DestroyHwQueue(WDDMQueue *queue) {
   D3DKMT_DESTROYHWQUEUE DestroyHwQueue = {
    .hHwQueue = queue->queue,
  };

  NTSTATUS ret = D3DKMTDestroyHwQueue(&DestroyHwQueue);
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

  priv_size = thunk_proxy::CreateSubmitPrivData(&priv_data, queue->queue, command_addr, command_size, true);

  D3DKMT_SUBMITCOMMANDTOHWQUEUE args = {0};
  args.hHwQueue = queue->queue;
  args.HwQueueProgressFenceId = fence_value;
  args.CommandBuffer = command_addr;
  args.CommandLength = command_size;
  args.pPrivateDriverData = priv_data;
  args.PrivateDriverDataSize = priv_size;

  NTSTATUS ret = D3DKMTSubmitCommandToHwQueue(&args);
  if (ret != STATUS_SUCCESS) {
    pr_err("fail %x\n", ret);
    thunk_proxy::DestroyPrivData(priv_data);
    return false;
  }

  thunk_proxy::DestroyPrivData(priv_data);

  return true;
}

} // namespace thunk
} // namespace wsl
