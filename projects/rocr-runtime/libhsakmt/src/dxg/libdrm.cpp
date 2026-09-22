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

#include <cstdint>

#if defined(__linux__)
#include "impl/wddm/types.h"
#endif
#include "impl/wddm/device.h"

HSAKMT_STATUS HSAKMTAPI hsaKmtGetAMDGPUDeviceHandle(
    HSAuint32 NodeId, HsaAMDGPUDeviceHandle *DeviceHandle) {
  CHECK_DXG_OPEN();

  wsl::thunk::WDDMDevice *pDevice = get_wddmdev(NodeId);
  if (pDevice != nullptr) {
    *DeviceHandle = reinterpret_cast<HsaAMDGPUDeviceHandle>(pDevice);
    return HSAKMT_STATUS_SUCCESS;
  }
  return HSAKMT_STATUS_ERROR;
}

HSAKMTAPI int amdgpu_device_initialize(int fd,
                                       uint32_t *major_version,
                                       uint32_t *minor_version,
                                       amdgpu_device_handle *device_handle) {
  return 0;
}

HSAKMTAPI int amdgpu_device_deinitialize(amdgpu_device_handle device_handle) {
  return 0;
}

HSAKMTAPI int amdgpu_query_gpu_info(amdgpu_device_handle dev,
                                    struct amdgpu_gpu_info *info) {
  wsl::thunk::WDDMDevice *pDevice =
    reinterpret_cast<wsl::thunk::WDDMDevice *>(dev);
  memset(info, 0, sizeof(*info));
  info->gpu_counter_freq = pDevice->GPUCounterFrequency() / 1000ull;
  return 0;
}

HSAKMTAPI int amdgpu_device_get_fd(amdgpu_device_handle dev) {
  return dxg_runtime->dxg_fd;
}

HSAKMTAPI int amdgpu_bo_cpu_map(amdgpu_bo_handle bo, void **cpu) {
  wsl::thunk::GpuMemory *gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory *>(bo);
  if (gpu_mem->IsSysMemFd())
    *cpu = gpu_mem->CpuAddress();
  return 0;
}

HSAKMTAPI int amdgpu_bo_free(amdgpu_bo_handle buf_handle) {
  wsl::thunk::GpuMemory *gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory *>(buf_handle);
  void *MemoryAddress = gpu_mem->IsVaAllocated() ? (void*)gpu_mem->GpuAddress() : (void*)gpu_mem->HandleApeAddress();
  auto ret = hsaKmtFreeMemory((void*)MemoryAddress, gpu_mem->Size());
  return ret == HSAKMT_STATUS_SUCCESS ? 0 : -1;
}

HSAKMTAPI int amdgpu_bo_export(amdgpu_bo_handle bo,
                               enum amdgpu_bo_handle_type type,
                               uint32_t *shared_handle) {
  *shared_handle = 0;
  return 0;
}

HSAKMTAPI int amdgpu_bo_import(amdgpu_device_handle dev,
                               enum amdgpu_bo_handle_type type,
                               uint32_t shared_handle,
                               struct amdgpu_bo_import_result *output) {
  if (type != amdgpu_bo_handle_type_dma_buf_fd) {
    pr_err("not implemented\n");
    return -1;
  }

  // return if handle is invalid
  if (static_cast<int>(shared_handle) == -1) {
    output->buf_handle = 0;
    return 0;
  }

  wsl::thunk::WDDMDevice *pDevice = reinterpret_cast<wsl::thunk::WDDMDevice *>(dev);
  wsl::thunk::GpuMemoryHandle mem_handle;
  bool is_ipc_memfd = is_ipc_sysmemfd(shared_handle);
  bool alloc_va = is_ipc_memfd;

  // kmt handle importer is false for dma_buf_fd
  HSAKMT_STATUS ret = import_dmabuf_fd(shared_handle, pDevice->NodeId(), alloc_va, is_ipc_memfd,
                                       &mem_handle, false);
  if (ret == HSAKMT_STATUS_SUCCESS) {
    //use GpuMemory object handle as drm buf handle
    output->buf_handle = reinterpret_cast<amdgpu_bo_handle>(mem_handle);
    return 0;
  } else {
    return -1;
  }
}

HSAKMTAPI int amdgpu_bo_va_op(amdgpu_bo_handle bo,
                              uint64_t offset,
                              uint64_t size,
                              uint64_t addr,
                              uint64_t flags,
                              uint32_t ops) {
  wsl::thunk::GpuMemory *gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory *>(bo);
  assert(gpu_mem != nullptr);

  switch(ops) {
    case AMDGPU_VA_OP_MAP:
      {
        if (gpu_mem->GpuAddress() == addr) {
          pr_info("bo is mapped already\n");
          return 0;
        } else if (gpu_mem->GpuAddress()) {
          pr_err("amdgpu_bo_va_op: GPU memory already mapped at %p, but requested to map at %p\n",
                 reinterpret_cast<void *>(gpu_mem->GpuAddress()), reinterpret_cast<void *>(addr));
          return -1;
        }
        auto code = gpu_mem->MapGpuVirtualAddress(static_cast<gpusize>(addr), size, offset);
        if (code != ErrorCode::Success)
          return -1;

        code = gpu_mem->MakeResident();
        if (code != ErrorCode::Success)
          return -1;
        // Wait on paging fence to ensure map and residency are completed. MakeResident()
        // updates the same paging fence as Map(), hence it's safe to wait for the last one.
        if (!gpu_mem->GetDevice()->WaitOnPagingFenceFromCpu()) {
          return -1;
        }
      }
      break;
    case AMDGPU_VA_OP_UNMAP:
      {
        auto code = gpu_mem->UnmapGpuVirtualAddress(static_cast<gpusize>(addr), size, offset);
        if (code != ErrorCode::Success)
          return -1;
        // Wait on paging fence to ensure unmap is completed. Evict() doesn't
        // update the fence.
        if (!gpu_mem->GetDevice()->WaitOnPagingFenceFromCpu()) {
          return -1;
        }
        gpu_mem->Evict();
      }
      break;
  }
  return 0;
}

HSAKMTAPI int drmCommandWriteRead(int fd, unsigned long drmCommandIndex,
                                  void *data, unsigned long size) {
  return 0;
}

// ================================================================================================
int amdgpu_bo_query_info(amdgpu_bo_handle buf_handle, struct amdgpu_bo_info* info) {
  wsl::thunk::GpuMemory *gpu_mem = reinterpret_cast<wsl::thunk::GpuMemory *>(buf_handle);
  if (gpu_mem == nullptr) {
    return -1;
  }
  info->alloc_size = gpu_mem->Size();
  info->preferred_heap = 0;
  info->phys_alignment = 0;
  info->alloc_flags = 0;
  return 0;
}

// ================================================================================================
int amdgpu_bo_set_metadata(amdgpu_bo_handle buf_handle, struct amdgpu_bo_metadata* info) {
  // Currently, we do not use metadata in WSL
  return 0;
}

