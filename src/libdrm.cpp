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

#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "libhsakmt.h"

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
  return 0;
}

HSAKMTAPI int amdgpu_bo_cpu_map(amdgpu_bo_handle bo, void **cpu) {
  return 0;
}

HSAKMTAPI int amdgpu_bo_free(amdgpu_bo_handle buf_handle) {
  return 0;
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
  HsaGraphicsResourceInfo GraphicsResourceInfo;
  HSAKMT_STATUS ret = hsaKmtImportDMABufHandle(shared_handle, &GraphicsResourceInfo);
  if (ret == HSAKMT_STATUS_SUCCESS) {
    //use GpuMemory object's address as drm buf handle
    output->buf_handle = reinterpret_cast<amdgpu_bo_handle>(GraphicsResourceInfo.MemoryAddress);
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
  switch(ops) {
    case AMDGPU_VA_OP_MAP:
      {
        wsl::thunk::GpuMemory *gpu_mem = get_gpu_mem(bo);
        assert(gpu_mem != nullptr);
        auto code = gpu_mem->MapGpuVirtualAddress(reinterpret_cast<gpusize>(addr), size, offset);
        if (code != ErrorCode::Success)
          return -1;

        code = gpu_mem->MakeResident();
        if (code != ErrorCode::Success)
          return -1;
      }
      break;
    case AMDGPU_VA_OP_UNMAP:
      {
        wsl::thunk::GpuMemory *gpu_mem = get_gpu_mem(bo);
        assert(gpu_mem != nullptr);
        auto code = gpu_mem->UnmapGpuVirtualAddress(reinterpret_cast<gpusize>(addr), size, offset);
        if (code != ErrorCode::Success)
          return -1;
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
