/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "hsakmt/hsakmt_virtio.h"
#include "hsakmt_virtio_device.h"

int vamdgpu_device_initialize(int fd, uint32_t* major_version, uint32_t* minor_version,
                              amdgpu_device_handle* device_handle) {
  return 0;
}
int vamdgpu_device_deinitialize(amdgpu_device_handle device_handle) {
  return 0;
}

int vamdgpu_query_gpu_info(amdgpu_device_handle handle, void* out) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_GPU_INFO,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) return -ENOMEM;

  int ret = vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  if (!ret) memcpy(out, &rsp->gpu_info, sizeof(struct amdgpu_gpu_info));

  return ret;
}

int vamdgpu_bo_cpu_map(amdgpu_bo_handle buf_handle, void** cpu) {
  return 0;
}

int vamdgpu_bo_free(amdgpu_bo_handle buf_handle) {
  return 0;
}

int vamdgpu_bo_export(amdgpu_bo_handle buf_handle, enum amdgpu_bo_handle_type type,
                      uint32_t* shared_handle) {
  return 0;
}

int vamdgpu_bo_import(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type,
                      uint32_t shared_handle, struct amdgpu_bo_import_result* output) {
  return 0;
}

int vamdgpu_bo_va_op(amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr,
                     uint64_t flags, uint32_t ops) {
  return 0;
}

int vdrmCommandWriteRead(int fd, unsigned long drmCommandIndex, void* data, unsigned long size) {
  return 0;
}

HSAKMT_STATUS vhsaKmtGetAMDGPUDeviceHandle(HSAuint32 NodeId, HsaAMDGPUDeviceHandle* DeviceHandle) {
  CHECK_VIRTIO_KFD_OPEN();

  return HSAKMT_STATUS_SUCCESS;
}
