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

/* amdgpu device initialize/deinitialize will be called in vhsakmtopen 
 * so just return ENOSYS here to avoid duplicate implementation
 */
int vamdgpu_device_initialize(int fd, uint32_t* major_version, uint32_t* minor_version,
                              amdgpu_device_handle* device_handle) {
  return -ENOSYS;
}
int vamdgpu_device_deinitialize(amdgpu_device_handle device_handle) {
  return -ENOSYS;
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

int vamdgpu_device_get_fd(amdgpu_device_handle device_handle) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_node* node = NULL;
  int fd = -1;
  
  pthread_mutex_lock(&dev->vhsakmt_mutex);
  for (uint32_t i = 0; i < dev->sys_props->NumNodes; i++) {
    if (dev->vhsakmt_nodes[i].amdgpu_device_handle == device_handle) {
      node = &dev->vhsakmt_nodes[i];
      fd = node->amdgpu_fd;
      break;
    }
  }
  pthread_mutex_unlock(&dev->vhsakmt_mutex);

  return fd;
}

int vdrmCommandWriteRead(int fd, unsigned long drmCommandIndex, void* data, unsigned long size) {
  CHECK_VIRTIO_KFD_OPEN();

  if (size > VHSAKMT_CCMD_QUERY_DRM_CMD_WRITE_READ_MAX_SIZE)
    return -EINVAL;

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req) + size),
      .type = VHSAKMT_CCMD_QUERY_DRM_CMD_WRITE_READ,
      .drm_cmd_write_read_args =
          {
              .fd = fd,
              .drmCommandIndex = drmCommandIndex,
              .size = size,
          },
  };

  memcpy(req.payload, data, size);

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr,
                          sizeof(struct vhsakmt_ccmd_query_info_rsp) + size);
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  memcpy(data, rsp->payload, size);

  return rsp->ret;
}

HSAKMT_STATUS vhsaKmtGetAMDGPUDeviceHandle(HSAuint32 NodeId, HsaAMDGPUDeviceHandle* DeviceHandle) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_node* node = vhsakmt_get_node_by_id(dev, NodeId);
  if (!node) return HSAKMT_STATUS_INVALID_HANDLE;

  if (node->amdgpu_device_handle) {
    *DeviceHandle = node->amdgpu_device_handle;
    return HSAKMT_STATUS_SUCCESS;
  }

  pthread_mutex_lock(&dev->vhsakmt_mutex);
  if (node->amdgpu_device_handle) {
    *DeviceHandle = node->amdgpu_device_handle;
    pthread_mutex_unlock(&dev->vhsakmt_mutex);
    return HSAKMT_STATUS_SUCCESS;
  }

  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_AMDGPU_DEVICE_HANDLE,
      .NodeID = NodeId,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_query_info_rsp));
  if (!rsp) {
    pthread_mutex_unlock(&dev->vhsakmt_mutex);
    return -ENOMEM;
  }

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  
  node->amdgpu_device_handle = (void*)rsp->device_handle_rsp.amdgpu_device_handle;
  node->amdgpu_fd = (int)rsp->device_handle_rsp.fd;
  pthread_mutex_unlock(&dev->vhsakmt_mutex);

  *DeviceHandle = node->amdgpu_device_handle;
  return rsp->ret;
}
