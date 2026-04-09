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

int vamdgpu_bo_cpu_map(amdgpu_bo_handle buf_handle, void** cpu) {
  return 0;
}

int vamdgpu_bo_free(amdgpu_bo_handle buf_handle) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle vdev = vhsakmt_dev();
  vhsakmt_bo_handle vbo = (vhsakmt_bo_handle)buf_handle;

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_AMDGPU_BO_FREE,
      .buf_handle = (uint64_t)buf_handle,
      .res_id = vbo->real.res_id,
  };

  rsp = vhsakmt_alloc_rsp(vdev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(vdev, &req.hdr, __FUNCTION__);

  pthread_mutex_lock(&vbo->amdgpu_bo.lock);
  if (vbo->amdgpu_bo.imported) {
    if (vhsakmt_atomic_dec_return(&vbo->amdgpu_bo.refcount) > 0) {
      pthread_mutex_unlock(&vbo->amdgpu_bo.lock);
      return HSAKMT_STATUS_SUCCESS;
    }
    vbo->amdgpu_bo.import_size = 0;
    vbo->amdgpu_bo.imported = false;
    vbo->bo_type &= (uint32_t)~VHSA_BO_AMDGPU;
  }
  pthread_mutex_unlock(&vbo->amdgpu_bo.lock);

  return rsp->ret;
}

int vamdgpu_bo_export(amdgpu_bo_handle buf_handle, enum amdgpu_bo_handle_type type,
                      uint32_t* shared_handle) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle vdev = vhsakmt_dev();
  vhsakmt_bo_handle bo = (vhsakmt_bo_handle)buf_handle;

  if (type != amdgpu_bo_handle_type_kms) {
    vhsa_err("%s: unsupported export type: %u\n", __FUNCTION__, type);
    return -EINVAL;
  }

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_AMDGPU_EXPORT,
      .res_id = bo->real.res_id,
      .amdgpu_export_args =
          {
              .buf_handle = (uint64_t)buf_handle,
              .type = (uint32_t)type,
          },
  };

  rsp = vhsakmt_alloc_rsp(vdev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(vdev, &req.hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  *shared_handle = rsp->shared_handle;

  return rsp->ret;
}

static vhsakmt_bo_handle vhsakmt_bo_from_resid(vhsakmt_device_handle dev, uint32_t res_id) {
  vhsakmt_bo_handle bo;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_MAP_USERPTR,
      .res_id = res_id,
  };
  struct vhsakmt_ccmd_memory_rsp* rsp =
      vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return NULL;

  rsp->map_userptr_rsp.userptr_handle = 0;
  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  bo = vhsakmt_find_bo_by_addr(dev, (void*)rsp->map_userptr_rsp.userptr_handle);

  return bo;
}

int vamdgpu_bo_import(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type,
                      uint32_t shared_handle, struct amdgpu_bo_import_result* output) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle vdev = vhsakmt_dev();
  vhsakmt_bo_handle obj;
  uint32_t bo_handle, res_id;
  int r;

  if (type != amdgpu_bo_handle_type_dma_buf_fd) {
    vhsa_err("%s: unsupported import type: %u\n", __FUNCTION__, type);
    return -EINVAL;
  }

  r = vhsakmt_handle_to_resid(vdev, shared_handle, &res_id, &bo_handle);
  if (r) return r;

  obj = vhsakmt_bo_from_resid(vdev, res_id);
  if (!obj) return HSAKMT_STATUS_INVALID_HANDLE;

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_AMDGPU_IMPORT,
      .res_id = res_id,
      .amdgpu_import_args =
          {
              .dev = (int64_t)dev,
              .type = (uint32_t)type,
              .shared_handle = shared_handle,
          },
  };

  rsp = vhsakmt_alloc_rsp(vdev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(vdev, &req.hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  pthread_mutex_lock(&obj->amdgpu_bo.lock);
  if (obj->amdgpu_bo.imported) {
    vhsa_debug("%s: bo already imported for shared_handle: %u\n", __FUNCTION__, shared_handle);
    vhsakmt_atomic_inc(&obj->amdgpu_bo.refcount);
    output->alloc_size = obj->amdgpu_bo.import_size;
    output->buf_handle = (amdgpu_bo_handle)obj;
    pthread_mutex_unlock(&obj->amdgpu_bo.lock);
    return HSAKMT_STATUS_SUCCESS;
  }

  memcpy(output, &rsp->amdgpu_import_rsp.output, sizeof(struct amdgpu_bo_import_result));

  obj->bo_type |= VHSA_BO_AMDGPU;
  obj->amdgpu_bo.imported = true;
  obj->amdgpu_bo.import_size = output->alloc_size;
  atomic_store(&obj->amdgpu_bo.refcount, 1);
  pthread_mutex_unlock(&obj->amdgpu_bo.lock);

  output->buf_handle = (amdgpu_bo_handle)obj;

  return rsp->ret;
}

int vamdgpu_bo_va_op(amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr,
                     uint64_t flags, uint32_t ops) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle vdev = vhsakmt_dev();
  vhsakmt_bo_handle vbo = (vhsakmt_bo_handle)bo;

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_AMDGPU_VA_OP,
      .res_id = vbo->real.res_id,
      .amdgpu_va_op_args =
          {
              .bo = (uint64_t)bo,
              .offset = offset,
              .size = size,
              .addr = addr,
              .flags = flags,
              .ops = ops,
          },
  };

  rsp = vhsakmt_alloc_rsp(vdev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(vdev, &req.hdr, __FUNCTION__);
  return rsp->ret;
}

int vamdgpu_bo_query_info(amdgpu_bo_handle bo, struct amdgpu_bo_info* info) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle vdev = vhsakmt_dev();
  vhsakmt_bo_handle vbo = (vhsakmt_bo_handle)bo;

  if (!(vbo->bo_type & VHSA_BO_AMDGPU)) return -EINVAL;

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_AMDGPU_BO_QUERY_INFO,
      .res_id = vbo->real.res_id,
  };

  rsp = vhsakmt_alloc_rsp(vdev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(vdev, &req.hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  memcpy(info, &rsp->query_bo_info, sizeof(struct amdgpu_bo_info));

  return rsp->ret;
}

int vamdgpu_bo_set_metadata(amdgpu_bo_handle bo, struct amdgpu_bo_metadata* info) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle vdev = vhsakmt_dev();
  vhsakmt_bo_handle vbo = (vhsakmt_bo_handle)bo;

  if (!(vbo->bo_type & VHSA_BO_AMDGPU)) return -EINVAL;

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(
          MEMORY, sizeof(struct vhsakmt_ccmd_memory_req) + sizeof(struct amdgpu_bo_metadata)),
      .type = VHSAKMT_CCMD_MEMORY_AMDGPU_BO_SET_METADATA,
      .res_id = vbo->real.res_id,
      .amdgpu_bo_metadata = *info,
  };
  rsp = vhsakmt_alloc_rsp(vdev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(vdev, &req.hdr, __FUNCTION__);
  return rsp->ret;
}
