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

#ifndef VIRTIO_GPU_H
#define VIRTIO_GPU_H

#include <pthread.h>
#include <stdint.h>
#include <xf86drm.h>

#include "virtgpu_drm.h"

#define VIRGL_RENDERER_CAPSET_HSAKMT 8
#define VIRTGPU_DRM_CONTEXT_AMDGPU 1
#define VHSA_MAX_DEVICES 10

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#define VHSA_ALIGN_UP(x, align) (((uint64_t)(x) + (align)-1) & ~(uint64_t)((align)-1))
#define VHSA_ALIGN_DOWN(x, align) ((uint64_t)(x) & ~(uint64_t)((align)-1))

#define virtio_gpu_ioctl(fd, name, args)                                                           \
  ({                                                                                               \
    int ret = drmIoctl((fd), DRM_IOCTL_##name, (args));                                            \
    ret;                                                                                           \
  })

struct virgl_renderer_capset_hsakmt {
  uint32_t wire_format_version;
  /* Underlying drm device version: */
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t version_patchlevel;
  uint32_t context_type;
  uint32_t pad;
};

struct virtio_gpu_shmem_base {
  uint32_t seqno;
  uint32_t rsp_mem_offset;
};

struct virtio_gpu_ccmd_req {
  uint32_t cmd;
  uint32_t len;
  uint32_t seqno;
  uint32_t rsp_off;
};

struct virtio_gpu_ccmd_rsp {
  uint32_t len;
};

struct virtio_gpu_shmem {
  struct virtio_gpu_shmem_base base;
  uint32_t async_error;
  uint32_t global_faults;
};

#define vhsakmt_shmem virtio_gpu_shmem
#define vhsakmt_ccmd_req virtio_gpu_ccmd_req
#define vhsakmt_ccmd_rsp virtio_gpu_ccmd_rsp

struct virtio_gpu_device {
  int fd;

  struct virtio_gpu_shmem* shmem;
  uint32_t shmem_handle;

  uint8_t* rsp_mem;
  uint32_t rsp_mem_len;
  uint32_t next_rsp_off;
  pthread_mutex_t rsp_lock;
  pthread_mutex_t eb_lock;

  uint32_t next_seqno;
  uint32_t reqbuf_len;
  uint32_t reqbuf_cnt;
  uint8_t* reqbuf;
};

struct virtio_gpu_device* virtio_gpu_init(int fd, uint32_t context_id);
void virtio_gpu_close(struct virtio_gpu_device* vgdev);
int virtio_gpu_exec_cmd(struct virtio_gpu_device* vgdev, struct virtio_gpu_ccmd_req* req,
                        bool sync);
void* virtio_gpu_alloc_rsp(struct virtio_gpu_device* vgdev, struct virtio_gpu_ccmd_req* req,
                           uint32_t size);
int virtio_gpu_map_handle(struct virtio_gpu_device* vgdev, uint32_t handle, uint64_t size,
                          void** addr, void* fixed_map);
void virtio_gpu_unmap(void* addr, uint64_t size);
int virtio_gpu_create_blob(struct virtio_gpu_device* vgdev,
                           struct drm_virtgpu_resource_create_blob* args);
int virtio_gpu_destroy_handle(struct virtio_gpu_device* vgdev, uint32_t bo_handle);
int virtio_gpu_res_id(struct virtio_gpu_device* vgdev, uint32_t handle, uint32_t* res_id);
int virtio_gpu_kfd_open(void);

#endif /* VIRTIO_GPU_H */
