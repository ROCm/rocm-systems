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

#include <errno.h>
#include <libsync.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "virtio_gpu.h"

#define SHMEM_SZ (25 * 0x1000)

static int set_context(int fd) {
  struct drm_virtgpu_context_set_param params[] = {
      {VIRTGPU_CONTEXT_PARAM_CAPSET_ID, VIRGL_RENDERER_CAPSET_HSAKMT},
      {VIRTGPU_CONTEXT_PARAM_NUM_RINGS, 64},
  };
  struct drm_virtgpu_context_init args = {
      .num_params = ARRAY_SIZE(params),
      .ctx_set_params = (uintptr_t)(params),
  };

  return virtio_gpu_ioctl(fd, VIRTGPU_CONTEXT_INIT, &args);
}

int virtio_gpu_map_handle(struct virtio_gpu_device* vgdev, uint32_t handle, uint64_t size,
                          void** addr, void* fixed_map) {
  struct drm_virtgpu_map args = {
      .handle = handle,
  };
  int r;

  r = virtio_gpu_ioctl(vgdev->fd, VIRTGPU_MAP, &args);
  if (r) return r;

  *addr = mmap(fixed_map, size, PROT_READ | PROT_WRITE, MAP_SHARED | (fixed_map ? MAP_FIXED : 0),
               vgdev->fd, args.offset);

  if (*addr == MAP_FAILED) return -EINVAL;

  return 0;
}

void virtio_gpu_unmap(void* addr, uint64_t size) { munmap(addr, size); }

static void virtio_gpu_bo_close(struct virtio_gpu_device* vgdev, uint32_t handle) {
  struct drm_gem_close args = {
      .handle = handle,
  };

  virtio_gpu_ioctl(vgdev->fd, GEM_CLOSE, &args);
}

static int virtio_gpu_shmem_init(struct virtio_gpu_device* vgdev, size_t size) {
  struct drm_virtgpu_resource_create_blob args = {
      .blob_mem = VIRTGPU_BLOB_MEM_HOST3D,
      .blob_flags = VIRTGPU_BLOB_FLAG_USE_MAPPABLE,
      .size = size,
      .blob_id = 0,
  };

  int r = virtio_gpu_ioctl(vgdev->fd, VIRTGPU_RESOURCE_CREATE_BLOB, &args);
  if (r) return r;

  r = virtio_gpu_map_handle(vgdev, args.bo_handle, size, (void**)&vgdev->shmem, NULL);
  if (r) {
    virtio_gpu_bo_close(vgdev, args.bo_handle);
    return r;
  }

  vgdev->shmem_handle = args.bo_handle;

  uint32_t offset = vgdev->shmem->base.rsp_mem_offset;
  vgdev->rsp_mem_len = size - offset;
  vgdev->rsp_mem = &((uint8_t*)vgdev->shmem)[offset];

  return 0;
}

struct virtio_gpu_device* virtio_gpu_init(int fd, uint32_t context_id) {
  struct virtio_gpu_device* vgdev;
  int r;

  r = set_context(fd);

  if (r) return NULL;

  vgdev = calloc(1, sizeof(*vgdev));
  if (!vgdev) return NULL;

  vgdev->fd = fd;

  vgdev->reqbuf = calloc(1, SHMEM_SZ);
  if (!vgdev->reqbuf) {
    free(vgdev);
    return NULL;
  }

  r = virtio_gpu_shmem_init(vgdev, SHMEM_SZ);
  if (r) {
    free(vgdev);
    return NULL;
  }

  pthread_mutex_init(&vgdev->rsp_lock, NULL);
  pthread_mutex_init(&vgdev->eb_lock, NULL);

  return vgdev;
}

void virtio_gpu_close(struct virtio_gpu_device* vgdev) {
  virtio_gpu_unmap(vgdev->shmem, SHMEM_SZ);
  virtio_gpu_bo_close(vgdev, vgdev->shmem_handle);

  pthread_mutex_destroy(&vgdev->rsp_lock);
  pthread_mutex_destroy(&vgdev->eb_lock);

  close(vgdev->fd);
  free(vgdev->reqbuf);
  free(vgdev);
}

void* virtio_gpu_alloc_rsp(struct virtio_gpu_device* vgdev, struct virtio_gpu_ccmd_req* req,
                           uint32_t size) {
  uint32_t off;

  pthread_mutex_lock(&vgdev->rsp_lock);

  size = VHSA_ALIGN_UP(size, 8);

  if ((vgdev->next_rsp_off + size) >= vgdev->rsp_mem_len) vgdev->next_rsp_off = 0;

  off = vgdev->next_rsp_off;
  vgdev->next_rsp_off += size;

  pthread_mutex_unlock(&vgdev->rsp_lock);

  req->rsp_off = off;
  struct virtio_gpu_ccmd_rsp* rsp = (void*)&vgdev->rsp_mem[off];
  rsp->len = size;

  return rsp;
}

static int virtio_gpu_execbuffer_locked(struct virtio_gpu_device* vgdev, void* cmd,
                                        uint32_t cmd_size, uint32_t* handles, uint32_t num_handles,
                                        int* fence_fd, int ring_idx, uint32_t num_in_syncobjs,
                                        uint32_t num_out_syncobjs,
                                        struct drm_virtgpu_execbuffer_syncobj* in_syncobjs,
                                        struct drm_virtgpu_execbuffer_syncobj* out_syncobjs,
                                        bool in_fence, bool out_fence) {
  struct drm_virtgpu_execbuffer eb = {
      .flags = (out_fence ? VIRTGPU_EXECBUF_FENCE_FD_OUT : 0) |
          (in_fence ? VIRTGPU_EXECBUF_FENCE_FD_IN : 0) | VIRTGPU_EXECBUF_RING_IDX,
      .size = cmd_size,
      .command = (uintptr_t)cmd,
      .bo_handles = (uintptr_t)handles,
      .num_bo_handles = num_handles,
      .fence_fd = *fence_fd,
      .ring_idx = ring_idx,
      .syncobj_stride = sizeof(struct drm_virtgpu_execbuffer_syncobj),
      .num_in_syncobjs = num_in_syncobjs,
      .num_out_syncobjs = num_out_syncobjs,
      .in_syncobjs = (uintptr_t)in_syncobjs,
      .out_syncobjs = (uintptr_t)out_syncobjs,
  };
  int r = virtio_gpu_ioctl(vgdev->fd, VIRTGPU_EXECBUFFER, &eb);
  if (r) return r;

  if (out_fence) *fence_fd = eb.fence_fd;

  return 0;
}

static int virtio_gpu_flush_locked(struct virtio_gpu_device* vgdev, int* fence) {
  int r;

  if (!vgdev->reqbuf_len) return 0;

  r = virtio_gpu_execbuffer_locked(vgdev, vgdev->reqbuf, vgdev->reqbuf_len, NULL, 0, fence, 0, 0, 0,
                                   NULL, NULL, false, !!fence);
  if (r) return r;

  vgdev->reqbuf_len = 0;
  vgdev->reqbuf_cnt = 0;

  return 0;
}

static int virtio_gpu_add_cmd(struct virtio_gpu_device* vgdev, struct virtio_gpu_ccmd_req* req) {
  req->seqno = ++vgdev->next_seqno;
  int r;

  if (vgdev->reqbuf_len + req->len > sizeof(vgdev->reqbuf)) {
    r = virtio_gpu_flush_locked(vgdev, NULL);
    if (r) return r;
  }

  memcpy(&vgdev->reqbuf[vgdev->reqbuf_len], req, req->len);
  vgdev->reqbuf_len += req->len;
  vgdev->reqbuf_cnt++;

  return 0;
}

static inline bool fence_before(uint32_t a, uint32_t b) { return (int32_t)(a - b) < 0; }

static void virtio_gpu_seqno_sync(struct virtio_gpu_device* vgdev,
                                  struct virtio_gpu_ccmd_req* req) {
  while (fence_before(vgdev->shmem->base.seqno, req->seqno)) sched_yield();
}

int virtio_gpu_exec_cmd(struct virtio_gpu_device* vgdev, struct virtio_gpu_ccmd_req* req,
                        bool sync) {
  int r = 0;
  int fence;

  pthread_mutex_lock(&vgdev->eb_lock);

  r = virtio_gpu_add_cmd(vgdev, req);

  if (r || !sync) goto out;

  r = virtio_gpu_flush_locked(vgdev, &fence);

out:
  pthread_mutex_unlock(&vgdev->eb_lock);
  if (r) return r;

  if (sync) {
    sync_wait(fence, -1);
    close(fence);
    virtio_gpu_seqno_sync(vgdev, req);
  }

  return r;
}

int virtio_gpu_create_blob(struct virtio_gpu_device* vgdev,
                           struct drm_virtgpu_resource_create_blob* args) {
  return virtio_gpu_ioctl(vgdev->fd, VIRTGPU_RESOURCE_CREATE_BLOB, args);
}

int virtio_gpu_destroy_handle(struct virtio_gpu_device* vgdev, uint32_t bo_handle) {
  struct drm_gem_close args = {
      .handle = bo_handle,
  };

  return virtio_gpu_ioctl(vgdev->fd, GEM_CLOSE, &args);
}

int virtio_gpu_res_id(struct virtio_gpu_device* vgdev, uint32_t handle, uint32_t* res_id) {
  struct drm_virtgpu_resource_info args = {
      .bo_handle = handle,
  };
  int r = virtio_gpu_ioctl(vgdev->fd, VIRTGPU_RESOURCE_INFO, &args);
  if (r) return r;

  *res_id = args.res_handle;
  return 0;
}

static int virtio_gpu_get_capset(int fd, struct virgl_renderer_capset_hsakmt* caps) {
  struct drm_virtgpu_get_caps args = {
      .cap_set_id = VIRGL_RENDERER_CAPSET_HSAKMT,
      .cap_set_ver = 0,
      .addr = (uintptr_t)caps,
      .size = sizeof(*caps),
  };

  memset(caps, 0, sizeof(*caps));

  return virtio_gpu_ioctl(fd, VIRTGPU_GET_CAPS, &args);
}

int virtio_gpu_kfd_open(void) {
  drmDevicePtr devices[VHSA_MAX_DEVICES];
  int num_devices = 0;
  int i, fd, ret;

  num_devices = drmGetDevices2(0, devices, ARRAY_SIZE(devices));
  if (num_devices <= 0) return -1;

  for (i = 0; i < num_devices; i++) {
    fd = open(devices[i]->nodes[DRM_NODE_RENDER], O_RDWR | O_CLOEXEC);
    if (fd < 0) continue;

    struct virgl_renderer_capset_hsakmt caps;
    ret = virtio_gpu_get_capset(fd, &caps);
    if (ret || caps.context_type != VIRTGPU_DRM_CONTEXT_AMDGPU) {
      close(fd);
      fd = -1;
      continue;
    }

    goto out;
  }

out:
  drmFreeDevices(devices, num_devices);
  return fd;
}
