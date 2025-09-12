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

#define VHSA_GL_METADATA_MAX_SIZE (0x50)

vhsakmt_bo_handle vhsakmt_entry_to_bo_handle(bo_entry e) { return (vhsakmt_bo_handle)e; }
bo_entry vhsakmt_bo_handle_to_entry(vhsakmt_bo_handle bo) { return &bo->rbtn; }
static inline bool vhsakmt_is_mem_bo(vhsakmt_bo_handle bo) { return (!bo->queue_id && !bo->event); }

static bool vhsakmt_mappable(HsaMemFlags flags) { return (!flags.ui32.Scratch); }

static bool vhsakmt_bo_mappable(vhsakmt_bo_handle bo) { return vhsakmt_mappable(bo->flags); }

void vhsakmt_insert_bo(vhsakmt_device_handle dev, vhsakmt_bo_handle bo, void* addr, uint64_t size) {
  bo->rbtn.key.addr = (unsigned long)addr;
  bo->rbtn.key.size = (unsigned long)size;

  pthread_mutex_lock(&dev->bo_handles_mutex);
  hsakmt_rbtree_insert(&dev->bo_rbt, &bo->rbtn);
  pthread_mutex_unlock(&dev->bo_handles_mutex);
}

static void vhsakmt_remove_entry(vhsakmt_device_handle dev, bo_entry entry) {
  if (!entry) return;

  pthread_mutex_lock(&dev->bo_handles_mutex);
  hsakmt_rbtree_delete(&dev->bo_rbt, entry);
  pthread_mutex_unlock(&dev->bo_handles_mutex);
}

void vhsakmt_remove_bo(vhsakmt_device_handle dev, vhsakmt_bo_handle bo) {
  bo_entry entry = vhsakmt_bo_handle_to_entry(bo);
  if (entry->key.addr == 0 && entry->key.size == 0) return;

  vhsakmt_remove_entry(dev, entry);
}

static bo_entry vhsakmt_rbt_search(vhsakmt_device_handle dev, void* addr) {
  vhsakmt_bo_handle bo;

  rbtree_key_t key = rbtree_key((uint64_t)addr, 0);
  pthread_mutex_lock(&dev->bo_handles_mutex);
  bo_entry n = rbtree_lookup_nearest(&dev->bo_rbt, &key, LKP_ADDR, RIGHT);
  pthread_mutex_unlock(&dev->bo_handles_mutex);
  if (n) {
    bo = vhsakmt_entry_to_bo_handle(n);
    if (bo->cpu_addr != addr) return NULL;
    return n;
  }

  return NULL;
}

static bo_entry vhsakmt_find_entry_by_addr(vhsakmt_device_handle dev, void* addr) {
  return vhsakmt_rbt_search(dev, addr);
}

vhsakmt_bo_handle vhsakmt_find_bo_by_addr(vhsakmt_device_handle dev, void* addr) {
  bo_entry entry = vhsakmt_find_entry_by_addr(dev, addr);

  if (entry) {
    vhsakmt_bo_handle bo = vhsakmt_entry_to_bo_handle(entry);
    if (!vhsakmt_is_mem_bo(bo)) return NULL;

    return bo;
  }

  return NULL;
}

void* vhsakmt_gpu_va(vhsakmt_device_handle dev, void* va) {
  if (!vhsakmt_is_userptr(dev, va)) return va;

  bo_entry entry = vhsakmt_find_entry_by_addr(dev, va);

  if (!entry) return NULL;

  return vhsakmt_entry_to_bo_handle(entry)->host_addr;
}

int vhsakmt_bo_cpu_map(vhsakmt_bo_handle bo, void** cpu, void* fixed_cpu) {
  int r;

  if (!vhsakmt_bo_mappable(bo)) return 0;

  pthread_mutex_lock(&bo->map_mutex);

  if (!bo->cpu_addr) {
    r = virtio_gpu_map_handle(bo->dev->vgdev, bo->real.handle, bo->size, cpu, fixed_cpu);
    if (r) {
      pthread_mutex_unlock(&bo->map_mutex);
      return r;
    }
    bo->cpu_addr = *cpu;
    atomic_fetch_add(&bo->real.map_count, 1);
  }
  pthread_mutex_unlock(&bo->map_mutex);

  return *cpu == MAP_FAILED;
}

int vhsakmt_bo_cpu_unmap(vhsakmt_bo_handle bo) {
  int r = 0;

  if (!vhsakmt_bo_mappable(bo)) return 0;

  pthread_mutex_lock(&bo->map_mutex);

  if (!bo->cpu_addr || bo->real.map_count == 0) {
    pthread_mutex_unlock(&bo->map_mutex);
    return 0;
  }

  if (vhsakmt_atomic_dec_return(&bo->real.map_count) <= 0) {
    if (bo->bo_type & VHSA_BO_KFD_MEM) {
      virtio_gpu_unmap(bo->cpu_addr, bo->size);
      vhsakmt_reserve_va(VHSA_VPTR_TO_UINT64(bo->cpu_addr), bo->size);
      bo->cpu_addr = NULL;
    }
  }

  pthread_mutex_unlock(&bo->map_mutex);
  return r;
}

static int vhsakmt_destroy_handle(vhsakmt_device_handle dev, vhsakmt_bo_handle bo) {
  int r = virtio_gpu_destroy_handle(dev->vgdev, bo->real.handle);
  free(bo);

  return r;
}

int vhsakmt_init_host_blob(vhsakmt_device_handle dev, size_t size, uint32_t blob_type,
                           uint32_t blob_flag, uint32_t blob_id, uint32_t bo_type, void* va_handle,
                           vhsakmt_bo_handle* bo_handle) {
  int r;
  vhsakmt_bo_handle bo;
  struct drm_virtgpu_resource_create_blob args = {
      .blob_mem = blob_type,
      .size = size,
      .blob_id = blob_id,
      .blob_flags = blob_flag,
  };

  r = virtio_gpu_create_blob(dev->vgdev, &args);
  if (r) return -EINVAL;

  bo = calloc(1, sizeof(struct vhsakmt_bo));
  if (!bo) {
    virtio_gpu_destroy_handle(dev->vgdev, args.bo_handle);
    return -ENOMEM;
  }

  bo->dev = dev;
  bo->size = size;
  bo->real.alloc_size = size;
  bo->bo_type = bo_type;
  bo->host_addr = va_handle;
  pthread_mutex_init(&bo->map_mutex, NULL);
  atomic_store(&bo->real.map_count, 0);
  atomic_store(&bo->refcount, 1);
  bo->real.handle = args.bo_handle;

  virtio_gpu_res_id(dev->vgdev, bo->real.handle, &bo->real.res_id);

  *bo_handle = bo;
  return 0;
}

static int vhsakmt_init_userptr_blob(vhsakmt_device_handle dev, void* addr, size_t size,
                                     vhsakmt_bo_handle* bo_handle, uint64_t* offset) {
  int r;
  struct drm_virtgpu_resource_create_blob args = {
      .blob_mem = VIRTGPU_BLOB_MEM_HOST3D_GUEST,
      .blob_flags = VIRTGPU_BLOB_FLAG_USE_USERPTR,
      .size = size,
      .blob_id = vhsakmt_atomic_inc_return(&dev->next_blob_id),
      .blob_userptr = (uint64_t)addr,
  };

  r = virtio_gpu_create_blob(dev->vgdev, &args);
  if (r < 0) return r;

  vhsakmt_bo_handle userptr = calloc(1, sizeof(struct vhsakmt_bo));
  if (!userptr) {
    virtio_gpu_destroy_handle(dev->vgdev, args.bo_handle);
    return -ENOMEM;
  }

  userptr->dev = dev;
  userptr->size = size;
  userptr->real.alloc_size = size;
  userptr->bo_type = VHSA_BO_USERPTR;
  userptr->cpu_addr = addr;
  pthread_mutex_init(&userptr->map_mutex, NULL);
  atomic_store(&userptr->real.map_count, 0);
  atomic_store(&userptr->refcount, 1);
  userptr->real.handle = args.bo_handle;

  virtio_gpu_res_id(dev->vgdev, userptr->real.handle, &userptr->real.res_id);

  *bo_handle = userptr;
  *offset = args.offset;
  return r;
}

int vhsakmt_create_mappable_blob_bo(vhsakmt_device_handle dev, size_t size, uint32_t blob_id,
                                    uint32_t bo_type, void* va_handle,
                                    vhsakmt_bo_handle* bo_handle) {
  int r;

  r = vhsakmt_init_host_blob(dev, size, VIRTGPU_BLOB_MEM_HOST3D, VIRTGPU_BLOB_FLAG_USE_MAPPABLE,
                             blob_id, bo_type, va_handle, bo_handle);
  if (r) return r;

  r = vhsakmt_bo_cpu_map(*bo_handle, &((*bo_handle)->cpu_addr), va_handle);
  if (r) {
    free(*bo_handle);
    *bo_handle = NULL;
    return -EINVAL;
  }

  if (va_handle && (va_handle != (*bo_handle)->cpu_addr))
    vhsa_warn("%s: target map: %p != real map: %p\n", __FUNCTION__, va_handle,
              (*bo_handle)->cpu_addr);

  vhsakmt_insert_bo(dev, *bo_handle, (*bo_handle)->cpu_addr, (*bo_handle)->size);
  return r;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtAllocMemory(HSAuint32 PreferredNode, HSAuint64 SizeInBytes,
                                           HsaMemFlags MemFlags, void** MemoryAddress) {
  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_memory_rsp* rsp;
  vhsakmt_bo_handle bo;
  int r;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_ALLOC,
      .blob_id = vhsakmt_atomic_inc_return(&dev->next_blob_id),
      .alloc_args =
          {
              .PreferredNode = PreferredNode,
              .SizeInBytes = SizeInBytes,
              .MemFlags = MemFlags,
          },
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  if (!rsp->memory_handle) return -ENOMEM;

  r = vhsakmt_init_host_blob(dev, SizeInBytes, VIRTGPU_BLOB_MEM_HOST3D,
                             vhsakmt_mappable(MemFlags) ? VIRTGPU_BLOB_FLAG_USE_MAPPABLE : 0,
                             req.blob_id, VHSA_BO_KFD_MEM, (void*)rsp->memory_handle, &bo);
  if (r) return r;

  if (!vhsakmt_mappable(MemFlags)) {
    bo->cpu_addr = bo->host_addr;
    if (MemFlags.ui32.Scratch) {
      vhsakmt_set_scratch_area(dev, PreferredNode, (uint64_t)bo->cpu_addr, SizeInBytes);
      bo->bo_type |= VHSA_BO_SCRATCH;
    }
  } else {
    r = vhsakmt_bo_cpu_map(bo, &bo->cpu_addr, bo->host_addr);
    if (r) {
      free(bo);
      return -ENOMEM;
    }
  }

  if (!MemFlags.ui32.Scratch) vhsakmt_insert_bo(dev, bo, bo->cpu_addr, bo->size);

  *MemoryAddress = bo->cpu_addr;

  vhsa_debug("alloc mem addr: %p, host addr: %p, size: %lx, res-id: %d, handble: %d\n",
             *MemoryAddress, bo->host_addr, SizeInBytes, bo->real.res_id, bo->real.handle);

  return rsp->ret;
}

int vhsakmt_bo_free(vhsakmt_device_handle dev, vhsakmt_bo_handle bo) {
  bo_entry entry;
  int r;

  if (vhsakmt_atomic_dec_return(&bo->refcount) > 0) return 0;

  entry = vhsakmt_bo_handle_to_entry(bo);
  if (entry->key.addr == 0 && entry->key.size == 0) return -EINVAL;

  /* do not free BOs of queue, let them be freed with queue */
  if (bo->bo_type & VHSA_BO_QUEUE_DOORBELL) {
    vhsa_err("%s: Try to free VHSA_BO_QUEUE_DOORBELL memory: %p\n", __FUNCTION__, bo->cpu_addr);
    return 0;
  }

  vhsakmt_remove_bo(dev, bo);

  if (bo->cpu_addr) vhsakmt_bo_cpu_unmap(bo);

  if (bo->event) free(bo->event);

  if (bo->gl_meta_data) free(bo->gl_meta_data);

  pthread_mutex_destroy(&bo->map_mutex);

  r = vhsakmt_destroy_handle(dev, bo);

  return r;
}

/* Only remove bo in rbtree */
static void vhsakmt_remove_userptr_bo(vhsakmt_device_handle dev, vhsakmt_bo_handle bo) {
  vhsakmt_remove_bo(dev, bo);
  free(bo);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtFreeMemory(void* MemoryAddress, HSAuint64 SizeInBytes) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  vhsakmt_bo_handle bo = vhsakmt_find_bo_by_addr(dev, MemoryAddress);
  if (!bo) return HSAKMT_STATUS_SUCCESS;

  vhsa_debug("%s: addr: %p, size: %lx, res_id: %d\n", __FUNCTION__, MemoryAddress, SizeInBytes,
             bo->real.res_id);

  return vhsakmt_bo_free(dev, bo);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtMapMemoryToGPUNodes(void* MemoryAddress, HSAuint64 MemorySizeInBytes,
                                                   HSAuint64* AlternateVAGPU,
                                                   HsaMemMapFlags MemMapFlags,
                                                   HSAuint64 NumberOfNodes, HSAuint32* NodeArray) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  size_t req_len =
      VHSA_ALIGN_UP(sizeof(struct vhsakmt_ccmd_memory_req) + NumberOfNodes * sizeof(*NodeArray), 8);
  struct vhsakmt_ccmd_memory_req* req;
  struct vhsakmt_ccmd_memory_rsp* rsp;
  vhsakmt_bo_handle bo;

  req = (void*)calloc(1, req_len);
  if (!req) return -ENOMEM;
  req->hdr = VHSAKMT_CCMD(MEMORY, req_len);
  req->type = VHSAKMT_CCMD_MEMORY_MAP_TO_GPU_NODES;
  req->map_to_GPU_nodes_args.MemorySizeInBytes = MemorySizeInBytes;
  req->map_to_GPU_nodes_args.MemMapFlags = MemMapFlags;
  req->map_to_GPU_nodes_args.NumberOfNodes = NumberOfNodes;

  bo = vhsakmt_find_bo_by_addr(dev, MemoryAddress);
  if (bo) {
    req->map_to_GPU_nodes_args.MemoryAddress = (uint64_t)bo->host_addr;
    if (bo->bo_type & VHSA_BO_USERPTR) vhsakmt_remove_userptr_bo(dev, bo);
  } else
    req->map_to_GPU_nodes_args.MemoryAddress = (uint64_t)MemoryAddress;

  memcpy(req->payload, NodeArray, NumberOfNodes * sizeof(*NodeArray));

  rsp = vhsakmt_alloc_rsp(dev, &req->hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) {
    free(req);
    return -ENOMEM;
  }

  vhsakmt_execbuf_cpu(dev, &req->hdr, __FUNCTION__);

  *AlternateVAGPU = rsp->alternate_vagpu;

  vhsa_debug("%s: gva: %p, hva: 0x%lx, size: %lx, AlternateVAGPU: %lx, ret: %d\n", __FUNCTION__,
             MemoryAddress, req->map_to_GPU_nodes_args.MemoryAddress, MemorySizeInBytes,
             *AlternateVAGPU, rsp->ret);

  free(req);
  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtUnmapMemoryToGPU(void* MemoryAddress) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  vhsakmt_bo_handle bo = vhsakmt_find_bo_by_addr(dev, MemoryAddress);
  if (!bo) return HSAKMT_STATUS_SUCCESS;

  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_UNMAP_TO_GPU,
      .MemoryAddress = (uint64_t)bo->host_addr,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  vhsa_debug("%s: gva: %p, hva: 0x%lx\n", __FUNCTION__, MemoryAddress, req.MemoryAddress);

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtAvailableMemory(HSAuint32 Node, HSAuint64* AvailableBytes) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_AVAIL_MEM,
      .Node = Node,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  *AvailableBytes = rsp->available_bytes;

  return rsp->ret;
}

static int vhsakmt_create_scratch_map_memory(vhsakmt_device_handle dev, void* MemoryAddress,
                                             HSAuint64 MemorySizeInBytes,
                                             HSAuint64* AlternateVAGPU) {
  vhsakmt_bo_handle out;
  int r;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_MAP_MEM_TO_GPU,
      .blob_id = vhsakmt_atomic_inc_return(&dev->next_blob_id),
      .map_to_GPU_args =
          {
              .MemoryAddress = (uint64_t)MemoryAddress,
              .MemorySizeInBytes = MemorySizeInBytes,
              .need_create_bo = true,
          },
  };

  struct vhsakmt_ccmd_memory_rsp* rsp =
      vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  if (rsp->ret) return rsp->ret;

  r = vhsakmt_init_host_blob(dev, MemorySizeInBytes, VIRTGPU_BLOB_MEM_HOST3D, 0, req.blob_id,
                             VHSA_BO_SCRATCH_MAP, NULL, &out);
  if (r) return r;

  // TODO: insert scratch bo into rbtree, or insert it in dev nodes.

  out->cpu_addr = MemoryAddress;
  out->host_addr = (void*)rsp->memory_handle;
  *AlternateVAGPU = rsp->alternate_vagpu;

  vhsa_debug(
      "%s: create scratch memory, gva: %p, memory_handle: 0x%p, alternate_vagpu: %p, size: %lx\n",
      __FUNCTION__, MemoryAddress, (void*)rsp->memory_handle, (void*)rsp->alternate_vagpu,
      MemorySizeInBytes);

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtMapMemoryToGPU(void* MemoryAddress, HSAuint64 MemorySizeInBytes,
                                              HSAuint64* AlternateVAGPU) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_memory_rsp* rsp;
  vhsakmt_bo_handle bo = vhsakmt_find_bo_by_addr(dev, MemoryAddress);
  if (!bo && vhsakmt_is_scratch_mem(dev, MemoryAddress))
    return vhsakmt_create_scratch_map_memory(dev, MemoryAddress, MemorySizeInBytes, AlternateVAGPU);

  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_MAP_MEM_TO_GPU,
      .map_to_GPU_args =
          {
              .MemoryAddress = bo ? (uint64_t)bo->host_addr : (uint64_t)MemoryAddress,
              .MemorySizeInBytes = MemorySizeInBytes,
          },
  };

  if (bo && (bo->bo_type & VHSA_BO_USERPTR)) vhsakmt_remove_userptr_bo(dev, bo);

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  vhsa_debug("%s: gva: %p, hva: 0x%lx, size: %lx\n", __FUNCTION__, MemoryAddress, req.MemoryAddress,
             MemorySizeInBytes);

  *AlternateVAGPU = rsp->alternate_vagpu;

  return rsp->ret;
}

static int vhsakmt_map_userptr(vhsakmt_device_handle dev, void* addr, size_t size, uint32_t res_id,
                               uint64_t* userptr_handle) {
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_MAP_USERPTR,
      .res_id = res_id,
  };
  struct vhsakmt_ccmd_memory_rsp* rsp =
      vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  rsp->map_userptr_rsp.userptr_handle = 0;
  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  *userptr_handle = rsp->map_userptr_rsp.userptr_handle;
  return rsp->ret;
}

static void* vhsakmt_map_to_gpu(void* addr, size_t size) {
  vhsakmt_device_handle dev = vhsakmt_dev();
  size_t offset = (uint64_t)addr % getpagesize();
  size_t map_size = (VHSA_ALIGN_UP(size + offset, getpagesize()) / getpagesize()) * getpagesize();
  uint64_t userptr_offset, userptr_handle = 0;
  vhsakmt_bo_handle userptr;
  int r;

  vhsa_debug("%s: addr: %p, size: 0x%lx, size + offset: 0x%lx, map_size: 0x%lx\n", __FUNCTION__,
             addr, size, size + offset, map_size);

  r = vhsakmt_init_userptr_blob(dev, addr, size, &userptr, &userptr_offset);
  if (r < 0) {
    vhsa_debug("%s: userptr create failed at address: %p, ret = %d\n", __FUNCTION__, addr, r);
    return NULL;
  }

  vhsakmt_map_userptr(dev, addr, size, userptr->real.res_id, &userptr_handle);
  if (!userptr_handle) {
    vhsa_debug("%s: map userptr failed at address: %p, ret = %d\n", __FUNCTION__, addr, r);
    vhsakmt_destroy_handle(dev, userptr);
    vhsakmt_remove_userptr_bo(dev, userptr);
    return NULL;
  }
  userptr->host_addr = VHSA_UINT64_TO_VPTR(VHSA_VPTR_TO_UINT64(userptr_handle) + offset);

  if (r > 0) {
    vhsa_debug("%s: userptr: %p already registered, offset: %lx\n", __FUNCTION__, addr,
               userptr_offset);
    userptr->host_addr =
        VHSA_UINT64_TO_VPTR(VHSA_VPTR_TO_UINT64(userptr->host_addr) + userptr_offset);
  }
  vhsakmt_insert_bo(dev, userptr, userptr->cpu_addr, userptr->size);

  vhsa_debug("%s: real gva: %p, gva: %p, hva: %p, size: %lx, offset: %" PRIu64
             ", map_size: 0x%lx\n",
             __FUNCTION__, addr, userptr->cpu_addr, userptr->host_addr, size, offset, map_size);
  return userptr->host_addr;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterMemoryWithFlags(void* MemoryAddress,
                                                       HSAuint64 MemorySizeInBytes,
                                                       HsaMemFlags MemFlags) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_memory_rsp* rsp;
  void* addr;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_REG_MEM_WITH_FLAG,
      .reg_mem_with_flag =
          {
              .MemorySizeInBytes = MemorySizeInBytes,
              .MemFlags = MemFlags,
          },
  };

  /* no need to register memory from lihsakmt / not a userptr */
  if (!vhsakmt_is_userptr(dev, MemoryAddress)) return HSAKMT_STATUS_SUCCESS;

  addr = vhsakmt_map_to_gpu(MemoryAddress, MemorySizeInBytes);
  if (!addr) {
    vhsa_debug("%s: register memory failed, gva: %p, size: %lx\n", __FUNCTION__, MemoryAddress,
               MemorySizeInBytes);
    return HSAKMT_STATUS_ERROR;
  }

  req.reg_mem_with_flag.MemoryAddress = (uint64_t)addr;

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  return rsp->ret;
}

static int vhsakmt_remove_clgl_bo(vhsakmt_device_handle dev, vhsakmt_bo_handle bo) {
  struct vhsakmt_ccmd_memory_rsp* rsp;
  struct vhsakmt_ccmd_memory_req req = {
      .hdr = VHSAKMT_CCMD(MEMORY, sizeof(struct vhsakmt_ccmd_memory_req)),
      .type = VHSAKMT_CCMD_MEMORY_DEREG_MEM,
      .res_id = bo->real.res_id,
      .MemoryAddress = (uint64_t)bo->cpu_addr,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_memory_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (rsp->ret) vhsa_err("%s: deregister failed clgl memory gva: %p\n", __FUNCTION__, bo->cpu_addr);

  vhsakmt_bo_free(dev, bo);

  vhsa_debug("%s: deregister clgl memory gva: %p, ret: %d\n", __FUNCTION__, bo->cpu_addr, rsp->ret);
  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtDeregisterMemory(void* MemoryAddress) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  vhsakmt_bo_handle bo = vhsakmt_find_bo_by_addr(dev, MemoryAddress);
  if (!bo) return HSAKMT_STATUS_SUCCESS;

  vhsa_debug("%s: remove userptr %p size: 0x%lx, res id: %d\n", __FUNCTION__, MemoryAddress,
             (size_t)bo->size, bo->real.res_id);

  if (bo->bo_type & VHSA_BO_CLGL)
    return vhsakmt_remove_clgl_bo(dev, bo);
  else {
    vhsakmt_remove_bo(dev, bo);
    free(bo);
  }

  return 0;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtQueryPointerInfo(const void* Pointer, HsaPointerInfo* PointerInfo) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  void* gpu_va = vhsakmt_gpu_va(dev, VHSA_UINT64_TO_VPTR(Pointer));
  if (!gpu_va) return -HSAKMT_STATUS_ERROR;
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, sizeof(struct vhsakmt_ccmd_query_info_req)),
      .type = VHSAKMT_CCMD_QUERY_POINTER_INFO,
      .pointer = VHSA_VPTR_TO_UINT64(gpu_va),
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr,
                          sizeof(struct vhsakmt_ccmd_query_info_rsp) +
                              QUERY_PTR_INFO_MAX_MAPPED_NODES * sizeof(uint32_t));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  memcpy(PointerInfo, &rsp->ptr_info, sizeof(HsaPointerInfo));

  if (PointerInfo->NMappedNodes && PointerInfo->MappedNodes) {
    if (PointerInfo->NMappedNodes > QUERY_PTR_INFO_MAX_MAPPED_NODES) {
      PointerInfo->NMappedNodes = QUERY_PTR_INFO_MAX_MAPPED_NODES;
      vhsa_debug(
          "%s: query pointer: %p info mapped nodes greater than QUERY_PTR_INFO_MAX_MAPPED_NODES\n",
          __FUNCTION__, Pointer);
    }

    PointerInfo->MappedNodes = calloc(PointerInfo->NMappedNodes, sizeof(uint32_t));
    if (!PointerInfo->MappedNodes) {
      PointerInfo->NMappedNodes = 0;
      return -HSAKMT_STATUS_NO_MEMORY;
    }
    memcpy(VHSA_UINT64_TO_VPTR(PointerInfo->MappedNodes), rsp->payload,
           PointerInfo->NMappedNodes * sizeof(uint32_t));
  }

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtGetTileConfig(HSAuint32 NodeId, HsaGpuTileConfig* config) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  uint8_t* config_cpy_addr = NULL;
  struct vhsakmt_ccmd_query_info_rsp* rsp;
  unsigned req_len = sizeof(struct vhsakmt_ccmd_query_info_req);
  unsigned rsp_len = sizeof(struct vhsakmt_ccmd_query_info_rsp) +
      config->NumTileConfigs * sizeof(HSAuint32) + config->NumMacroTileConfigs * sizeof(HSAuint32);

  struct vhsakmt_ccmd_query_info_req req = {
      .hdr = VHSAKMT_CCMD(QUERY_INFO, req_len),
      .type = VHSAKMT_CCMD_QUERY_TILE_CONFIG,
      .tile_config_args.NodeId = NodeId,
      .tile_config_args.config = *config,
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, rsp_len);
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  memcpy(config, &rsp->tile_config_rsp, sizeof(HsaGpuTileConfig));
  config_cpy_addr = ((uint8_t*)rsp->payload);
  memcpy(config->TileConfig, config_cpy_addr, config->NumTileConfigs * sizeof(HSAuint32));
  config_cpy_addr += config->NumTileConfigs * sizeof(HSAuint32);
  memcpy(config->MacroTileConfig, config_cpy_addr, config->NumMacroTileConfigs * sizeof(HSAuint32));

  return rsp->ret;
}

static int vhsakmt_create_clgl_bo(vhsakmt_device_handle dev, void* addr, size_t size,
                                  uint32_t res_id, uint32_t bo_handle, void* meta_data) {
  vhsakmt_bo_handle out = calloc(1, sizeof(struct vhsakmt_bo));
  if (!out) return -ENOMEM;

  out->dev = dev;
  out->size = size;
  atomic_store(&out->real.map_count, 0);
  atomic_store(&out->refcount, 1);

#ifdef CLGL_EXPORT_RESID
  out->real.res_id = GraphicsResourceHandle;
#else
  out->real.res_id = res_id;
#endif

  /* GL bo handle from GL context*/
  out->real.handle = bo_handle;
  out->bo_type |= VHSA_BO_CLGL;
  if (meta_data) out->gl_meta_data = meta_data;

  out->host_addr = addr;

  vhsakmt_insert_bo(dev, out, addr, out->size);

  return 0;
}

static int vhsakmt_gfxhandle_to_resid(vhsakmt_device_handle dev, uint32_t gfx_handle,
                                      uint32_t* res_id, uint32_t* bo_handle) {
  int r = drmPrimeFDToHandle(dev->vgdev->fd, gfx_handle, bo_handle);
  if (r) {
    vhsa_err("%s: drmPrimeFDToHandle failed for handle: %u\n", __FUNCTION__, gfx_handle);
    return r;
  }

  virtio_gpu_res_id(dev->vgdev, *bo_handle, res_id);

  vhsa_debug("%s: register praphics handle: handle: %d, bo_handle: %d, res_id: %d\n", __FUNCTION__,
             gfx_handle, *bo_handle, *res_id);

  return 0;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtRegisterGraphicsHandleToNodes(
    HSAuint64 GraphicsResourceHandle, HsaGraphicsResourceInfo* GraphicsResourceInfo,
    HSAuint64 NumberOfNodes, HSAuint32* NodeArray) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  uint32_t bo_handle, res_id;
  uint64_t meta_data_size = VHSA_GL_METADATA_MAX_SIZE;
  unsigned req_len = sizeof(struct vhsakmt_ccmd_gl_inter_req) + NumberOfNodes * sizeof(NodeArray);
  struct vhsakmt_ccmd_gl_inter_req* req;
  struct vhsakmt_ccmd_gl_inter_rsp* rsp;
  int r;

  req = calloc(1, req_len);
  if (!req) return -ENOMEM;

  req->hdr = VHSAKMT_CCMD(GL_INTER, req_len);
  req->type = VHSAKMT_CCMD_GL_REG_GHD_TO_NODES;
  req->reg_ghd_to_nodes.NumberOfNodes = NumberOfNodes;
  req->reg_ghd_to_nodes.res_handle = GraphicsResourceHandle;

#ifdef CLGL_EXPORT_RESID
  req->reg_ghd_to_nodes.GraphicsResourceHandle = GraphicsResourceHandle;
#else
  r = vhsakmt_gfxhandle_to_resid(dev, GraphicsResourceHandle, &res_id, &bo_handle);
  if (r) return r;

  req->reg_ghd_to_nodes.GraphicsResourceHandle = bo_handle;
  req->reg_ghd_to_nodes.res_handle = res_id;
#endif

  memcpy(req->payload, NodeArray, NumberOfNodes * sizeof(NodeArray));

  rsp =
      vhsakmt_alloc_rsp(dev, &req->hdr, sizeof(struct vhsakmt_ccmd_gl_inter_rsp) + meta_data_size);
  if (!rsp) {
    r = -ENOMEM;
    goto free_out;
  }

  vhsakmt_execbuf_cpu(dev, &req->hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  memcpy(GraphicsResourceInfo, &rsp->info, sizeof(HsaGraphicsResourceInfo));
  if (rsp->info.MetadataSizeInBytes) {
    GraphicsResourceInfo->Metadata = calloc(1, GraphicsResourceInfo->MetadataSizeInBytes);
    if (!GraphicsResourceInfo->Metadata) {
      r = -ENOMEM;
      goto free_out;
    }

    memcpy(VHSA_UINT64_TO_VPTR(GraphicsResourceInfo->Metadata), rsp->payload,
           GraphicsResourceInfo->MetadataSizeInBytes);
  } else
    GraphicsResourceInfo->Metadata = NULL;

  vhsa_debug("%s: register graphics handle: handle: %ld hva: %p, size: %lx\n", __FUNCTION__,
             GraphicsResourceHandle, GraphicsResourceInfo->MemoryAddress,
             GraphicsResourceInfo->SizeInBytes);

  r = vhsakmt_create_clgl_bo(dev, GraphicsResourceInfo->MemoryAddress,
                             GraphicsResourceInfo->SizeInBytes, res_id, bo_handle,
                             VHSA_UINT64_TO_VPTR(GraphicsResourceInfo->Metadata));
  if (r) goto free_out;

  r = rsp->ret;

free_out:
  /* close exported FD after register or close it when deregistre. Close after register here. */
  close(GraphicsResourceHandle);
  free(req);
  return r;
}
