/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef HSAKMT_VIRTIO_DEVICE_H
#define HSAKMT_VIRTIO_DEVICE_H

#include "hsakmt_virtio_proto.h"
#include "rbtree.h"
#include "virtio_gpu.h"
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

#define vhsakmt_atomic_inc_return(ptr) (atomic_fetch_add((ptr), 1) + 1)
#define vhsakmt_atomic_dec_return(ptr) (atomic_fetch_sub((ptr), 1) - 1)

#define VHSA_VPTR_TO_UINT64(vptr) ((uint64_t)(unsigned long)(vptr))
#define VHSA_UINT64_TO_VPTR(v) ((void*)(unsigned long)(v))

extern int vhsakmt_debug_level;
#define vhsakmt_print(level, fmt, ...)                                                             \
  do {                                                                                             \
    if (level <= vhsakmt_debug_level) fprintf(stderr, fmt, ##__VA_ARGS__);                         \
  } while (0)
#define VHSAKMT_DEBUG_LEVEL_DEFAULT -1
#define VHSAKMT_DEBUG_LEVEL_ERR 3
#define VHSAKMT_DEBUG_LEVEL_WARNING 4
#define VHSAKMT_DEBUG_LEVEL_INFO 6
#define VHSAKMT_DEBUG_LEVEL_DEBUG 7
#define vhsa_err(fmt, ...) vhsakmt_print(VHSAKMT_DEBUG_LEVEL_ERR, fmt, ##__VA_ARGS__)
#define vhsa_warn(fmt, ...) vhsakmt_print(VHSAKMT_DEBUG_LEVEL_WARNING, fmt, ##__VA_ARGS__)
#define vhsa_info(fmt, ...) vhsakmt_print(VHSAKMT_DEBUG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define vhsa_debug(fmt, ...) vhsakmt_print(VHSAKMT_DEBUG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)

struct vhsakmt_device;
struct vhsakmt_bo;

typedef struct vhsakmt_device* vhsakmt_device_handle;
typedef struct vhsakmt_bo* vhsakmt_bo_handle;
typedef rbtree_node_t* bo_entry;

extern pthread_mutex_t dev_mutex;
extern vhsakmt_device_handle dev_list;

#define VHSA_BO_KFD_MEM 1 << 0 /* allocated from KFD (hsaKmtAllocMemory) */
#define VHSA_BO_USERPTR 1 << 1
#define VHSA_BO_QUEUE_BUFFER 1 << 2   /* allocated from KFD, but used for queue CMD submit */
#define VHSA_BO_QUEUE_DOORBELL 1 << 3 /* doorbell memory */
#define VHSA_BO_QUEUE_RW_PTR 1 << 4   /* queue read write ptr, from host map to guest*/
/* allocated from KFD, but used for AQL queue read write ptr */
#define VHSA_BO_QUEUE_AQL_RW_PTR 1 << 5
#define VHSA_BO_CLGL 1 << 6 /* CLGL memory, imported from mesa GL */
/* allocated from KFD, but is scratch memory, do not need map and unmap in ioctrl */
#define VHSA_BO_SCRATCH 1 << 7
#define VHSA_BO_QUEUE 1 << 8
#define VHSA_BO_EVENT 1 << 9
#define VHSA_BO_SCRATCH_MAP 1 << 10

#define VHSA_SDMA_NONE UINT32_MAX

#define CHECK_VIRTIO_KFD_OPEN()                                                                    \
  do {                                                                                             \
    if (dev_list == NULL) return HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;                       \
  } while (0)

struct vhsakmt_node {
  HsaNodeProperties node_props;
  void* doorbell_base;
  uint64_t scratch_start;
  uint64_t scratch_size;
};

struct vhsakmt_device {
  struct virtio_gpu_device* vgdev;
  int refcount;
  pthread_mutex_t bo_handles_mutex;
  rbtree_t bo_rbt;

  struct vhsakmt_bo* shmem_bo;

  uint32_t reqbuf_max;
  uint32_t next_blob_id;

  uint64_t vm_start;
  uint64_t vm_size;

  pthread_mutex_t vhsakmt_mutex;
  struct vhsakmt_node* vhsakmt_nodes;
  HsaSystemProperties* sys_props;
};

struct vhsakmt_bo {
  rbtree_node_t rbtn;
  struct vhsakmt_device* dev;

  int refcount;
  unsigned size;
  void* cpu_addr;
  void* host_addr;
  HsaMemFlags flags;
  uint32_t bo_type;
  uint32_t blob_id;
  pthread_mutex_t map_mutex;

  union {
    struct {
      uint32_t handle;
      uint32_t res_id;
      uint64_t offset;
      uint64_t alloc_size;
      int map_count;
    } real;
  };

  vHsaEvent* event;
  uint64_t queue_id;
  vhsakmt_bo_handle rw_bo;
  void* gl_meta_data;
};

/*hsakmt_virtio_memory.c*/
vhsakmt_bo_handle vhsakmt_entry_to_bo_handle(bo_entry e);
bo_entry vhsakmt_bo_handle_to_entry(vhsakmt_bo_handle bo);

void vhsakmt_insert_bo(vhsakmt_device_handle dev, vhsakmt_bo_handle bo, void* addr, uint64_t size);
void vhsakmt_remove_bo(vhsakmt_device_handle dev, vhsakmt_bo_handle bo);
vhsakmt_bo_handle vhsakmt_find_bo_by_addr(vhsakmt_device_handle dev, void* addr);
void* vhsakmt_gpu_va(vhsakmt_device_handle dev, void* va);

int vhsakmt_bo_cpu_unmap(vhsakmt_bo_handle bo);
int vhsakmt_bo_cpu_map(vhsakmt_bo_handle bo_handle, void** cpu, void* fixed_cpu);
int vhsakmt_create_mappable_blob_bo(vhsakmt_device_handle dev, size_t size, uint32_t blob_id,
                                    uint32_t bo_type, void* va_handle,
                                    vhsakmt_bo_handle* bo_handle);
int vhsakmt_bo_free(vhsakmt_device_handle dev, vhsakmt_bo_handle bo);
int vhsakmt_init_host_blob(vhsakmt_device_handle dev, size_t size, uint32_t blob_type,
                           uint32_t blob_flag, uint32_t blob_id, uint32_t bo_type, void* va_handle,
                           vhsakmt_bo_handle* bo_handle);

/*hsakmt_virtio_openclose.c*/
vhsakmt_device_handle vhsakmt_dev(void);

/*hsakmt_virtio_vm.c*/
void* vhsakmt_vm_start(void);
int vhsakmt_reserve_va(uint64_t start, uint64_t size);
void vhsakmt_dereserve_va(uint64_t start, uint64_t size);
void vhsakmt_set_scratch_area(vhsakmt_device_handle dev, uint32_t node, uint64_t start,
                              uint64_t size);
void vhsakmt_set_vm_area(vhsakmt_device_handle dev, uint64_t start, uint64_t size);
int vhsakmt_set_node_doorbell(vhsakmt_device_handle dev, uint32_t node, void* doorbell);
void* vhsakmt_node_doorbell(vhsakmt_device_handle dev, uint32_t node);
bool vhsakmt_is_scratch_mem(vhsakmt_device_handle dev, void* addr);
bool vhsakmt_is_userptr(vhsakmt_device_handle dev, void* addr);

/*hsakmt_virtio_device.c*/
int vhsakmt_execbuf_cpu(vhsakmt_device_handle dev, struct vhsakmt_ccmd_req* req, const char* from);
void* vhsakmt_alloc_rsp(vhsakmt_device_handle dev, struct vhsakmt_ccmd_req* req, uint32_t sz);

/*hsakmt_virtio_event.c*/
void* vhsakmt_event_host_handle(HsaEvent* h);

#ifdef __cplusplus
}
#endif

#endif
