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

#ifndef VHSAKMT_VIRTIO_PROTO_H
#define VHSAKMT_VIRTIO_PROTO_H

#include "hsakmt/linux/kfd_ioctl.h"
#include "hsakmt/hsakmt.h"

#include <drm/amdgpu_drm.h>
#include <libdrm/amdgpu.h>
#include <stdint.h>

#include "virtio_gpu.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wpadded"
#endif

/* defined in other header file in virglrenderer */
#define VHSAKMT_DEFINE_CAST(parent, child)                                                         \
  static inline struct child* to_##child(struct parent* x) { return (struct child*)x; }

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define VHSAKMT_STATIC_ASSERT_SIZE(t)                                                              \
  static_assert(sizeof(struct t) % 8 == 0, "sizeof(struct " #t ") not multiple of 8");             \
  static_assert(_Alignof(struct t) <= 8, "alignof(struct " #t ") too large");
#else
#define VHSAKMT_STATIC_ASSERT_SIZE(t)
#endif

enum vhsakmt_ccmd {
  VHSAKMT_CCMD_NOP = 1, /* No payload, can be used to sync with host */
  VHSAKMT_CCMD_QUERY_INFO,
  VHSAKMT_CCMD_EVENT,
  VHSAKMT_CCMD_MEMORY,
  VHSAKMT_CCMD_QUEUE,
  VHSAKMT_CCMD_GL_INTER,
};

typedef struct _vHsaEvent {
  HsaEvent event;
  uint64_t event_handle;
  uint64_t bo_handle;
  uint32_t res_id;
  uint32_t pad;
} vHsaEvent;
VHSAKMT_STATIC_ASSERT_SIZE(_vHsaEvent)

struct vhsakmt_event_shmem {
  uint32_t trigered_events_num;
  uint32_t pad;
  HsaEvent trigered_events[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_event_shmem)

#define VHSAKMT_CCMD(_cmd, _len)                                                                   \
  ((struct vhsakmt_ccmd_req){                                                                      \
      .cmd = VHSAKMT_CCMD_##_cmd,                                                                  \
      .len = (_len),                                                                               \
  })

struct vhsakmt_ccmd_nop_req {
  struct vhsakmt_ccmd_req hdr;
};

/*
 * VHSAKMT_CCMD_QUERY
 */
enum vhsakmt_ccmd_query_type {
  VHSAKMT_CCMD_QUERY_GPU_INFO = 0,
  VHSAKMT_CCMD_QUERY_OPEN_KFD,
  VHSAKMT_CCMD_QUERY_GET_VER,
  VHSAKMT_CCMD_QUERY_REL_SYS_PROP,
  VHSAKMT_CCMD_QUERY_GET_SYS_PROP,
  VHSAKMT_CCMD_QUERY_GET_NODE_PROP,
  VHSAKMT_CCMD_QUERY_GET_XNACK_MODE,
  VHSAKMT_CCMD_QUERY_RUN_TIME_ENABLE,
  VHSAKMT_CCMD_QUERY_RUN_TIME_DISABLE,
  VHSAKMT_CCMD_QUERY_GET_NOD_MEM_PROP,
  VHSAKMT_CCMD_QUERY_GET_NOD_CACHE_PROP,
  VHSAKMT_CCMD_QUERY_GET_NOD_IO_LINK_PROP,
  VHSAKMT_CCMD_QUERY_GET_CLOCK_COUNTERS,
  VHSAKMT_CCMD_QUERY_POINTER_INFO,
  VHSAKMT_CCMD_QUERY_TILE_CONFIG,
  VHSAKMT_CCMD_QUERY_NANO_TIME,
  VHSAKMT_CCMD_QUERY_GET_RUNTIME_CAPS,
};

#define QUERY_PTR_INFO_MAX_MAPPED_NODES 3

typedef struct _query_req_run_time_enable_args {
  /* void*     rDebug, bypassed by payload */
  uint8_t pad[3];
  uint8_t setupTtmp;
  uint32_t __pad;
} query_req_run_time_enable_args;
VHSAKMT_STATIC_ASSERT_SIZE(_query_req_run_time_enable_args)

typedef struct _query_req_node_mem_prop_args {
  uint32_t NodeId;
  uint32_t NumBanks;
} query_req_node_mem_prop_args;
VHSAKMT_STATIC_ASSERT_SIZE(_query_req_node_mem_prop_args)

typedef struct _query_req_node_cache_prop_args {
  uint32_t NodeId;
  uint32_t ProcessorId;
  uint32_t NumCaches;
  uint32_t pad;
} query_req_node_cache_prop_args;
VHSAKMT_STATIC_ASSERT_SIZE(_query_req_node_cache_prop_args)

typedef struct _query_req_node_io_link_args {
  uint32_t NodeId;
  uint32_t NumIoLinks;
} query_req_node_io_link_args;
VHSAKMT_STATIC_ASSERT_SIZE(_query_req_node_io_link_args)

typedef struct _query_tile_config {
  HsaGpuTileConfig config;
  uint32_t NodeId;
  uint32_t pad;
} query_tile_config;
VHSAKMT_STATIC_ASSERT_SIZE(_query_tile_config)

typedef struct _query_open_kfd_args {
  uint64_t cur_vm_start;
} query_open_kfd_args;
VHSAKMT_STATIC_ASSERT_SIZE(_query_open_kfd_args)

typedef struct _query_open_kfd_rsp {
  uint64_t vm_start;
  uint64_t vm_size;
} query_open_kfd_rsp;
VHSAKMT_STATIC_ASSERT_SIZE(_query_open_kfd_rsp)

typedef struct _query_nano_time_rsp {
  uint64_t nano_time;
} query_nano_time_rsp;
VHSAKMT_STATIC_ASSERT_SIZE(_query_nano_time_rsp)

struct vhsakmt_ccmd_query_info_req {
  struct vhsakmt_ccmd_req hdr;
  struct drm_amdgpu_info info;
  uint32_t type;
  uint32_t pad;
  union {
    uint64_t pointer;
    uint32_t NodeID; /* some query API just need node ID */
    query_req_run_time_enable_args run_time_enable_args;
    query_req_node_mem_prop_args node_mem_prop_args;
    query_req_node_cache_prop_args node_cache_prop_args;
    query_req_node_io_link_args node_io_link_args;
    query_tile_config tile_config_args;
    query_open_kfd_args open_kfd_args;
  };

  uint8_t payload[];
};
VHSAKMT_DEFINE_CAST(vhsakmt_ccmd_req, vhsakmt_ccmd_query_info_req)
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_query_info_req)
#define VHSAKMT_CCMD_QUERY_MAX_TILE_CONFIG 128
#define VHSAKMT_CCMD_QUERY_MAX_GET_NOD_MEM_PROP 128
#define VHSAKMT_CCMD_QUERY_MAX_GET_NOD_CACHE_PROP 128
#define VHSAKMT_CCMD_QUERY_MAX_GET_NOD_IO_LINK_PROP 128

struct vhsakmt_ccmd_query_info_rsp {
  struct vhsakmt_ccmd_rsp hdr;
  int32_t ret;
  union {
    query_open_kfd_rsp open_kfd_rsp;
    query_nano_time_rsp nano_time_rsp;
    HsaGpuTileConfig tile_config_rsp;
    HsaPointerInfo ptr_info;
    struct amdgpu_gpu_info gpu_info;
    HsaVersionInfo kfd_version;
    HsaSystemProperties sys_props;
    HsaNodeProperties node_props;
    int32_t xnack_mode;
    HsaClockCounters clock_counters;
    uint32_t caps;
    uint64_t pad[9];
  };
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_query_info_rsp)

/*
 * VHSAKMT_CCMD_EVENT
 */
enum vhsakmt_ccmd_event_type {
  VHSAKMT_CCMD_EVENT_CREATE,
  VHSAKMT_CCMD_EVENT_DESTROY,
  VHSAKMT_CCMD_EVENT_SET,
  VHSAKMT_CCMD_EVENT_RESET,
  VHSAKMT_CCMD_EVENT_QUERY_STATE,
  VHSAKMT_CCMD_EVENT_WAIT_ON_MULTI_EVENTS,

  VHSAKMT_CCMD_EVENT_SET_TRAP,

};
typedef struct _event_req_create_args {
  HsaEventDescriptor EventDesc;
  uint8_t ManualReset;
  uint8_t IsSignaled;
  uint8_t pad[6];
} event_req_create_args;
VHSAKMT_STATIC_ASSERT_SIZE(_event_req_create_args)

typedef struct _event_req_wait_args {
  HsaEvent Event;
  uint32_t Milliseconds;
  uint32_t pad;
} event_req_wait_args;
VHSAKMT_STATIC_ASSERT_SIZE(_event_req_wait_args)

typedef struct _event_req_wait_ext_args {
  HsaEvent Event;
  uint64_t event_age;
  uint32_t Milliseconds;
  uint32_t pad;
} event_req_wait_ext_args;
VHSAKMT_STATIC_ASSERT_SIZE(_event_req_wait_ext_args)

typedef struct _event_req_wait_on_multi_args {
  /*HsaEvent*   Events[], in playloud*/
  uint32_t NumEvents;
  uint32_t Milliseconds;
  uint8_t WaitOnAll;
  uint8_t pad[7];
} event_req_wait_on_multi_args;
VHSAKMT_STATIC_ASSERT_SIZE(_event_req_wait_on_multi_args)

typedef struct _event_req_wait_on_multi_ext_args {
  /*HsaEvent*   Events[], in playloud*/
  uint32_t NumEvents;
  uint32_t Milliseconds;
  uint64_t event_age;
  uint8_t WaitOnAll;
  uint8_t pad[7];
} event_req_wait_on_multi_ext_args;
VHSAKMT_STATIC_ASSERT_SIZE(_event_req_wait_on_multi_ext_args)

typedef struct _event_set_trap_handler_args {
  uint64_t TrapHandlerBaseAddress;
  uint64_t TrapHandlerSizeInBytes;
  uint64_t TrapBufferBaseAddress;
  uint64_t TrapBufferSizeInBytes;
  uint32_t NodeId;
  uint32_t pad;
} event_set_trap_handler_args;
VHSAKMT_STATIC_ASSERT_SIZE(_event_set_trap_handler_args)

struct vhsakmt_ccmd_event_req {
  struct vhsakmt_ccmd_req hdr;
  union {
    HsaEvent Event; /* For set, reset, query. */
    HsaEvent* event_hanele;
    event_req_wait_args wait_args;
    event_req_create_args create_args;
    event_req_wait_ext_args wait_ext_args;
    event_req_wait_on_multi_args wait_on_multi_args;
    event_req_wait_on_multi_ext_args wait_on_multi_ext_args;
    event_set_trap_handler_args set_trap_handler_args;
  };
  uint32_t type;
  uint32_t sync_shmem_res_id;
  uint64_t blob_id;
  uint32_t res_id;
  uint32_t pad;
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_event_req)
VHSAKMT_DEFINE_CAST(vhsakmt_ccmd_req, vhsakmt_ccmd_event_req)

struct vhsakmt_ccmd_event_rsp {
  struct vhsakmt_ccmd_rsp hdr;
  int32_t ret;
  vHsaEvent vevent;
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_event_rsp)

/*
 * VHSAKMT_CCMD_MEMORY
 */
enum vhsakmt_ccmd_memory_type {
  VHSAKMT_CCMD_MEMORY_ALLOC,
  VHSAKMT_CCMD_MEMORY_MAP_TO_GPU_NODES,
  VHSAKMT_CCMD_MEMORY_FREE,
  VHSAKMT_CCMD_MEMORY_UNMAP_TO_GPU,
  VHSAKMT_CCMD_MEMORY_AVAIL_MEM,
  VHSAKMT_CCMD_MEMORY_MAP_MEM_TO_GPU,
  VHSAKMT_CCMD_MEMORY_REG_MEM_WITH_FLAG,
  VHSAKMT_CCMD_MEMORY_DEREG_MEM,
  VHSAKMT_CCMD_MEMORY_MAP_USERPTR,
};

typedef struct _memory_req_alloc_args {
  uint32_t PreferredNode;
  HsaMemFlags MemFlags;
  uint64_t SizeInBytes;
  uint64_t MemoryAddress;
} memory_req_alloc_args;
VHSAKMT_STATIC_ASSERT_SIZE(_memory_req_alloc_args)

typedef struct _memory_req_free_args {
  uint64_t MemoryAddress;
  uint64_t SizeInBytes;
} memory_req_free_args;
VHSAKMT_STATIC_ASSERT_SIZE(_memory_req_free_args)

typedef struct _memory_req_map_to_GPU_nodes_args {
  uint64_t MemoryAddress;
  uint64_t MemorySizeInBytes;
  uint64_t AlternateVAGPU;
  HsaMemMapFlags MemMapFlags;
  uint32_t pad;
  uint64_t NumberOfNodes;
  uint32_t* NodeArray;
} memory_req_map_to_GPU_nodes_args;
VHSAKMT_STATIC_ASSERT_SIZE(_memory_req_map_to_GPU_nodes_args)

typedef struct _memory_map_mem_to_gpu_args {
  uint64_t MemoryAddress;
  uint64_t MemorySizeInBytes;
  uint8_t need_create_bo;
  uint8_t pad[7];
} memory_map_mem_to_gpu_args;
VHSAKMT_STATIC_ASSERT_SIZE(_memory_map_mem_to_gpu_args)

typedef struct _memory_reg_mem_with_flag {
  uint64_t MemoryAddress;
  uint64_t MemorySizeInBytes;
  HsaMemFlags MemFlags;
  uint32_t pad;
} memory_reg_mem_with_flag;
VHSAKMT_STATIC_ASSERT_SIZE(_memory_reg_mem_with_flag)

struct vhsakmt_ccmd_memory_req {
  struct vhsakmt_ccmd_req hdr;
  union {
    uint64_t MemoryAddress;
    uint32_t Node;
    memory_req_alloc_args alloc_args;
    memory_req_map_to_GPU_nodes_args map_to_GPU_nodes_args;
    memory_req_free_args free_args;
    memory_map_mem_to_gpu_args map_to_GPU_args;
    memory_reg_mem_with_flag reg_mem_with_flag;
  };
  uint64_t blob_id;
  uint32_t type;
  uint32_t res_id;
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_memory_req)
VHSAKMT_DEFINE_CAST(vhsakmt_ccmd_req, vhsakmt_ccmd_memory_req)

typedef struct _vhsakmt_ccmd_memory_map_userptr_rsp {
  uint64_t userptr_handle;
  uint32_t npfns;
  uint32_t pad;
} vhsakmt_ccmd_memory_map_userptr_rsp;
VHSAKMT_STATIC_ASSERT_SIZE(_vhsakmt_ccmd_memory_map_userptr_rsp)

struct vhsakmt_ccmd_memory_rsp {
  struct vhsakmt_ccmd_rsp hdr;
  int32_t ret;
  union {
    vhsakmt_ccmd_memory_map_userptr_rsp map_userptr_rsp;
    uint64_t memory_handle;
    uint64_t alternate_vagpu;
    uint64_t available_bytes;
  };
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_memory_rsp)

/*
 * VHSAKMT_CCMD_QUEUE
 */
enum vhsakmt_ccmd_queue_type {
  VHSAKMT_CCMD_QUEUE_CREATE,
  VHSAKMT_CCMD_QUEUE_DESTROY,
};

typedef struct _vHsaQueueResource {
  HsaQueueResource r;
  uint64_t host_doorbell;
  uint64_t host_doorbell_offset;
  uint64_t host_write_offset;
  uint64_t host_read_offset;
  uint64_t host_rw_handle;
  uint64_t queue_handle;
} vHsaQueueResource;
VHSAKMT_STATIC_ASSERT_SIZE(_vHsaQueueResource)

typedef struct _queue_req_create {
  uint32_t NodeId;
  HSA_QUEUE_TYPE Type;
  uint32_t QueuePercentage;
  uint32_t pad;
  HSA_QUEUE_PRIORITY Priority;
  uint32_t pad1;
  uint32_t SdmaEngineId;
  uint64_t QueueAddress;
  uint64_t QueueSizeInBytes;
  HsaEvent* Event;
  HsaQueueResource* QueueResource;
  uint64_t* Queue_write_ptr_aql;
  uint64_t* Queue_read_ptr_aql;
} queue_req_create;
VHSAKMT_STATIC_ASSERT_SIZE(_queue_req_create)

struct vhsakmt_ccmd_queue_req {
  struct vhsakmt_ccmd_req hdr;
  union {
    HSA_QUEUEID QueueId;
    queue_req_create create_queue_args;
  };
  uint64_t blob_id;          /* For queue create, queue resource */
  uint64_t rw_ptr_blob_id;   /* For queue create, r/w ptr memory mapping */
  uint64_t doorbell_blob_id; /* For queue create, doorbell ptr memory mapping */
  uint32_t res_id;
  uint32_t type;
  uint32_t queue_mem_res_id;
  uint32_t pad;
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_queue_req)
VHSAKMT_DEFINE_CAST(vhsakmt_ccmd_req, vhsakmt_ccmd_queue_req)

struct vhsakmt_ccmd_queue_rsp {
  struct vhsakmt_ccmd_rsp hdr;
  int32_t ret;
  vHsaQueueResource vqueue_res;
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_queue_rsp)

/*
 * VHSAKMT_CCMD_GL_INTER
 */
enum vhsakmt_ccmd_gl_inter_type {
  VHSAKMT_CCMD_GL_REG_GHD_TO_NODES,
};

typedef struct _gl_inter_req_reg_ghd_to_nodes {
  uint64_t GraphicsResourceHandle;
  uint64_t NumberOfNodes;  // NodeArray in payload
  uint32_t res_handle;
  uint32_t pad;
} gl_inter_req_reg_ghd_to_nodes;
VHSAKMT_STATIC_ASSERT_SIZE(_gl_inter_req_reg_ghd_to_nodes)

struct vhsakmt_ccmd_gl_inter_req {
  struct vhsakmt_ccmd_req hdr;
  union {
    gl_inter_req_reg_ghd_to_nodes reg_ghd_to_nodes;
  };
  uint32_t type;
  uint32_t pad;
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_gl_inter_req)
VHSAKMT_DEFINE_CAST(vhsakmt_ccmd_req, vhsakmt_ccmd_gl_inter_req)

struct vhsakmt_ccmd_gl_inter_rsp {
  struct vhsakmt_ccmd_rsp hdr;
  int32_t ret;
  union {
    HsaGraphicsResourceInfo info;
  };
  uint8_t payload[];
};
VHSAKMT_STATIC_ASSERT_SIZE(vhsakmt_ccmd_gl_inter_rsp)

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif
