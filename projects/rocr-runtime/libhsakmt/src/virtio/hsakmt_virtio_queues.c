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

static inline uint64_t vhsakmt_doorbell_page_size(void) { return 0x2000; }
static inline uint64_t vhsakmt_queue_page_size(void) { return getpagesize(); }

HSAKMT_STATUS HSAKMTAPI vhsaKmtSetTrapHandler(HSAuint32 NodeId, void* TrapHandlerBaseAddress,
                                              HSAuint64 TrapHandlerSizeInBytes,
                                              void* TrapBufferBaseAddress,
                                              HSAuint64 TrapBufferSizeInBytes) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_event_rsp* rsp;
  struct vhsakmt_ccmd_event_req req = {
      .hdr = VHSAKMT_CCMD(EVENT, sizeof(struct vhsakmt_ccmd_event_req)),
      .type = VHSAKMT_CCMD_EVENT_SET_TRAP,
      .set_trap_handler_args =
          {
              .NodeId = NodeId,
              .TrapHandlerBaseAddress = (uint64_t)TrapHandlerBaseAddress,
              .TrapHandlerSizeInBytes = TrapHandlerSizeInBytes,
              .TrapBufferBaseAddress = (uint64_t)TrapBufferBaseAddress,
              .TrapBufferSizeInBytes = TrapBufferSizeInBytes,
          },
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_event_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  return rsp->ret;
}

static int vhsakmt_find_aql_rw_bo(vhsakmt_device_handle dev, uint64_t aql_ptr,
                                  uint32_t* aql_bo_res_id) {
  uint64_t aql_base_ptr = VHSA_ALIGN_DOWN(aql_ptr, getpagesize());

  vhsakmt_bo_handle bo = vhsakmt_find_bo_by_addr(dev, (void*)aql_base_ptr);
  if (!bo) return -EINVAL;

  bo->bo_type |= VHSA_BO_QUEUE_AQL_RW_PTR;
  *aql_bo_res_id = bo->real.res_id;
  return 0;
}

static int vhsakmt_create_doorbell_blob_bo(vhsakmt_device_handle dev, uint32_t node, size_t size,
                                           uint32_t blob_id, uint64_t host_handle,
                                           vhsakmt_bo_handle* bo_handle) {
  int r;

  r = vhsakmt_create_mappable_blob_bo(dev, size, blob_id, VHSA_BO_QUEUE_DOORBELL,
                                      (void*)host_handle, bo_handle);
  if (r) return r;

  r = vhsakmt_set_node_doorbell(dev, node, (*bo_handle)->cpu_addr);

  return r;
}

static int vhsakmt_create_queue_rw_blob_bo(vhsakmt_device_handle dev, size_t size, uint32_t blob_id,
                                           uint64_t host_handle, vhsakmt_bo_handle* bo_handle) {
  int r;

  r = vhsakmt_create_mappable_blob_bo(dev, size, blob_id, VHSA_BO_QUEUE_RW_PTR, NULL, bo_handle);
  if (r) return r;

  (*bo_handle)->host_addr = (void*)host_handle;
  return r;
}

static int vhsakmt_create_queue_blob_bo(vhsakmt_device_handle dev, size_t size, uint32_t blob_id,
                                        uint64_t queue_id, vhsakmt_bo_handle rw_bo_handle,
                                        vhsakmt_bo_handle* bo_handle) {
  int r;

  r = vhsakmt_init_host_blob(dev, size, VIRTGPU_BLOB_MEM_HOST3D, 0, blob_id, VHSA_BO_QUEUE, NULL,
                             bo_handle);
  if (r) return r;

  vhsakmt_insert_bo(dev, *bo_handle, *bo_handle, (*bo_handle)->size);

  (*bo_handle)->queue_id = queue_id;
  (*bo_handle)->rw_bo = rw_bo_handle;

  return r;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtCreateQueueExt(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                                              HSAuint32 QueuePercentage,
                                              HSA_QUEUE_PRIORITY Priority, HSAuint32 SdmaEngineId,
                                              void* QueueAddress, HSAuint64 QueueSizeInBytes,
                                              HsaEvent* Event, HsaQueueResource* QueueResource) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  vhsakmt_bo_handle rw_bo_handle = NULL, doorbell_bo, queue_bo, queue_mem_bo;
  struct vhsakmt_ccmd_queue_rsp* rsp;
  struct vhsakmt_ccmd_queue_req req = {
      .hdr = VHSAKMT_CCMD(QUEUE, sizeof(struct vhsakmt_ccmd_queue_req)),
      .type = VHSAKMT_CCMD_QUEUE_CREATE,
      .create_queue_args =
          {
              .NodeId = NodeId,
              .Type = Type,
              .QueuePercentage = QueuePercentage,
              .Priority = Priority,
              .SdmaEngineId = SdmaEngineId,
              .QueueAddress = (uint64_t)QueueAddress,
              .QueueSizeInBytes = QueueSizeInBytes,
              .Event = Event ? vhsakmt_event_host_handle(Event) : 0,
              .Queue_write_ptr_aql = QueueResource->Queue_write_ptr_aql,
              .Queue_read_ptr_aql = QueueResource->Queue_read_ptr_aql,
          },
      .blob_id = vhsakmt_atomic_inc_return(&dev->next_blob_id), /* For queue resource */
      .doorbell_blob_id = vhsakmt_node_doorbell(dev, NodeId)
          ? 0
          : vhsakmt_atomic_inc_return(&dev->next_blob_id), /* For queue doorbell memory map */
  };
  int r;

  /* Queue ptr memory is allocated by hsakmtallocmemory in host then mapped into guest, but their
   * address are not aligned. */
  if (Type == HSA_QUEUE_COMPUTE_AQL) {
    r = vhsakmt_find_aql_rw_bo(dev, QueueResource->QueueWptrValue, &req.res_id);
    if (r) {
      vhsa_debug("%s: can not find the AQL queue R/W BO: %p\n", __FUNCTION__,
                 QueueResource->Queue_write_ptr_aql);
      return HSAKMT_STATUS_NO_MEMORY;
    }

    vhsa_debug("%s: create AQL queue, read ptr: %p, write ptr: %p, res id: %d\n", __FUNCTION__,
               QueueResource->Queue_read_ptr_aql, QueueResource->Queue_write_ptr_aql, req.res_id);
  } else
    /* For queue not CP AQL, it use r/w ptr by itself. */
    req.rw_ptr_blob_id = vhsakmt_atomic_inc_return(&dev->next_blob_id);

  queue_mem_bo = vhsakmt_find_bo_by_addr(dev, QueueAddress);
  if (!queue_mem_bo) {
    vhsa_err("%s: can not find the queue memory BO: %p\n", __FUNCTION__, QueueAddress);
    return HSAKMT_STATUS_NO_MEMORY;
  }
  queue_mem_bo->bo_type |= VHSA_BO_QUEUE_AQL_RW_PTR;
  req.queue_mem_res_id = queue_mem_bo->real.res_id;

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_queue_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (rsp->ret) {
    vhsa_err("%s: queue create failed, ret: %d", __FUNCTION__, rsp->ret);
    return rsp->ret;
  }

  /* Map doorbell */
  if (req.doorbell_blob_id) {
    r = vhsakmt_create_doorbell_blob_bo(
        dev, NodeId, vhsakmt_doorbell_page_size(), req.doorbell_blob_id,
        rsp->vqueue_res.host_doorbell - rsp->vqueue_res.host_doorbell_offset, &doorbell_bo);
    if (r) {
      vhsa_err("%s: doorbell create failed, doorbell: %lx\n", __FUNCTION__,
               rsp->vqueue_res.host_doorbell);
      return r;
    }
    vhsa_debug("%s: create doorbell: %p, size: 0x%x\n", __FUNCTION__, doorbell_bo->cpu_addr,
               doorbell_bo->size);
  }

  QueueResource->Queue_DoorBell_aql = (void*)rsp->vqueue_res.host_doorbell;
  vhsa_debug("%s: queue create, Doorbell: %p\n", __FUNCTION__, QueueResource->Queue_DoorBell_aql);

  /* Map R/W pointer.
   * For a queue is not a COMPUTE AQL, the R/W PTR not using the input address,
   * uses the queue memory allocated by hsakmtallocmemory, a page align address.
   */
  if (Type != HSA_QUEUE_COMPUTE_AQL) {
    r = vhsakmt_create_queue_rw_blob_bo(dev, vhsakmt_queue_page_size(), req.rw_ptr_blob_id,
                                        rsp->vqueue_res.host_rw_handle, &rw_bo_handle);
    if (r) {
      vhsa_debug("%s: queue rw ptr create failed, host addr: %p\n", __FUNCTION__,
                 (void*)rsp->vqueue_res.host_rw_handle);
      return r;
    }

    QueueResource->Queue_write_ptr_aql = VHSA_UINT64_TO_VPTR(
        VHSA_VPTR_TO_UINT64(rw_bo_handle->cpu_addr) + rsp->vqueue_res.host_write_offset);
    QueueResource->Queue_read_ptr_aql = VHSA_UINT64_TO_VPTR(
        VHSA_VPTR_TO_UINT64(rw_bo_handle->cpu_addr) + rsp->vqueue_res.host_read_offset);

    vhsa_debug("%s: queue create: write ptr gva: %p, read ptr gva: %p, base hva: %lx\n",
               __FUNCTION__, QueueResource->Queue_write_ptr_aql, QueueResource->Queue_read_ptr_aql,
               rsp->vqueue_res.host_rw_handle);
  }

  r = vhsakmt_create_queue_blob_bo(dev, QueueSizeInBytes, req.blob_id, rsp->vqueue_res.r.QueueId,
                                   rw_bo_handle, &queue_bo);
  if (r) {
    vhsa_err("%s: queue create failed, queue ID: 0x%lx\n", __FUNCTION__, rsp->vqueue_res.r.QueueId);
    return r;
  }
  QueueResource->QueueId = (uint64_t)queue_bo;
  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtCreateQueue(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                                           HSAuint32 QueuePercentage, HSA_QUEUE_PRIORITY Priority,
                                           void* QueueAddress, HSAuint64 QueueSizeInBytes,
                                           HsaEvent* Event, HsaQueueResource* QueueResource) {
  return vhsaKmtCreateQueueExt(NodeId, Type, QueuePercentage, Priority, VHSA_SDMA_NONE,
                               QueueAddress, QueueSizeInBytes, Event, QueueResource);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtDestroyQueue(HSA_QUEUEID QueueId) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  int r;

  /* queue ID: vhsakmt_bo_handle -> real queue ID*/
  vhsakmt_bo_handle bo = (vhsakmt_bo_handle)QueueId;
  vhsakmt_bo_handle rw_bo = bo->rw_bo;

  r = vhsakmt_bo_free(dev, bo);
  if (rw_bo) vhsakmt_bo_free(dev, rw_bo);

  vhsa_debug("%s: queue res id: %d, queue ID: %" PRIu64 ", ret = %d\n", __FUNCTION__,
             bo->real.res_id, bo->queue_id, r);

  return r;
}
