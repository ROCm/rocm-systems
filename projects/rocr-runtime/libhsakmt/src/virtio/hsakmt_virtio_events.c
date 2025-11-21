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

int vhsakmt_debug_level;

void* vhsakmt_event_host_handle(HsaEvent* h) { return (void*)((vHsaEvent*)h)->event_handle; }

static inline int32_t vhsakmt_event_res_id(HsaEvent* h) { return ((vHsaEvent*)h)->res_id; }

static inline vhsakmt_bo_handle vhsakmt_event_bo_handle(HsaEvent* h) {
  return (vhsakmt_bo_handle)((vHsaEvent*)h)->bo_handle;
}

static int vhsakmt_create_event_blob_bo(vhsakmt_device_handle dev, size_t size, uint32_t blob_id,
                                        vHsaEvent* vevent_handle, vhsakmt_bo_handle* bo_handle) {
  int r;

  r = vhsakmt_init_host_blob(dev, size, VIRTGPU_BLOB_MEM_HOST3D, 0, blob_id, VHSA_BO_EVENT,
                             (void*)vevent_handle->event_handle, bo_handle);
  if (r) return r;

  (*bo_handle)->event = vevent_handle;
  vevent_handle->bo_handle = (uint64_t)(*bo_handle);
  vevent_handle->res_id = (*bo_handle)->real.res_id;
  vhsakmt_insert_bo(dev, *bo_handle, vevent_handle, size);
  return r;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtCreateEvent(HsaEventDescriptor* EventDesc, _Bool ManualReset,
                                           _Bool IsSignaled, HsaEvent** Event) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_event_rsp* rsp;
  vhsakmt_bo_handle event_bo;
  vHsaEvent* e;
  int r;
  struct vhsakmt_ccmd_event_req req = {
      .hdr = VHSAKMT_CCMD(EVENT, sizeof(struct vhsakmt_ccmd_event_req)),
      .type = VHSAKMT_CCMD_EVENT_CREATE,
      .create_args.EventDesc = *EventDesc,
      .create_args.ManualReset = ManualReset,
      .create_args.IsSignaled = IsSignaled,
      .blob_id = vhsakmt_atomic_inc_return(&dev->next_blob_id),
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_event_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);
  if (rsp->ret) return rsp->ret;

  e = calloc(1, sizeof(vHsaEvent));
  if (!e) return -ENOMEM;

  memcpy(e, &rsp->vevent, sizeof(vHsaEvent));

  r = vhsakmt_create_event_blob_bo(dev, sizeof(vHsaEvent), req.blob_id, e, &event_bo);
  if (r) {
    free(e);
    return -ENOMEM;
  }

  *Event = (HsaEvent*)e;

  vhsa_debug(
      "%s: event addr: %p, hw123: %lx, %lx, %x, type: %d, id: %x, host handle: 0x%lx, res id: %d\n",
      __FUNCTION__, e, e->event.EventData.HWData1, e->event.EventData.HWData2,
      e->event.EventData.HWData3, e->event.EventData.EventType, e->event.EventId, e->event_handle,
      event_bo->real.res_id);

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtDestroyEvent(HsaEvent* Event) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_bo* bo;

  if (Event == NULL) return HSAKMT_STATUS_SUCCESS;

  bo = vhsakmt_event_bo_handle(Event);
  if (!bo) return HSAKMT_STATUS_SUCCESS;

  return vhsakmt_bo_free(dev, bo);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtSetEvent(HsaEvent* Event) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_event_rsp* rsp;
  struct vhsakmt_ccmd_event_req req = {
      .hdr = VHSAKMT_CCMD(EVENT, sizeof(struct vhsakmt_ccmd_event_req)),
      .type = VHSAKMT_CCMD_EVENT_SET,
      .event_hanele = vhsakmt_event_host_handle(Event),
      .res_id = vhsakmt_event_res_id(Event),
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_event_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtResetEvent(HsaEvent* Event) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_event_rsp* rsp;
  struct vhsakmt_ccmd_event_req req = {
      .hdr = VHSAKMT_CCMD(EVENT, sizeof(struct vhsakmt_ccmd_event_req)),
      .type = VHSAKMT_CCMD_EVENT_RESET,
      .event_hanele = vhsakmt_event_host_handle(Event),
      .res_id = vhsakmt_event_res_id(Event),
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_event_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtQueryEventState(HsaEvent* Event) {
  CHECK_VIRTIO_KFD_OPEN();

  vhsakmt_device_handle dev = vhsakmt_dev();
  struct vhsakmt_ccmd_event_rsp* rsp;
  struct vhsakmt_ccmd_event_req req = {
      .hdr = VHSAKMT_CCMD(EVENT, sizeof(struct vhsakmt_ccmd_event_req)),
      .type = VHSAKMT_CCMD_EVENT_QUERY_STATE,
      .event_hanele = vhsakmt_event_host_handle(Event),
      .res_id = vhsakmt_event_res_id(Event),
  };

  rsp = vhsakmt_alloc_rsp(dev, &req.hdr, sizeof(struct vhsakmt_ccmd_event_rsp));
  if (!rsp) return -ENOMEM;

  vhsakmt_execbuf_cpu(dev, &req.hdr, __FUNCTION__);

  return rsp->ret;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnMultipleEvents(HsaEvent* Events[], HSAuint32 NumEvents,
                                                    bool WaitOnAll, HSAuint32 Milliseconds) {
  return HSAKMT_STATUS_ERROR;
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnEvent(HsaEvent* Event, HSAuint32 Milliseconds) {
  return vhsaKmtWaitOnMultipleEvents(&Event, 1, true, Milliseconds);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnEvent_Ext(HsaEvent* Event, HSAuint32 Milliseconds,
                                               uint64_t* event_age) {
  return vhsaKmtWaitOnMultipleEvents(&Event, 1, true, Milliseconds);
}

HSAKMT_STATUS HSAKMTAPI vhsaKmtWaitOnMultipleEvents_Ext(HsaEvent* Events[], HSAuint32 NumEvents,
                                                        bool WaitOnAll, HSAuint32 Milliseconds,
                                                        uint64_t* event_age) {
  return vhsaKmtWaitOnMultipleEvents(Events, NumEvents, WaitOnAll, Milliseconds);
}
