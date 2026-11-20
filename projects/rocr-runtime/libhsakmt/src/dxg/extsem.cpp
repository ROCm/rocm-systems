/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hsakmt/hsakmt.h"
#include "impl/wddm/device.h"
#include "impl/wddm/queue.h"
#include "librocdxg.h"

using namespace wsl::thunk;

static inline HSAuint64 EncodeExtSemHandle(D3DKMT_HANDLE syncobj,
                                           HSAuint32 node_id) {
  return (static_cast<HSAuint64>(syncobj) << 32) | node_id;
}

static inline void DecodeExtSemHandle(HSAuint64 packed,
                                      D3DKMT_HANDLE *syncobj,
                                      HSAuint32 *node_id) {
  *syncobj = static_cast<D3DKMT_HANDLE>(packed >> 32);
  *node_id = static_cast<HSAuint32>(packed & 0xFFFFFFFFu);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtImportExternalSemaphore(
    HSAuint32 NodeId,
    void *NtHandle,
    HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE Type,
    HSA_EXTERNAL_SEMAPHORE_HANDLE *OutHandle) {
  CHECK_DXG_OPEN();

  if (NtHandle == nullptr || OutHandle == nullptr)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  // NT and KMT are both openable, by different D3DKMT entry points; an FD
  // is not routable through WDDM at all.
  if (Type != HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 &&
      Type != HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT)
    return HSAKMT_STATUS_NOT_SUPPORTED;

  WDDMDevice *device = get_wddmdev(NodeId);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  D3DKMT_HANDLE syncobj = 0;
  if (Type == HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32) {
    if (!device->OpenSyncobjFromNtHandle(NtHandle, &syncobj))
      return HSAKMT_STATUS_ERROR;
  } else {
    // A KMT handle is a D3DKMT_HANDLE smuggled through a void*, not a
    // pointer; narrow it back rather than dereferencing it.
    const auto kmt = static_cast<D3DKMT_HANDLE>(
        reinterpret_cast<uintptr_t>(NtHandle));
    if (!device->OpenSyncobjFromKmtHandle(kmt, &syncobj))
      return HSAKMT_STATUS_ERROR;
  }

  OutHandle->handle = EncodeExtSemHandle(syncobj, NodeId);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateExternalSemaphore(
    HSAuint32 NodeId,
    HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE Type,
    HSAuint64 InitialValue,
    HSAuint32 Timeline,
    HSA_EXTERNAL_SEMAPHORE_HANDLE *OutHandle,
    void **OutNtHandle) {
  CHECK_DXG_OPEN();

  if (OutHandle == nullptr || OutNtHandle == nullptr)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  // NT and KMT are both exportable, by different mechanisms; nothing else is.
  const bool nt_sharing = (Type == HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32);
  if (Type != HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 &&
      Type != HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT)
    return HSAKMT_STATUS_NOT_SUPPORTED;

  WDDMDevice *device = get_wddmdev(NodeId);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  D3DKMT_HANDLE syncobj = 0;
  D3DKMT_HANDLE kmt_shared = 0;
  if (!device->CreateSharedSyncobj(&syncobj, InitialValue, Timeline != 0, nt_sharing,
                                   &kmt_shared))
    return HSAKMT_STATUS_ERROR;

  void *nt_handle = nullptr;
  if (nt_sharing) {
    if (!device->ExportSyncobjToNtHandle(syncobj, &nt_handle)) {
      // Roll back so a failed export does not strand a shared syncobj, which
      // the KMD refcounts and would not reclaim with the device.
      device->DestroySyncobj(syncobj);
      return HSAKMT_STATUS_ERROR;
    }
  } else {
    // The KMD hands back the global handle at creation; there is nothing to
    // create here and nothing for the caller to close.
    if (kmt_shared == 0) {
      device->DestroySyncobj(syncobj);
      return HSAKMT_STATUS_ERROR;
    }
    nt_handle = reinterpret_cast<void *>(static_cast<uintptr_t>(kmt_shared));
  }

  OutHandle->handle = EncodeExtSemHandle(syncobj, NodeId);
  *OutNtHandle = nt_handle;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyExternalSemaphore(
    HSA_EXTERNAL_SEMAPHORE_HANDLE Handle) {
  CHECK_DXG_OPEN();

  D3DKMT_HANDLE syncobj = 0;
  HSAuint32 node_id = 0;
  DecodeExtSemHandle(Handle.handle, &syncobj, &node_id);

  if (syncobj == 0) return HSAKMT_STATUS_INVALID_HANDLE;

  WDDMDevice *device = get_wddmdev(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // Propagate WDDM destroy failure so the HSA layer can fail close()
  // and callers don't silently leak the imported sync object.
  if (!device->DestroySyncobj(syncobj))
    return HSAKMT_STATUS_ERROR;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueueSignalExternalSemaphore(
    HSA_QUEUEID QueueId,
    HSA_EXTERNAL_SEMAPHORE_HANDLE Handle,
    HSAuint64 Value) {
  CHECK_DXG_OPEN();
  if (QueueId == 0) return HSAKMT_STATUS_INVALID_HANDLE;

  D3DKMT_HANDLE syncobj = 0;
  HSAuint32 node_id = 0;
  DecodeExtSemHandle(Handle.handle, &syncobj, &node_id);
  if (syncobj == 0) return HSAKMT_STATUS_INVALID_HANDLE;

  WDDMDevice *device = get_wddmdev(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // HSA_QUEUEID is a (WDDMQueue *); reverse-cast matches dxg/queues.cpp.
  auto *queue = reinterpret_cast<WDDMQueue *>(QueueId);

  // The syncobj belongs to the semaphore's node; issuing on a queue from a
  // different adapter mixes mismatched handles. Require the nodes to match.
  if (queue->device == nullptr ||
      queue->device->NodeId() != static_cast<int>(node_id))
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // GpuSignal takes the value by pointer; keep it in an addressable local.
  uint64_t fence_value = Value;
  if (!device->GpuSignal(queue->context, &syncobj, &fence_value, 1))
    return HSAKMT_STATUS_ERROR;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueueWaitExternalSemaphore(
    HSA_QUEUEID QueueId,
    HSA_EXTERNAL_SEMAPHORE_HANDLE Handle,
    HSAuint64 Value) {
  CHECK_DXG_OPEN();
  if (QueueId == 0) return HSAKMT_STATUS_INVALID_HANDLE;

  D3DKMT_HANDLE syncobj = 0;
  HSAuint32 node_id = 0;
  DecodeExtSemHandle(Handle.handle, &syncobj, &node_id);
  if (syncobj == 0) return HSAKMT_STATUS_INVALID_HANDLE;

  WDDMDevice *device = get_wddmdev(node_id);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  auto *queue = reinterpret_cast<WDDMQueue *>(QueueId);

  // Same cross-adapter guard as the signal path.
  if (queue->device == nullptr ||
      queue->device->NodeId() != static_cast<int>(node_id))
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  // GpuWait takes the WDDMQueue * directly (pulling .context out
  // internally), unlike GpuSignal which takes the bare D3DKMT_HANDLE.
  // Keep fence_value alive for the ioctl.
  uint64_t fence_value = Value;
  if (!device->GpuWait(queue, &syncobj, &fence_value, 1))
    return HSAKMT_STATUS_ERROR;
  return HSAKMT_STATUS_SUCCESS;
}
