/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hsakmt/hsakmt.h"
#include "impl/wddm/device.h"
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

  // Only OPAQUE_WIN32 (NT handle) is wired today. The other enum
  // values are accepted in the typedef for forward compatibility but
  // must be rejected here.
  if (Type != HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32)
    return HSAKMT_STATUS_NOT_SUPPORTED;

  WDDMDevice *device = get_wddmdev(NodeId);
  if (device == nullptr) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  D3DKMT_HANDLE syncobj = 0;
  if (!device->OpenSyncobjFromNtHandle(NtHandle, &syncobj))
    return HSAKMT_STATUS_ERROR;

  OutHandle->handle = EncodeExtSemHandle(syncobj, NodeId);
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
