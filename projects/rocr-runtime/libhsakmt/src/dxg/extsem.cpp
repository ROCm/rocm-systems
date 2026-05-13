/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
