/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hsakmt/hsakmt.h"

HSAKMT_STATUS HSAKMTAPI hsaKmtImportExternalSemaphore(
    HSAuint32 NodeId, void *NtHandle,
    HSA_EXTERNAL_SEMAPHORE_HANDLE_TYPE Type,
    HSA_EXTERNAL_SEMAPHORE_HANDLE *OutHandle) {
  (void)NodeId; (void)NtHandle; (void)Type; (void)OutHandle;
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyExternalSemaphore(
    HSA_EXTERNAL_SEMAPHORE_HANDLE Handle) {
  (void)Handle;
  return HSAKMT_STATUS_NOT_SUPPORTED;
}
