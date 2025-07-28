/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
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

#include <cassert>
#include <cstring>


static uint32_t runtime_capabilities_mask = 0;

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgRegister(HSAuint32 NodeId) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgUnregister(HSAuint32 NodeId) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgWavefrontControl(
    HSAuint32 NodeId, HSA_DBG_WAVEOP Operand, HSA_DBG_WAVEMODE Mode,
    HSAuint32 TrapId, HsaDbgWaveMessage *DbgWaveMsgRing) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgAddressWatch(
    HSAuint32 NodeId, HSAuint32 NumWatchPoints, HSA_DBG_WATCH_MODE WatchMode[],
    void *WatchAddress[], HSAuint64 WatchMask[], HsaEvent *WatchEvent[]) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCheckRuntimeDebugSupport(void) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRuntimeEnable(void *rDebug, bool setupTtmp) {
  HSAKMT_STATUS result = hsaKmtCheckRuntimeDebugSupport();

  if (result)
    return result;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRuntimeDisable(void) {
  HSAKMT_STATUS result = hsaKmtCheckRuntimeDebugSupport();

  if (result)
    return HSAKMT_STATUS_SUCCESS;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetRuntimeCapabilities(HSAuint32 *caps_mask) {
  CHECK_DXG_OPEN();
  *caps_mask = runtime_capabilities_mask;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgEnable(void **runtime_info,
                                        HSAuint32 *data_size) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}
HSAKMT_STATUS HSAKMTAPI hsaKmtDbgDisable(void) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgGetDeviceData(void **data,
                                               HSAuint32 *n_entries,
                                               HSAuint32 *entry_size) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDbgGetQueueData(void **data, HSAuint32 *n_entries,
                                              HSAuint32 *entry_size,
                                              bool suspend_queues) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDebugTrapIoctl(struct kfd_ioctl_dbg_trap_args *args, HSA_QUEUEID *Queues,
                     HSAuint64 *DebugReturn) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}
