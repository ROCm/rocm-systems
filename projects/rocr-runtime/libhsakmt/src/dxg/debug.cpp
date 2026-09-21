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

#include "topology.hpp"

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

  // The caller owns the topology snapshot the WDDMDevices belong to; this probe
  // deliberately takes no reference of its own. Acquiring here would enumerate
  // the adapters only to destroy them again on return, leaving the caller to
  // build the very same list a second time.
  //
  // An empty device list therefore means no snapshot is held (or the machine
  // has no adapter). Say so, rather than letting the loop below fall through
  // and report support for devices nobody has looked at.
  if (dxg_topology->wdevices_.empty()) {
    pr_warn_once(
        "DXG system properties must be acquired before checking runtime debug support\n");
    return HSAKMT_STATUS_NOT_SUPPORTED;
  }

  for (auto&& device : dxg_topology->wdevices_) {
    if (Wkmi::KmdDbgVersion version;
        !device->GetKmdDbgVersion(&version) || version.major != 1 || version.minor < 2) {
       return HSAKMT_STATUS_NOT_SUPPORTED;
     }
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRuntimeEnable(void *rDebug, bool setupTtmp) {
  /* No acquire here either: the caller holds the snapshot across this call.
   * ROCr's KfdDriver takes it in Init() before enabling, precisely so the
   * device list this walks is the one BuildTopology() goes on to use. The
   * probe reports NOT_SUPPORTED when that list is empty, so a caller that has
   * not acquired never reaches the loop below.
   */
  HSAKMT_STATUS result = hsaKmtCheckRuntimeDebugSupport();

  if (result)
    return result;

  for (auto&& device : dxg_topology->wdevices_) {
    if (!device->RegisterRuntimeState(1 /* always valid debug state */, rDebug, setupTtmp)) {
      return HSAKMT_STATUS_NOT_SUPPORTED;
    }
  }

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

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetCoreRuntimeInfo(struct kfd_runtime_info *runtime_info) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetCoreDeviceInfo(HSAuint32 gpu_id,
                        struct kfd_dbg_device_info_entry *device_info) {
  CHECK_DXG_OPEN();
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
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

