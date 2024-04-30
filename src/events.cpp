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

#include <cstdio>
#include <cassert>
#include <thread>
#include <chrono>
#include "libhsakmt.h"

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateEvent(HsaEventDescriptor *EventDesc,
                                          bool ManualReset, bool IsSignaled,
                                          HsaEvent **Event) {
  CHECK_DXG_OPEN();

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyEvent(HsaEvent *Event) {
  CHECK_DXG_OPEN();

  if (!Event)
    return HSAKMT_STATUS_INVALID_HANDLE;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetEvent(HsaEvent *Event) {
  CHECK_DXG_OPEN();

  if (!Event)
    return HSAKMT_STATUS_INVALID_HANDLE;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtResetEvent(HsaEvent *Event) {
  CHECK_DXG_OPEN();

  if (!Event)
    return HSAKMT_STATUS_INVALID_HANDLE;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryEventState(HsaEvent *Event) {
  CHECK_DXG_OPEN();

  if (!Event)
    return HSAKMT_STATUS_INVALID_HANDLE;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnEvent(HsaEvent *Event,
                                          HSAuint32 Milliseconds) {
  return hsaKmtWaitOnEvent_Ext(Event, Milliseconds, NULL);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnEvent_Ext(HsaEvent *Event,
                                              HSAuint32 Milliseconds,
                                              uint64_t *event_age) {
  if (!Event)
    return HSAKMT_STATUS_INVALID_HANDLE;

  return hsaKmtWaitOnMultipleEvents_Ext(&Event, 1, true, Milliseconds,
                                        event_age);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnMultipleEvents(HsaEvent *Events[],
                                                   HSAuint32 NumEvents,
                                                   bool WaitOnAll,
                                                   HSAuint32 Milliseconds) {
  return hsaKmtWaitOnMultipleEvents_Ext(Events, NumEvents, WaitOnAll,
                                        Milliseconds, NULL);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtWaitOnMultipleEvents_Ext(HsaEvent *Events[],
                                                       HSAuint32 NumEvents,
                                                       bool WaitOnAll,
                                                       HSAuint32 Milliseconds,
                                                       uint64_t *event_age) {
  CHECK_DXG_OPEN();

  if (!Events)
    return HSAKMT_STATUS_INVALID_HANDLE;

  if (NumEvents == 1 && Events[0] == nullptr) {
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    return HSAKMT_STATUS_SUCCESS;
  }

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenSMI(HSAuint32 NodeId, int *fd) {
  CHECK_DXG_OPEN();

  pr_debug("[%s] node %d\n", __func__, NodeId);
  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}
