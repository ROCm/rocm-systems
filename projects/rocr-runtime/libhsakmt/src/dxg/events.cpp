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
#include "impl/wddm/device.h"
#include "impl/wddm/event.h"
#include "hsa-runtime/inc/amd_hsa_signal.h"

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateEvent(HsaEventDescriptor *EventDesc,
                                          bool ManualReset, bool IsSignaled,
                                          HsaEvent **Event) {
  CHECK_DXG_OPEN();
  if (EventDesc->EventType >= HSA_EVENTTYPE_MAXID) {
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }
  // Allocate thunk HSA event
  std::unique_ptr<wsl::thunk::Event> event(new wsl::thunk::Event());
  if (!event) {
    return HSAKMT_STATUS_ERROR;
  }
  // Initialize the HSA event class
  if (!event->Init(*EventDesc)) {
    return HSAKMT_STATUS_ERROR;
  }

  *Event = event.get();
  event.release();
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyEvent(HsaEvent *Event) {
  CHECK_DXG_OPEN();
  if (Event == nullptr) {
    return HSAKMT_STATUS_SUCCESS;
  }
  delete static_cast<wsl::thunk::Event*>(Event);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetEvent(HsaEvent *Event) {
  CHECK_DXG_OPEN();
  if (Event == nullptr) {
    return HSAKMT_STATUS_SUCCESS;
  }
  static_cast<wsl::thunk::Event*>(Event)->Set();

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtResetEvent(HsaEvent *Event) {
  CHECK_DXG_OPEN();
  if (Event == nullptr) {
    return HSAKMT_STATUS_SUCCESS;
  }
  static_cast<wsl::thunk::Event*>(Event)->Reset();

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryEventState(HsaEvent *Event) {
  CHECK_DXG_OPEN();
  if (Event == nullptr) {
    return HSAKMT_STATUS_SUCCESS;
  }


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
  if (Events == nullptr) {
    return HSAKMT_STATUS_SUCCESS;
  }

  if (NumEvents == 1 && Events[0] == nullptr) {
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    return HSAKMT_STATUS_SUCCESS;
  }
  HSAKMT_STATUS status =
      wsl::thunk::WDDMDevice::WaitOnMultipleEvents(Events, NumEvents, WaitOnAll, Milliseconds);

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenSMI(HSAuint32 NodeId, int *fd) {
  CHECK_DXG_OPEN();
  pr_debug("node id %d\n", NodeId);

  return HSAKMT_STATUS_SUCCESS;
}
