/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <cinttypes>
#include <cstddef>
#include <chrono>
#include <cerrno>
#include <vector>

#if !defined(WIN32)
#include <sys/eventfd.h>
#include <poll.h>
#include <unistd.h>
#endif

#include "impl/wddm/device.h"
#include "impl/wddm/event.h"

using namespace std::chrono;

namespace wsl {
namespace thunk {

#if defined(WIN32)
// ================================================================================================
EventWin::EventWin() : os_event_(nullptr), is_reference_(false) {
  EventId = 0;
  memset(&EventData, 0, sizeof(EventData));
}

// ================================================================================================
EventWin::~EventWin() {
  // Force device 0, since KMD should handle multiple devices.
  WDDMDevice* device = WddmDevice(0);  // Event->EventData.HWData3
  assert(device && "Couldn't obtain a device!");
  if (EventId != 0) {
    if (!device->UnregisterEvent(EventId, os_event_)) {
      pr_err("KMD deregister failed!");
    }
    if (os_event_ != nullptr) {
      CloseHandle(os_event_);
    }
    os_event_ = nullptr;
  }
}

// ================================================================================================
bool EventWin::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
  // Allocate OS specific events to handle HSA event, force device 0 and KMD should handle
  // multiiple devices.
  WDDMDevice* device = WddmDevice(0);  // EventDesc->NodeId
  assert(device && "Couldn't obtain a device!");
  // Allocate OS event
  SECURITY_ATTRIBUTES attributes = {};
  os_event_ = CreateEventW(&attributes, false, false, nullptr);
  if (os_event_ == nullptr) {
    pr_debug("CreateEventW call failed\n");
    return false;
  }

  // Register OS event in KMD
  EventId = device->RegisterEvent(event_desc.EventType, os_event_, &EventData.HWData2);
  if (EventId == 0) {
    // KMD ran out of slots or failed
    CloseHandle(os_event_);
    os_event_ = nullptr;
    return false;
  }
  EventData.EventType = event_desc.EventType;
  // ROCR doesn't use HWData1 or HWData3 fields, so save NodeId here...
  EventData.HWData3 = event_desc.NodeId;

  EventData.EventData.SyncVar.SyncVar.UserData = event_desc.SyncVar.SyncVar.UserData;
  EventData.EventData.SyncVar.SyncVarSize = event_desc.SyncVar.SyncVarSize;
  return true;
}

// ================================================================================================
bool EventWin::Set() const {
  // Windows function returns non-zero values to indicate success.
  if (SetEvent(os_event_) == FALSE) {
    pr_err("OS set event failed!");
    return false;
  }
  return true;
}

// ================================================================================================
bool EventWin::Reset() const {
  // Windows function returns non-zero values to indicate success.
  if (ResetEvent(os_event_) == FALSE) {
    return false;
  }
  return true;
}

// ================================================================================================
bool EventWin::Open(EventHandle handle, bool isReference) {
  return true;
}

// ================================================================================================
bool EventWin::Wait(std::chrono::duration<float> timeout  // max time to wait
  ) const {
  const uint32_t retCode =
      WaitForSingleObject(os_event_, duration_cast<milliseconds>(timeout).count());
  switch (retCode) {
    case WAIT_OBJECT_0:
      break;

    case WAIT_ABANDONED:
      break;

    case WAIT_TIMEOUT:
      break;

    case WAIT_FAILED:
      break;

    default:
      break;
  }
  return true;
}
#else
// ================================================================================================
EventLnx::EventLnx() : efd_(-1), syncobj_(0) {
  EventId = 0;
  memset(&EventData, 0, sizeof(EventData));
}

// ================================================================================================
EventLnx::~EventLnx() {
  // Force device 0, since KMD should handle multiple devices.
  WDDMDevice* device = WddmDevice(0);  // Event->EventData.HWData3
  assert(device && "Couldn't obtain a device!");
  if (EventId != 0) {
    if (!device->DestroyEvent(EventId, efd_, syncobj_)) {
      pr_err("KMD deregister failed!\n");
    }
    syncobj_ = 0;
    efd_ = -1;
  }
}

// ================================================================================================
bool EventLnx::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
  // Allocate OS specific events to handle HSA event, force device 0 and KMD should handle
  // multiiple devices.
  WDDMDevice* device = WddmDevice(0);  // EventDesc->NodeId
  assert(device && "Couldn't obtain a device!");

  if (!device->CreateEvent(&efd_, event_desc.EventType, &EventId, &EventData.HWData2,
                           &syncobj_)) {
    return false;
  }
  EventData.EventType = event_desc.EventType;
  // ROCR doesn't use HWData1 or HWData3 fields, so save NodeId here...
  EventData.HWData3 = event_desc.NodeId;

  EventData.EventData.SyncVar.SyncVar.UserData = event_desc.SyncVar.SyncVar.UserData;
  EventData.EventData.SyncVar.SyncVarSize = event_desc.SyncVar.SyncVarSize;
  return true;
}

// ================================================================================================
bool EventLnx::Set() const {
  const uint64_t one = 1;
  if (write(efd_, &one, sizeof(one)) != static_cast<ssize_t>(sizeof(one))) {
    pr_err("eventfd write failed!");
    return false;
  }
  return true;
}

// ================================================================================================
bool EventLnx::Reset() const {
  // An eventfd counter stays readable until drained; loop until EAGAIN so a
  // subsequent poll() blocks until the event is signaled again.
  uint64_t value;
  while (read(efd_, &value, sizeof(value)) == static_cast<ssize_t>(sizeof(value))) {}
  return true;
}

// ================================================================================================
bool EventLnx::Wait(std::chrono::duration<float> timeout  // max time to wait
  ) const {
  HsaEvent* self = const_cast<EventLnx*>(this);
  HSAKMT_STATUS status = WaitOnMultipleEvents(
      &self, 1, true, static_cast<uint32_t>(duration_cast<milliseconds>(timeout).count()));
  return status == HSAKMT_STATUS_SUCCESS;
}

// ================================================================================================
HSAKMT_STATUS EventLnx::WaitOnMultipleEvents(HsaEvent* events[], uint32_t num_elems,
                                             bool wait_all, uint32_t msec) {
  if (num_elems == 0)
    return HSAKMT_STATUS_SUCCESS;

  constexpr uint32_t kWaitTimeout = 6000;  // 6 seconds
  if (!dxg_runtime->disable_wait_timeout_ && msec > kWaitTimeout)
    msec = kWaitTimeout;

  std::vector<struct pollfd> pfds(num_elems);
  std::vector<bool> signaled(num_elems, false);
  for (uint32_t i = 0; i < num_elems; i++) {
    pfds[i].fd = reinterpret_cast<EventLnx*>(events[i])->efd_;
    pfds[i].events = POLLIN;
    pfds[i].revents = 0;
  }

  auto done = [&]() -> bool {
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_elems; i++)
      if (signaled[i])
        count++;
    return wait_all ? (count == num_elems) : (count > 0);
  };

  constexpr uint32_t kInfinite = 0xFFFFFFFF;
  const bool infinite = (msec == kInfinite);
  auto deadline = steady_clock::now() + milliseconds(msec);

  while (true) {
    int timeout_ms = -1;
    if (!infinite) {
      auto now = steady_clock::now();
      timeout_ms = now >= deadline
                       ? 0
                       : static_cast<int>(
                             duration_cast<milliseconds>(deadline - now).count());
    }

    for (uint32_t i = 0; i < num_elems; i++)
      pfds[i].revents = 0;

    int ret = poll(pfds.data(), num_elems, timeout_ms);
    if (ret < 0) {
      if (errno == EINTR)
        continue;
      pr_err("poll fail %d\n", errno);
      return HSAKMT_STATUS_WAIT_FAILURE;
    }
    if (ret == 0)
      return done() ? HSAKMT_STATUS_SUCCESS : HSAKMT_STATUS_WAIT_TIMEOUT;

    // An eventfd stays readable until Reset() drains it, so accumulate the set
    // of signaled events across poll() iterations to honor wait_all.
    for (uint32_t i = 0; i < num_elems; i++) {
      if (pfds[i].revents & (POLLIN | POLLERR | POLLHUP))
        signaled[i] = true;
    }

    if (done())
      return HSAKMT_STATUS_SUCCESS;
    if (!infinite && steady_clock::now() >= deadline)
      return HSAKMT_STATUS_WAIT_TIMEOUT;
  }
}
#endif

}  // namespace thunk
}  // namespace wsl

