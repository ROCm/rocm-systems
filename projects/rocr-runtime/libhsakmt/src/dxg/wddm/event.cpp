/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <cinttypes>
#include <cstddef>
#include <chrono>

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

// ================================================================================================
Event::Event() : os_event_(nullptr), is_reference_(false) {
  EventId = 0;
  memset(&EventData, 0, sizeof(EventData));
}

#if defined(WIN32)
// ================================================================================================
Event::~Event() {
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
bool Event::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
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
bool Event::Set() const {
  // Windows function returns non-zero values to indicate success.
  if (SetEvent(os_event_) == FALSE) {
    pr_err("OS set event failed!");
    return false;
  }
  return true;
}

// ================================================================================================
bool Event::Reset() const {
  // Windows function returns non-zero values to indicate success.
  if (ResetEvent(os_event_) == FALSE) {
    return false;
  }
  return true;
}

// ================================================================================================
bool Event::Open(EventHandle handle, bool isReference) {
  return true;
}

// ================================================================================================
bool Event::Wait(std::chrono::duration<float> timeout  // max time to wait
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

static int EventFd(void* h) { return static_cast<int>(reinterpret_cast<intptr_t>(h)); }
static void* FdToHandle(int fd) { return reinterpret_cast<void*>(static_cast<intptr_t>(fd)); }

// ================================================================================================
Event::~Event() {
  // Force device 0, since KMD should handle multiple devices.
  int fd = EventFd(os_event_);
  if (fd >= 0) {
    WDDMDevice* device = WddmDevice(0);  // Event->EventData.HWData3
    if (device && EventId != 0) {
      device->UnregisterEvent(EventId, os_event_);
    }
    close(fd);
  }
  os_event_ = FdToHandle(-1);
}

// ================================================================================================
bool Event::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
  // Allocate OS specific events to handle HSA event, force device 0 and KMD should handle
  // multiple devices.
  WDDMDevice* device = WddmDevice(0);  // EventDesc->NodeId
  assert(device && "Couldn't obtain a device!");

  int fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) {
    pr_err("eventfd() call failed\n");
    return false;
  }
  os_event_ = FdToHandle(fd);

  // Register OS event in KMD
  EventId = device->RegisterEvent(event_desc.EventType, os_event_, &EventData.HWData2);
  if (EventId == 0) {
    pr_debug("RegisterEvent returned 0 (KMD event registration unavailable on Linux)\n");
  }
  EventData.EventType = event_desc.EventType;
  EventData.HWData3 = event_desc.NodeId;

  EventData.EventData.SyncVar.SyncVar.UserData = event_desc.SyncVar.SyncVar.UserData;
  EventData.EventData.SyncVar.SyncVarSize = event_desc.SyncVar.SyncVarSize;
  return true;
}

// ================================================================================================
bool Event::Set() const {
  int fd = EventFd(os_event_);
  if (fd < 0) {
    pr_err("OS set event failed!");
    return false;
  }
  uint64_t val = 1;
  return write(fd, &val, sizeof(val)) == sizeof(val);
}

// ================================================================================================
bool Event::Reset() const {
  int fd = EventFd(os_event_);
  if (fd < 0) return false;
  uint64_t val;
  while (read(fd, &val, sizeof(val)) == sizeof(val)) {}
  return true;
}

// ================================================================================================
bool Event::Open(EventHandle handle, bool isReference) {
  os_event_ = handle;
  is_reference_ = isReference;
  return true;
}

// ================================================================================================
bool Event::Wait(std::chrono::duration<float> timeout  // max time to wait
  ) const {
  int fd = EventFd(os_event_);
  if (fd < 0) return false;

  const int timeout_ms = static_cast<int>(
      duration_cast<milliseconds>(timeout).count());

  struct pollfd pfd = {};
  pfd.fd = fd;
  pfd.events = POLLIN;

  int ret = poll(&pfd, 1, timeout_ms);
  if (ret > 0 && (pfd.revents & POLLIN)) {
    uint64_t val;
    read(fd, &val, sizeof(val));
    return true;
  }
  return false;
}
#endif

}  // namespace thunk
}  // namespace wsl

