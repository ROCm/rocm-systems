/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>
#include <cinttypes>
#include <cstddef>
#include <chrono>

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
// ================================================================================================
Event::~Event() {
  // Force device 0, since KMD should handle multiple devices.
  WDDMDevice* device = WddmDevice(0);  // Event->EventData.HWData3
  assert(device && "Couldn't obtain a device!");
  if (EventId != 0) {
    os_event_ = nullptr;
  }
  assert(!"Unimplemented!");
}

// ================================================================================================
bool Event::Init(const HsaEventDescriptor& event_desc, const wchar_t* pName) {
  // Allocate OS specific events to handle HSA event, force device 0 and KMD should handle
  // multiiple devices.
  WDDMDevice* device = WddmDevice(0);  // EventDesc->NodeId
  assert(device && "Couldn't obtain a device!");
  assert(!"Unimplemented!");
  return true;
}

// ================================================================================================
bool Event::Set() const {
  assert(!"Unimplemented!");
  return true;
}

// ================================================================================================
bool Event::Reset() const {
  assert(!"Unimplemented!");
  return true;
}

// ================================================================================================
bool Event::Open(EventHandle handle, bool isReference) {
  assert(!"Unimplemented!");
  return true;
}

// ================================================================================================
bool Event::Wait(std::chrono::duration<float> timeout  // max time to wait
  ) const {
  assert(!"Unimplemented!");
  return true;
}
#endif

}  // namespace thunk
}  // namespace wsl

