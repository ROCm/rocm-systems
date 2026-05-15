/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cinttypes>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <utility>
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "impl/wddm/gpu_memory.h"
#include "hsa-runtime/inc/hsa_ext_amd.h"
#include "hsa-runtime/inc/amd_hsa_queue.h"
#include "hsa-runtime/inc/amd_hsa_signal.h"
#include "impl/wddm/cmd_util.h"

namespace wsl {
namespace thunk {

/**
 ***********************************************************************************************************************
 * @brief Interrupt based event for HSA signals implementation
 *
 * Event objects start out in the _reset_ state.
 ***********************************************************************************************************************
 */
class Event final : public HsaEvent {
 public:
  Event();
  ~Event();

  // @note: No virtual methods are allowed in this class, unless THUNK will expose it to the user.

  bool Init(const HsaEventDescriptor& event_desc, const wchar_t* pName = nullptr);
  bool Set() const;
  bool Reset() const;
  bool Wait(std::chrono::duration<float> timeout) const;

  typedef void* EventHandle;

  /// Returns a handle to the actual OS event primitive associated with this object.
  EventHandle GetHandle() const { return os_event_; }

  /// Open event handle.
  bool Open(EventHandle handle, bool isReference);

 private:
  EventHandle os_event_; // OS-specific event handle.
  bool is_reference_;    // If true, the event is a global sharing object handle (not a duplicate)
                         // which is imported from external, so it can't be closed in the currect
                         // destructor, and can only be closed by the creater.

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

}  // namespace thunk
}  // namespace wsl
