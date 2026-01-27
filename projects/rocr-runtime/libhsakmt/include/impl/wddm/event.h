////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

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
