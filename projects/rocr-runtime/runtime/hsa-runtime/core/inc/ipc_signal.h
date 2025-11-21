////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTME_CORE_INC_IPC_SIGNAL_H_
#define HSA_RUNTME_CORE_INC_IPC_SIGNAL_H_

#include <atomic>
#include <utility>

#include "core/inc/signal.h"
#include "core/inc/default_signal.h"
#include "core/util/locks.h"

namespace rocr {
namespace core {

/// @brief Container for ipc shared memory.
class SharedMemory {
 public:
  SharedMemory(const hsa_amd_ipc_memory_t* handle, size_t len);
  ~SharedMemory();
  SharedMemory(SharedMemory&&);

  void* ptr() const { return ptr_; }

 private:
  void* ptr_;
};

/// @brief Container for ipc signal abi block.
class SharedMemorySignal {
 public:
  explicit SharedMemorySignal(const hsa_amd_ipc_memory_t* handle) : signal_(handle, 4096) {
    if (!signal()->IsValid())
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "IPC Signal handle is invalid.");
  }
  SharedSignal* signal() const { return reinterpret_cast<SharedSignal*>(signal_.ptr()); }

 private:
  SharedMemory signal_;
};

/// @brief Memory only signal using a shared memory ABI block.
class IPCSignal : private SharedMemorySignal, public BusyWaitSignal {
 public:
  /// @brief Creates a sharable handle for an IPC enabled signal.
  static void CreateHandle(Signal* signal, hsa_amd_ipc_signal_t* ipc_handle);

  /// @brief Opens an IPC signal from its IPC handle.
  static Signal* Attach(const hsa_amd_ipc_signal_t* ipc_handle);

  /// @brief Determines if a Signal* can be safely converted to BusyWaitSignal*
  /// via static_cast.
  static __forceinline bool IsType(Signal* ptr) { return ptr->IsType(&rtti_id()); }

 protected:
  bool _IsA(rtti_t id) const {
    if (id == &rtti_id()) return true;
    return BusyWaitSignal::_IsA(id);
  }

 private:
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
      return rtti_id_;
  }
  static KernelMutex lock_;

  explicit IPCSignal(SharedMemorySignal&& abi_block)
      : SharedMemorySignal(std::move(abi_block)), BusyWaitSignal(signal(), true) {}

  DISALLOW_COPY_AND_ASSIGN(IPCSignal);
};

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTME_CORE_INC_IPC_SIGNAL_H_
