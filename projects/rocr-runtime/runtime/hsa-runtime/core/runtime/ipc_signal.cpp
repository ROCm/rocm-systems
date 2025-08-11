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

#include "core/inc/ipc_signal.h"

#include <utility>

#include "core/inc/runtime.h"
#include "core/inc/exceptions.h"

namespace rocr {
namespace core {

KernelMutex IPCSignal::lock_;

SharedMemory::SharedMemory(const hsa_amd_ipc_memory_t* handle, size_t len) {
  hsa_status_t err = Runtime::runtime_singleton_->IPCAttach(handle, len, 0, NULL, &ptr_);
  if (err != HSA_STATUS_SUCCESS) throw AMD::hsa_exception(err, "IPC memory attach failed.");
}

SharedMemory::SharedMemory(SharedMemory&& rhs) {
  ptr_ = rhs.ptr_;
  rhs.ptr_ = nullptr;
}

SharedMemory::~SharedMemory() {
  if (ptr_ == nullptr) return;
  auto err = Runtime::runtime_singleton_->IPCDetach(ptr_);
  assert(err == HSA_STATUS_SUCCESS && "IPC detach failed.");
}

void IPCSignal::CreateHandle(Signal* signal, hsa_amd_ipc_signal_t* ipc_handle) {
  if (!signal->isIPC())
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "Signal must be IPC enabled.");
  SharedSignal* shared = SharedSignal::Convert(Convert(signal));
  hsa_status_t err = Runtime::runtime_singleton_->IPCCreate(shared, 4096, ipc_handle);
  if (err != HSA_STATUS_SUCCESS) throw AMD::hsa_exception(err, "IPC memory create failed.");
}

Signal* IPCSignal::Attach(const hsa_amd_ipc_signal_t* ipc_signal_handle) {
  SharedMemorySignal shared(ipc_signal_handle);

  if (!(shared.signal()->IsIPC()))
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "IPC memory does not contain an IPC signal abi block.");

  hsa_signal_t handle = SharedSignal::Convert(shared.signal());

  ScopedAcquire<KernelMutex> lock(&lock_);
  Signal* ret = core::Signal::DuplicateHandle(handle);
  if (ret == nullptr) ret = new IPCSignal(std::move(shared));
  return ret;
}

}  // namespace core
}  // namespace rocr
