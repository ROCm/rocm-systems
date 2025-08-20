/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "hip_event.hpp"
#if !defined(_MSC_VER)
#include <unistd.h>
#else
#include <io.h>
#endif

// ================================================================================================
namespace hip {

hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags);

bool IPCEvent::createIpcEventShmemIfNeeded() {
  if (ipc_evt_.ipc_shmem_) {
    // ipc_shmem_ already created, no need to create it again
    return true;
  }

#if !defined(_MSC_VER)
  static std::atomic<int> counter{0};
  ipc_evt_.ipc_name_ = "/hip_" + std::to_string(getpid()) + "_" + std::to_string(counter++);
#else
  char name_template[] = "/hip_XXXXXX";
  _mktemp_s(name_template, sizeof(name_template));
  ipc_evt_.ipc_name_ = name_template;
  ipc_evt_.ipc_name_.replace(0, 5, "/hip_");
#endif

  if (!amd::Os::MemoryMapFileTruncated(
          ipc_evt_.ipc_name_.c_str(),
          const_cast<const void**>(reinterpret_cast<void**>(&(ipc_evt_.ipc_shmem_))),
          sizeof(hip::ihipIpcEventShmem_t))) {
    return false;
  }

  ipc_evt_.ipc_shmem_->owners = 1;
  ipc_evt_.ipc_shmem_->read_index = -1;
  ipc_evt_.ipc_shmem_->write_index = 0;
  for (uint32_t sig_idx = 0; sig_idx < IPC_SIGNALS_PER_EVENT; ++sig_idx) {
    ipc_evt_.ipc_shmem_->signal[sig_idx] = 0;
  }

  // device sets 0 to this ptr when the ipc event is completed
  hipError_t status =
      ihipHostRegister(&ipc_evt_.ipc_shmem_->signal, sizeof(uint32_t) * IPC_SIGNALS_PER_EVENT, 0);
  if (status != hipSuccess) {
    return false;
  }
  return true;
}

// ================================================================================================
hipError_t IPCEvent::query() {
  if (ipc_evt_.ipc_shmem_) {
    int prev_read_idx = ipc_evt_.ipc_shmem_->read_index;
    int offset = (prev_read_idx % IPC_SIGNALS_PER_EVENT);
    if (ipc_evt_.ipc_shmem_->read_index < prev_read_idx + IPC_SIGNALS_PER_EVENT &&
        ipc_evt_.ipc_shmem_->signal[offset] != 0) {
      return hipErrorNotReady;
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::synchronize() {
  if (ipc_evt_.ipc_shmem_) {
    int prev_read_idx = ipc_evt_.ipc_shmem_->read_index;
    if (prev_read_idx >= 0) {
      int offset = (prev_read_idx % IPC_SIGNALS_PER_EVENT);
      while ((ipc_evt_.ipc_shmem_->read_index < prev_read_idx + IPC_SIGNALS_PER_EVENT) &&
             (ipc_evt_.ipc_shmem_->signal[offset] != 0)) {
        amd::Os::sleep(1);
      }
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::streamWait(hip::Stream* stream, uint flags) {
  int offset = ipc_evt_.ipc_shmem_->read_index;
  hipError_t status =
      ihipStreamOperation(reinterpret_cast<hipStream_t>(stream), ROCCLR_COMMAND_STREAM_WAIT_VALUE,
                          &(ipc_evt_.ipc_shmem_->signal[offset]), 0, 1, 1, sizeof(uint32_t));
  return status;
}

// ================================================================================================
hipError_t IPCEvent::recordCommand(amd::Command*& command, amd::HostQueue* stream, uint32_t flags,
                                   bool batch_flush) {
  command = new amd::Marker(*stream, kMarkerDisableFlush);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  amd::Event& tEvent = command->event();
  createIpcEventShmemIfNeeded();
  int write_index = ipc_evt_.ipc_shmem_->write_index++;
  int offset = write_index % IPC_SIGNALS_PER_EVENT;
  while (ipc_evt_.ipc_shmem_->signal[offset] != 0) {
    amd::Os::sleep(1);
  }
  // Lock signal.
  ipc_evt_.ipc_shmem_->signal[offset] = 1;
  ipc_evt_.ipc_shmem_->owners_device_id = deviceId();
  command->enqueue();

  // Set event_ in order to release marked command when event is destroyed
  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();

  // device writes 0 to signal after the hipEventRecord command is completed
  // the signal value is checked by WaitThenDecrementSignal cb
  hipError_t status =
      ihipStreamOperation(reinterpret_cast<hipStream_t>(stream), ROCCLR_COMMAND_STREAM_WRITE_VALUE,
                          &(ipc_evt_.ipc_shmem_->signal[offset]), 0, 0, 0, sizeof(uint32_t));

  if (status != hipSuccess) {
    return status;
  }

  // Update read index to indicate new signal.
  int expected = write_index - 1;
  while (!ipc_evt_.ipc_shmem_->read_index.compare_exchange_weak(expected, write_index)) {
    amd::Os::sleep(1);
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::GetHandle(ihipIpcEventHandle_t* handle) {
  if (!createIpcEventShmemIfNeeded()) {
    return hipErrorInvalidValue;
  }
  ipc_evt_.ipc_shmem_->owners_device_id = deviceId();
  ipc_evt_.ipc_shmem_->owners_process_id = amd::Os::getProcessId();
  memset(handle->shmem_name, 0, HIP_IPC_HANDLE_SIZE);
  ipc_evt_.ipc_name_.copy(handle->shmem_name, std::string::npos);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::OpenHandle(ihipIpcEventHandle_t* handle) {
  ipc_evt_.ipc_name_ = handle->shmem_name;
  if (!amd::Os::MemoryMapFileTruncated(ipc_evt_.ipc_name_.c_str(),
                                       (const void**)&(ipc_evt_.ipc_shmem_),
                                       sizeof(ihipIpcEventShmem_t))) {
    return hipErrorInvalidValue;
  }

  if (amd::Os::getProcessId() == ipc_evt_.ipc_shmem_->owners_process_id.load()) {
    // If this is in the same process, return error.
    return hipErrorInvalidContext;
  }

  ipc_evt_.ipc_shmem_->owners += 1;
  // device sets 0 to this ptr when the ipc event is completed
  hipError_t status = hipSuccess;
  status =
      ihipHostRegister(&ipc_evt_.ipc_shmem_->signal, sizeof(uint32_t) * IPC_SIGNALS_PER_EVENT, 0);
  return status;
}

// ================================================================================================
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  HIP_INIT_API(hipIpcGetEventHandle, handle, event);

  if (handle == nullptr || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  HIP_RETURN(e->GetHandle(reinterpret_cast<ihipIpcEventHandle_t*>(handle)));
}

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  HIP_INIT_API(hipIpcOpenEventHandle, event, handle);

  hipError_t status = hipSuccess;
  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  status = ihipEventCreateWithFlags(event, hipEventDisableTiming | hipEventInterprocess);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(*event);
  ihipIpcEventHandle_t* iHandle = reinterpret_cast<ihipIpcEventHandle_t*>(&handle);

  status = e->OpenHandle(iHandle);
  // Free the event in case of failure
  if (status != hipSuccess) {
    delete e;
  }
  HIP_RETURN(status);
}
}  // namespace hip
