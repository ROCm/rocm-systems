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

// ================================================================================================
bool IPCEvent::createIpcEventShmemIfNeeded() {
  // Early return if shared memory already exists
  if (ipc_evt_.ipc_shmem_) {
    return true;
  }

  // Generate unique IPC name
#if !defined(_MSC_VER)
  static std::atomic<int> counter{0};
  ipc_evt_.ipc_name_ = "/hip_" + std::to_string(getpid()) + "_" + std::to_string(counter++);
#else
  char name_template[] = "/hip_XXXXXX";
  _mktemp_s(name_template, sizeof(name_template));
  ipc_evt_.ipc_name_ = name_template;
  ipc_evt_.ipc_name_.replace(0, 5, "/hip_");
#endif

  // Create memory-mapped file for shared memory
  auto** shmem_ptr = reinterpret_cast<void**>(&ipc_evt_.ipc_shmem_);
  if (!amd::Os::MemoryMapFileTruncated(ipc_evt_.ipc_name_.c_str(),
                                       const_cast<const void**>(shmem_ptr),
                                       sizeof(hip::ihipIpcEventShmem_t))) {
    return false;
  }

  // Initialize shared memory fields
  auto* const shmem = ipc_evt_.ipc_shmem_;
  shmem->owners = 1;
  shmem->read_index = -1;
  shmem->write_index = 0;
  std::fill_n(shmem->signal, IPC_SIGNALS_PER_EVENT, 0);

  // Register signal array with device
  constexpr size_t kSignalArraySize = sizeof(uint32_t) * IPC_SIGNALS_PER_EVENT;
  const auto status = ihipHostRegister(&shmem->signal, kSignalArraySize, 0);
  return status == hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::query() {
  if (ipc_evt_.ipc_shmem_) {
    const int prev_read_idx = ipc_evt_.ipc_shmem_->read_index;
    const int offset = prev_read_idx % IPC_SIGNALS_PER_EVENT;

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
  const int offset = ipc_evt_.ipc_shmem_->read_index;
  return ihipStreamOperation(
      reinterpret_cast<hipStream_t>(stream),
      ROCCLR_COMMAND_STREAM_WAIT_VALUE,
      &(ipc_evt_.ipc_shmem_->signal[offset]), 0, 1, 1,
      sizeof(uint32_t));
}

// ================================================================================================
hipError_t IPCEvent::recordCommand(amd::Command*& command, amd::HostQueue* stream, uint32_t flags,
                                   bool batch_flush) {
  command = new amd::Marker(*stream, kMarkerDisableFlush);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  createIpcEventShmemIfNeeded();

  // Allocate signal slot for this event
  auto* const shmem = ipc_evt_.ipc_shmem_;
  const int write_index = shmem->write_index++;
  const int offset = write_index % IPC_SIGNALS_PER_EVENT;
  auto& signal = shmem->signal[offset];

  // Wait for signal slot to become available
  while (signal != 0) {
    amd::Os::sleep(1);
  }

  // Lock signal and set device ID
  signal = 1;
  shmem->owners_device_id = deviceId();
  command->enqueue();

  // Set event_ to release marked command when event is destroyed
  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();

  // Device writes 0 to signal after hipEventRecord command completes
  const auto status = ihipStreamOperation(reinterpret_cast<hipStream_t>(stream),
                                          ROCCLR_COMMAND_STREAM_WRITE_VALUE, &signal, 0, 0, 0,
                                          sizeof(uint32_t));
  if (status != hipSuccess) {
    return status;
  }

  // Update read index to indicate new signal
  int expected = write_index - 1;
  while (!shmem->read_index.compare_exchange_weak(expected, write_index)) {
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

  // Map shared memory from IPC handle
  auto** shmem_ptr = reinterpret_cast<void**>(&ipc_evt_.ipc_shmem_);
  if (!amd::Os::MemoryMapFileTruncated(ipc_evt_.ipc_name_.c_str(),
                                       const_cast<const void**>(shmem_ptr),
                                       sizeof(ihipIpcEventShmem_t))) {
    return hipErrorInvalidValue;
  }

  auto* const shmem = ipc_evt_.ipc_shmem_;

  // Prevent opening in the same process
  const auto current_process_id = amd::Os::getProcessId();
  if (current_process_id == shmem->owners_process_id.load()) {
    return hipErrorInvalidContext;
  }

  shmem->owners += 1;

  // Register signal array with device
  constexpr size_t kSignalArraySize = sizeof(uint32_t) * IPC_SIGNALS_PER_EVENT;
  return ihipHostRegister(&shmem->signal, kSignalArraySize, 0);
}

// ================================================================================================
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  HIP_INIT_API(hipIpcGetEventHandle, handle, event);

  if (handle == nullptr || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto e = reinterpret_cast<hip::Event*>(event);
  HIP_RETURN(e->GetHandle(reinterpret_cast<ihipIpcEventHandle_t*>(handle)));
}

// ================================================================================================
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  HIP_INIT_API(hipIpcOpenEventHandle, event, handle);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Create IPC event with timing disabled
  constexpr uint32_t kIpcEventFlags = hipEventDisableTiming | hipEventInterprocess;
  auto status = ihipEventCreateWithFlags(event, kIpcEventFlags);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }

  auto* const e = reinterpret_cast<hip::Event*>(*event);
  auto* const iHandle = reinterpret_cast<ihipIpcEventHandle_t*>(&handle);

  const auto open_status = e->OpenHandle(iHandle);
  if (open_status != hipSuccess) {
    delete e;
  }
  HIP_RETURN(open_status);
}
}  // namespace hip
