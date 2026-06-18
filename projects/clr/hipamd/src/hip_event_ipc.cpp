/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
hipError_t ihipCreateIpcEventByType(hipEvent_t* event, ihipIpcEventHandleType type);
void ihipDestroyIpcEvent(hipEvent_t event);

// ================================================================================================
bool IPCEventEmulated::createIpcEventShmemIfNeeded() {
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
hipError_t IPCEventEmulated::query() {
  std::scoped_lock lock(lock_);
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
hipError_t IPCEventEmulated::synchronize() {
  std::scoped_lock lock(lock_);
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
hipError_t IPCEventEmulated::streamWait(hip::Stream* stream, uint flags) {
  std::scoped_lock lock(lock_);
  const int offset = ipc_evt_.ipc_shmem_->read_index;
  return ihipStreamOperation(
      reinterpret_cast<hipStream_t>(stream),
      ROCCLR_COMMAND_STREAM_WAIT_VALUE,
      &(ipc_evt_.ipc_shmem_->signal[offset]), 0, 1, 1,
      sizeof(uint32_t));
}

// ================================================================================================
hipError_t IPCEventEmulated::recordCommand(amd::Command*& command, amd::HostQueue* stream,
                                           uint32_t flags, bool batch_flush) {
  // Graph event-record nodes call this directly; lock_ is recursive so the
  // normal addMarker path is unaffected.
  std::scoped_lock lock(lock_);
  command = new amd::Marker(*stream, kMarkerDisableFlush);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEventEmulated::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  // Guard event_/shmem against concurrent query/synchronize/streamWait. Graph
  // event-record nodes call this directly; lock_ is recursive.
  std::scoped_lock lock(lock_);
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
hipError_t IPCEventEmulated::GetHandle(ihipIpcEventHandle_t* handle) {
  std::scoped_lock lock(lock_);
  if (!createIpcEventShmemIfNeeded()) {
    return hipErrorInvalidValue;
  }
  ipc_evt_.ipc_shmem_->owners_device_id = deviceId();
  ipc_evt_.ipc_shmem_->owners_process_id = amd::Os::getProcessId();
  handle->type = kIpcEventHandleEmulated;
  handle->creator_pid = static_cast<int32_t>(amd::Os::getProcessId());
  memset(handle->shmem_name, 0, IHIP_IPC_EVENT_HANDLE_SIZE);
  ipc_evt_.ipc_name_.copy(handle->shmem_name, std::string::npos);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEventEmulated::OpenHandle(ihipIpcEventHandle_t* handle) {
  std::scoped_lock lock(lock_);
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
// IPCEvent implementation (true IPC signals for supported backends)
// Record: standard barrier + async handler sets IPC signal to 0 when GPU work completes
// StreamWait: barrier packet with IPC signal as dep_signal (GPU waits until signal reaches 0)
// ================================================================================================

hipError_t IPCEvent::createIpcSignalIfNeeded() {
  if (ipc_signal_ != nullptr) {
    return hipSuccess;
  }

  auto* dev = g_devices[deviceId()]->devices()[0];
  ipc_signal_ = dev->createIpcSignal();
  if (ipc_signal_ == nullptr) {
    return hipErrorInvalidValue;
  }

  const auto ws = (flags_ & hipEventBlockingSync)
      ? amd::device::Signal::WaitState::Blocked
      : amd::device::Signal::WaitState::Active;
  if (!ipc_signal_->Init(*dev, 1, ws)) {
    delete ipc_signal_;
    ipc_signal_ = nullptr;
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

IPCEvent::~IPCEvent() {
  if (ipc_signal_ != nullptr) {
    // If the event was recorded (signal armed), wait for any in-flight barrier
    // to finish before destroying the signal; otherwise the GPU could write to
    // freed memory.  Skip the wait when the signal was never recorded (still at
    // its initial value) — waiting would hang forever.
    if (event_ != nullptr) {
      ipc_signal_->Wait(1, amd::device::Signal::Condition::Lt, UINT64_MAX);
    }
    delete ipc_signal_;
    ipc_signal_ = nullptr;
  }
}

// ================================================================================================
hipError_t IPCEvent::GetHandle(ihipIpcEventHandle_t* handle) {
  std::scoped_lock lock(lock_);
  auto status = createIpcSignalIfNeeded();
  if (status != hipSuccess) {
    return status;
  }

  handle->type = kIpcEventHandleROCr;
  handle->creator_pid = static_cast<int32_t>(amd::Os::getProcessId());
  if (!ipc_signal_->IpcExport(handle->ipc_signal_handle, IHIP_IPC_EVENT_HANDLE_SIZE)) {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::OpenHandle(ihipIpcEventHandle_t* handle) {
  std::scoped_lock lock(lock_);
  if (handle->type != kIpcEventHandleROCr) {
    return hipErrorInvalidValue;
  }

  if (static_cast<int32_t>(amd::Os::getProcessId()) == handle->creator_pid) {
    return hipErrorInvalidContext;
  }

  auto* dev = g_devices[deviceId()]->devices()[0];
  ipc_signal_ = dev->createIpcSignal();
  if (ipc_signal_ == nullptr) {
    return hipErrorInvalidValue;
  }

  if (!ipc_signal_->IpcImport(handle->ipc_signal_handle, IHIP_IPC_EVENT_HANDLE_SIZE)) {
    delete ipc_signal_;
    ipc_signal_ = nullptr;
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::recordCommand(amd::Command*& command, amd::HostQueue* stream,
                                   uint32_t flags, bool batch_flush) {
  // Protect ipc_signal_ creation against concurrent access. Not all callers hold
  // Event::lock() (e.g. graph event-record nodes call this directly from
  // CreateCommand); lock_ is recursive, so the normal addMarker path is safe.
  std::scoped_lock lock(lock_);

  auto status = createIpcSignalIfNeeded();
  if (status != hipSuccess) {
    return status;
  }

  auto* marker = new amd::Marker(*stream, kMarkerDisableFlush);
  marker->setIpcCompletionSignal(ipc_signal_);
  command = marker;
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  // Protect event_/ipc_signal_ against concurrent query/synchronize/streamWait.
  // Not all callers hold Event::lock() (e.g. graph event-record nodes enqueue
  // directly), so take it here; lock_ is recursive, so the normal addMarker path
  // that already holds it is unaffected.
  std::scoped_lock lock(lock_);

  // A single shared IPC signal cannot represent overlapping recordings, and the
  // consumer is attached to this exact signal (it cannot be rotated). So we must
  // serialize re-recordings: wait for the previous record's GPU work to drain
  // before re-arming, otherwise the absolute Reset(1) races with the prior
  // barrier's pending decrement and a waiter can wake on the wrong recording.
  // Skip the wait on the first record (signal still at its initial value, never
  // decremented) — waiting there would hang forever.
  if (event_ != nullptr) {
    ipc_signal_->Wait(1, amd::device::Signal::Condition::Lt, UINT64_MAX);
  }

  // Re-arm the signal; GPU barrier will decrement to 0 when work completes
  ipc_signal_->Reset(1);

  command->enqueue();

  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();

  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::synchronize() {
  std::scoped_lock lock(lock_);

  if (ipc_signal_ == nullptr) {
    return hipSuccess;
  }

  ipc_signal_->Wait(1, amd::device::Signal::Condition::Lt, UINT64_MAX);
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::query() {
  std::scoped_lock lock(lock_);

  if (ipc_signal_ == nullptr) {
    return hipSuccess;
  }

  if (ipc_signal_->Load() >= 1) {
    return hipErrorNotReady;
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t IPCEvent::streamWait(hip::Stream* stream, uint flags) {
  std::scoped_lock lock(lock_);

  if (ipc_signal_ == nullptr) {
    return hipSuccess;
  }

  if (ipc_signal_->Load() < 1) {
    return hipSuccess;
  }

  // Dispatch a barrier that waits on the IPC signal as dep_signal
  auto* marker = new amd::Marker(*stream, kMarkerDisableFlush);
  marker->setIpcDepSignal(ipc_signal_);
  marker->enqueue();
  return hipSuccess;
}

// ================================================================================================
// HIP API functions for IPC events
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

  auto* const iHandle = reinterpret_cast<ihipIpcEventHandle_t*>(&handle);

  // Select event implementation based on the handle's type field rather than
  // a runtime probe — the opener must match the exporter's implementation.
  auto status = ihipCreateIpcEventByType(event, iHandle->type);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }

  auto* const e = reinterpret_cast<hip::Event*>(*event);
  const auto open_status = e->OpenHandle(iHandle);
  if (open_status != hipSuccess) {
    ihipDestroyIpcEvent(*event);
    *event = nullptr;
  }
  HIP_RETURN(open_status);
}
}  // namespace hip
