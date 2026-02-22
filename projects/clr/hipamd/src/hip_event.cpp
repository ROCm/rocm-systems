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
#include "hip_graph_internal.hpp"

#if !defined(_MSC_VER)
#include <unistd.h>
#endif

namespace hip {

// Guards global event set
static std::shared_mutex eventSetLock{};
static std::unordered_set<hipEvent_t> eventSet;

// ================================================================================================
bool Event::ready() {
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  if (CheckHwEvent() || event_->status() == CL_COMPLETE) {
    return true;
  }

  event_->notifyCmdQueue();
  return false;
}

// ================================================================================================
bool EventDD::ready() {
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  return CheckHwEvent() || (event_->status() == CL_COMPLETE);
}

// ================================================================================================
hipError_t Event::query() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  return ready() ? hipSuccess : hipErrorNotReady;
}

// ================================================================================================
hipError_t Event::synchronize() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  constexpr bool kWaitCompletion = true;
  const amd::SyncPolicy policy =
      (flags_ == hipEventBlockingSync) ? amd::SyncPolicy::Blocking : amd::SyncPolicy::Auto;

  const amd::Device* device = g_devices[deviceId()]->devices()[0];
  if (!device->IsHwEventReady(*event_, kWaitCompletion, policy)) {
    event_->awaitCompletion();
  }
  return hipSuccess;
}

// ================================================================================================
bool Event::awaitEventCompletion() { return event_->awaitCompletion(); }

// ================================================================================================
bool EventDD::awaitEventCompletion() {
  constexpr bool kWaitCompletion = true;
  const amd::SyncPolicy policy =
      (flags_ == hipEventBlockingSync) ? amd::SyncPolicy::Blocking : amd::SyncPolicy::Auto;
  return g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, kWaitCompletion, policy);
}

// ================================================================================================
hipError_t Event::elapsedTime(Event& eStop, float& ms) {
  amd::ScopedLock startLock(lock_);

  // Handle same event case
  if (this == &eStop) {
    if (event_ == nullptr || (flags_ & hipEventDisableTiming)) {
      return hipErrorInvalidHandle;
    }
    ms = 0.f;
    return ready() ? hipSuccess : hipErrorNotReady;
  }

  amd::ScopedLock stopLock(eStop.lock());

  // Validate events
  if (event_ == nullptr || eStop.event() == nullptr) {
    return hipErrorInvalidHandle;
  }
  if ((flags_ | eStop.flags_) & hipEventDisableTiming) {
    return hipErrorInvalidHandle;
  }
  if (!ready() || !eStop.ready()) {
    return hipErrorNotReady;
  }

  constexpr float kNsToMs = 1.0f / 1000000.0f;

  if (event_ == eStop.event_) {
    // Events are the same, which indicates the stream is empty and likely
    // eventRecord is called on another stream. For such cases insert and measure a marker.
    auto* command = new amd::Marker(*event_->command().queue(), kMarkerDisableFlush);
    command->enqueue();
    command->awaitCompletion();
    ms = static_cast<float>(static_cast<int64_t>(command->event().profilingInfo().end_) -
                            time(false)) * kNsToMs;
    command->release();
  } else {
    // Note: with direct dispatch eStop.ready() relies on HW event, but CPU status can be delayed.
    // Hence for now make sure CPU status is updated by calling awaitCompletion();
    awaitEventCompletion();
    eStop.awaitEventCompletion();
    ms = static_cast<float>(eStop.time(false) - time(false)) * kNsToMs;
  }
  return hipSuccess;
}

// ================================================================================================
int64_t Event::time(bool getStartTs) const {
  assert(event_ != nullptr);
  if (getStartTs) {
    return static_cast<int64_t>(event_->profilingInfo().start_);
  } else {
    return static_cast<int64_t>(event_->profilingInfo().end_);
  }
}

// ================================================================================================
int64_t EventDD::time(bool getStartTs) const {
  assert(event_ != nullptr);
  uint64_t start = 0, end = 0;
  g_devices[deviceId()]->devices()[0]->getHwEventTime(*event_, &start, &end);

  // Select the requested timestamp and fallback to CPU profiling if not available
  const uint64_t timestamp = getStartTs ? start : end;
  if (timestamp == 0) {
    return Event::time(getStartTs);
  }
  return static_cast<int64_t>(timestamp);
}
// ================================================================================================
hipError_t Event::streamWaitCommand(amd::Command*& command, hip::Stream* stream) {
  const amd::Command::EventWaitList eventWaitList =
      (event_ != nullptr) ? amd::Command::EventWaitList{event_} : amd::Command::EventWaitList{};

  command = new amd::Marker(*stream, kMarkerDisableFlush, eventWaitList);
  // Since we only need to have a dependency on an existing event,
  // we may not need to flush any caches.
  command->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  return hipSuccess;
}
// ================================================================================================
hipError_t Event::streamWait(hip::Stream* stream, uint flags) {
  amd::ScopedLock lock(lock_);

  // Early return if event is not recorded, same stream, or already ready
  if ((event_ == nullptr) || (event_->command().queue() == stream) || ready()) {
    return hipSuccess;
  }

  if (!event_->notifyCmdQueue()) {
    return hipErrorLaunchOutOfResources;
  }

  amd::Command* command;
  if (const auto status = streamWaitCommand(command, stream); status != hipSuccess) {
    return status;
  }

  command->enqueue();
  command->release();
  return hipSuccess;
}

// ================================================================================================
hipError_t Event::recordCommand(amd::Command*& command, amd::HostQueue* stream, uint32_t ext_flags,
                                bool batch_flush) {
  if (command != nullptr) {
    return hipSuccess;
  }

  const auto flags = (ext_flags == 0) ? flags_ : ext_flags;
  
  const auto releaseFlags = [&]() {
    if (flags & hipEventDisableSystemFence) {
      return amd::Device::kCacheStateIgnore;
    }
    return amd::Device::kCacheStateInvalid;
  }();

  constexpr bool kMarkerTs = true;
  constexpr bool kFlushCache = false;
  command = new hip::EventMarker(*stream, kFlushCache, kMarkerTs, releaseFlags, batch_flush);
  return hipSuccess;
}

// ================================================================================================
hipError_t Event::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  command->enqueue();

  amd::Event& new_event = command->event();
  if (event_ == &new_event) {
    return hipSuccess;
  }

  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &new_event;

  return hipSuccess;
}

// ================================================================================================
hipError_t Event::addMarker(hip::Stream* hip_stream, amd::Command* command, bool batch_flush) {
  // Keep the lock always at the beginning of this to avoid a race. SWDEV-277847
  amd::ScopedLock lock(lock_);
  if (const auto status = recordCommand(command, hip_stream, 0, batch_flush);
      status != hipSuccess) {
    return status;
  }
  return enqueueRecordCommand(hip_stream, command);
}

// ================================================================================================
bool isValid(hipEvent_t event) {
  if (event == nullptr) {
    return true;
  }

  std::shared_lock lock(eventSetLock);
  return eventSet.find(event) != eventSet.end();
}

// ================================================================================================
hipError_t ihipEventCreateWithFlags(hipEvent_t* event, uint32_t flags) {
  // Define supported event flags
  constexpr uint32_t kSupportedFlags = hipEventDefault | hipEventBlockingSync |
                                       hipEventDisableTiming | hipEventReleaseToDevice |
                                       hipEventReleaseToSystem | hipEventInterprocess |
                                       hipEventDisableSystemFence;
  constexpr uint32_t kReleaseFlags = (hipEventReleaseToDevice | hipEventReleaseToSystem |
                                      hipEventDisableSystemFence);

  // Helper to count set bits for validating multiple release flags
  constexpr auto countBits = [](uint32_t num) {
    uint32_t count = 0;
    while (num) {
      num &= num - 1;
      ++count;
    }
    return count;
  };

  // Validate flags: no unsupported flags, max one release flag,
  // interprocess requires disable timing
  if ((flags & ~kSupportedFlags) || (countBits(flags & kReleaseFlags) > 1) ||
      ((flags & hipEventInterprocess) && !(flags & hipEventDisableTiming))) {
    return hipErrorInvalidValue;
  }

  // Create appropriate event type based on flags
  hip::Event* e = nullptr;
  if (flags & hipEventInterprocess) {
    e = new hip::IPCEvent(flags);
  } else if (AMD_DIRECT_DISPATCH) {
    e = new hip::EventDD(flags);
  } else {
    e = new hip::Event(flags);
  }
  *event = reinterpret_cast<hipEvent_t>(e);

  // Register event in global set
  std::unique_lock lock(hip::eventSetLock);
  hip::eventSet.insert(*event);

  return hipSuccess;
}

// ================================================================================================
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  HIP_INIT_API(hipEventCreateWithFlags, event, flags);

  if (event == nullptr) {
    return hipErrorInvalidValue;
  }

  HIP_RETURN(ihipEventCreateWithFlags(event, flags), *event);
}

// ================================================================================================
hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_INIT_API(hipEventCreate, event);

  if (event == nullptr) {
    return hipErrorInvalidValue;
  }

  HIP_RETURN(ihipEventCreateWithFlags(event, 0), *event);
}

// ================================================================================================
hipError_t hipEventDestroy(hipEvent_t event) {
  HIP_INIT_API(hipEventDestroy, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  std::unique_lock lock(hip::eventSetLock);
  if (hip::eventSet.erase(event) == 0) {
    return hipErrorContextIsDestroyed;
  }

  auto* e = reinterpret_cast<hip::Event*>(event);
  // Handle capture stream cleanup (stream might be destroyed first)
  hipStream_t stream = e->GetCaptureStream();
  if (hip::isValid(stream) && stream != nullptr && stream != hipStreamLegacy) {
    reinterpret_cast<hip::Stream*>(stream)->EraseCaptureEvent(event);
  }

  delete e;
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

  // Validate parameters
  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (start == nullptr || stop == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  auto* const eStart = reinterpret_cast<hip::Event*>(start);
  auto* const eStop = reinterpret_cast<hip::Event*>(stop);

  if (eStart->deviceId() != eStop->deviceId()) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  HIP_RETURN(eStart->elapsedTime(*eStop, *ms), "Elapsed Time = ", *ms);
}

// ================================================================================================
hipError_t hipEventRecord_common(hipEvent_t event, hipStream_t stream, uint32_t flags) {
  // Validate flags and event
  if (flags != hipEventRecordDefault && flags != hipEventRecordExternal) {
    return hipErrorInvalidValue;
  }
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  getStreamPerThread(stream);
  auto* const e = reinterpret_cast<hip::Event*>(event);
  auto* const hip_stream = hip::getStream(stream);
  if (hip_stream == nullptr) {
    return hipErrorInvalidValue;
  }

  // Clean up previous capture stream association
  hipStream_t lastCaptureStream = e->GetCaptureStream();
  if (hip::isValid(lastCaptureStream) && lastCaptureStream != nullptr &&
      lastCaptureStream != hipStreamLegacy) {
    reinterpret_cast<hip::Stream*>(lastCaptureStream)->EraseCaptureEvent(event);
  }
  e->SetCaptureStream(stream);

  // Handle stream capture mode
  if (stream != nullptr && stream != hipStreamLegacy &&
      hip_stream->GetCaptureStatus() == hipStreamCaptureStatusActive) {
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] Current capture node EventRecord on stream : %p, Event %p", stream, event);
    hip_stream->SetCaptureEvent(event);
    const auto& lastCapturedNodes = hip_stream->GetLastCapturedNodes();
    e->SetNodesPrevToRecorded(lastCapturedNodes);

    if (flags == hipEventRecordExternal) {
      auto* const node = new hip::GraphEventRecordNode(event);
      const auto status = hip::ihipGraphAddNode(node, hip_stream->GetCaptureGraph(),
                                                lastCapturedNodes.data(),
                                                lastCapturedNodes.size(), false);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "hipEventRecord add external event node failed");
        return status;
      }
      hip_stream->SetLastCapturedNode(node);
    }
    return hipSuccess;
  }

  // Normal event recording
  if (e->deviceId() != hip_stream->DeviceId()) {
    return hipErrorInvalidResourceHandle;
  }
  return e->addMarker(hip_stream, nullptr, !hip::Event::kBatchFlush);
}

// ================================================================================================
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(hipEventRecord, event, stream);
  HIP_RETURN(hipEventRecord_common(event, stream, hipEventRecordDefault));
}

// ================================================================================================
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(hipEventRecord, event, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipEventRecord_common(event, stream, hipEventRecordDefault));
}

// ================================================================================================
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, uint32_t flags) {
  HIP_INIT_API(hipEventRecordWithFlags, event, stream, flags);
  HIP_RETURN(hipEventRecord_common(event, stream, flags));
}

// ================================================================================================
hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(hipEventSynchronize, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  auto* e = reinterpret_cast<hip::Event*>(event);
  const auto hip_stream = e->GetCaptureStream();
  
  // Check for active capture
  if (hip_stream != nullptr && hip_stream != hipStreamLegacy) {
    auto* s = reinterpret_cast<hip::Stream*>(hip_stream);
    if (s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
      s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
      HIP_RETURN(hipErrorCapturedEvent);
    }
  }
  
  if (hip::Stream::StreamCaptureOngoing(hip_stream)) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  const auto status = e->synchronize();
  // Release freed memory for all memory pools on the device
  g_devices[e->deviceId()]->ReleaseFreedMemory();
  HIP_RETURN(status);
}

// ================================================================================================
hipError_t ihipEventQuery(hipEvent_t event) {
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  auto* e = reinterpret_cast<hip::Event*>(event);
  const auto hip_stream = e->GetCaptureStream();

  // Check for active capture
  if (hip_stream != nullptr && hip_stream != hipStreamLegacy) {
    auto* s = reinterpret_cast<hip::Stream*>(hip_stream);
    if (s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
      s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
      HIP_RETURN(hipErrorCapturedEvent);
    }
  }
  
  if (hip::Stream::StreamCaptureOngoing(hip_stream)) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }
  
  return e->query();
}

// ================================================================================================
hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(hipEventQuery, event);
  HIP_RETURN(ihipEventQuery(event));
}
}  // namespace hip
