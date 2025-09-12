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

bool Event::ready() {
  if (event_->status() != CL_COMPLETE) {
    event_->notifyCmdQueue();
  }
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  bool ready = CheckHwEvent();
  if (!ready) {
    ready = (event_->status() == CL_COMPLETE);
  }
  return ready;
}

bool EventDD::ready() {
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  bool ready = CheckHwEvent();
  // FIXME: Remove status check entirely
  if (!ready) {
    ready = (event_->status() == CL_COMPLETE);
  }
  return ready;
}

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

  auto hip_device = g_devices[deviceId()];
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  static constexpr bool kWaitCompletion = true;
  amd::SyncPolicy policy =
      (flags_ == hipEventBlockingSync) ? amd::SyncPolicy::Blocking : amd::SyncPolicy::Auto;
  if (!hip_device->devices()[0]->IsHwEventReady(*event_, kWaitCompletion, policy)) {
    event_->awaitCompletion();
  }
  return hipSuccess;
}

// ================================================================================================
bool Event::awaitEventCompletion() { return event_->awaitCompletion(); }

bool EventDD::awaitEventCompletion() {
  amd::SyncPolicy policy =
      (flags_ == hipEventBlockingSync) ? amd::SyncPolicy::Blocking : amd::SyncPolicy::Auto;
  return g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, true, policy);
}

hipError_t Event::elapsedTime(Event& eStop, float& ms) {
  amd::ScopedLock startLock(lock_);
  if (this == &eStop) {
    ms = 0.f;
    if (event_ == nullptr) {
      return hipErrorInvalidHandle;
    }

    if (flags_ & hipEventDisableTiming) {
      return hipErrorInvalidHandle;
    }

    if (!ready()) {
      return hipErrorNotReady;
    }

    return hipSuccess;
  }
  amd::ScopedLock stopLock(eStop.lock());

  if (event_ == nullptr || eStop.event() == nullptr) {
    return hipErrorInvalidHandle;
  }

  if ((flags_ | eStop.flags_) & hipEventDisableTiming) {
    return hipErrorInvalidHandle;
  }

  if (!ready() || !eStop.ready()) {
    return hipErrorNotReady;
  }

  if (event_ == eStop.event_) {
    // Events are the same, which indicates the stream is empty and likely
    // eventRecord is called on another stream. For such cases insert and measure a
    // marker.
    amd::Command* command = new amd::Marker(*event_->command().queue(), kMarkerDisableFlush);
    command->enqueue();
    command->awaitCompletion();
    ms = static_cast<float>(static_cast<int64_t>(command->event().profilingInfo().end_) -
                            time(false)) /
         1000000.f;
    command->release();
  } else {
    // Note: with direct dispatch eStop.ready() relies on HW event, but CPU status can be delayed.
    // Hence for now make sure CPU status is updated by calling awaitCompletion();
    awaitEventCompletion();
    eStop.awaitEventCompletion();
    ms = static_cast<float>(eStop.time(false) - time(false)) / 1000000.f;
  }
  return hipSuccess;
}

int64_t Event::time(bool getStartTs) const {
  assert(event_ != nullptr);
  if (getStartTs) {
    return static_cast<int64_t>(event_->profilingInfo().start_);
  } else {
    return static_cast<int64_t>(event_->profilingInfo().end_);
  }
}

int64_t EventDD::time(bool getStartTs) const {
  uint64_t start = 0, end = 0;
  assert(event_ != nullptr);
  g_devices[deviceId()]->devices()[0]->getHwEventTime(*event_, &start, &end);
  // FIXME: This is only needed if the command had to wait CL_COMPLETE status
  if (start == 0 || end == 0) {
    return Event::time(getStartTs);
  }
  if (getStartTs) {
    return static_cast<int64_t>(start);
  } else {
    return static_cast<int64_t>(end);
  }
}
// ================================================================================================
hipError_t Event::streamWaitCommand(amd::Command*& command, hip::Stream* stream) {
  amd::Command::EventWaitList eventWaitList;
  if (event_ != nullptr) {
    eventWaitList.push_back(event_);
  }
  command = new amd::Marker(*stream, kMarkerDisableFlush, eventWaitList);
  // Since we only need to have a dependency on an existing event,
  // we may not need to flush any caches.
  command->setCommandEntryScope(amd::Device::kCacheStateIgnore);

  if (command == NULL) {
    return hipErrorOutOfMemory;
  }
  return hipSuccess;
}
// ================================================================================================
hipError_t Event::streamWait(hip::Stream* stream, uint flags) {
  // Access to event_ object must be lock protected
  amd::ScopedLock lock(lock_);
  if ((event_ == nullptr) || (event_->command().queue() == stream) || ready()) {
    return hipSuccess;
  }
  if (!event_->notifyCmdQueue()) {
    return hipErrorLaunchOutOfResources;
  }
  amd::Command* command;
  hipError_t status = streamWaitCommand(command, stream);
  if (status != hipSuccess) {
    return status;
  }
  command->enqueue();
  command->release();
  return hipSuccess;
}

// ================================================================================================
hipError_t Event::recordCommand(amd::Command*& command, amd::HostQueue* stream, uint32_t ext_flags,
                                bool batch_flush) {
  if (command == nullptr) {
    int32_t releaseFlags =
        ((ext_flags == 0) ? flags_ : ext_flags) &
        (hipEventReleaseToDevice | hipEventReleaseToSystem | hipEventDisableSystemFence);
    if (releaseFlags & hipEventDisableSystemFence) {
      releaseFlags = amd::Device::kCacheStateIgnore;
    } else {
      releaseFlags = amd::Device::kCacheStateInvalid;
    }
    // Always submit a EventMarker.
    constexpr bool kMarkerTs = true;
    command =
        new hip::EventMarker(*stream, !kMarkerDisableFlush, kMarkerTs, releaseFlags, batch_flush);
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t Event::enqueueRecordCommand(hip::Stream* stream, amd::Command* command) {
  command->enqueue();
  if (event_ == &command->event()) {
    return hipSuccess;
  }
  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();

  return hipSuccess;
}

// ================================================================================================
hipError_t Event::addMarker(hip::Stream* hip_stream, amd::Command* command, bool batch_flush) {
  // Keep the lock always at the beginning of this to avoid a race. SWDEV-277847
  amd::ScopedLock lock(lock_);
  hipError_t status = recordCommand(command, hip_stream, 0, batch_flush);
  if (status != hipSuccess) {
    return hipSuccess;
  }
  status = enqueueRecordCommand(hip_stream, command);
  return status;
}

// ================================================================================================
bool isValid(hipEvent_t event) {
  // NULL event is always valid
  if (event == nullptr) {
    return true;
  }

  std::shared_lock lock(eventSetLock);
  if (eventSet.find(event) == eventSet.end()) {
    return false;
  }

  return true;
}

// ================================================================================================
hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
                            hipEventReleaseToDevice | hipEventReleaseToSystem |
                            hipEventInterprocess | hipEventDisableSystemFence;

  const unsigned releaseFlags =
      (hipEventReleaseToDevice | hipEventReleaseToSystem | hipEventDisableSystemFence);
  // can't set any unsupported flags.
  // can set only one of the release flags.
  // if hipEventInterprocess flag is set, then hipEventDisableTiming flag also must be set
  const bool illegalFlags = (flags & ~supportedFlags) || ([](unsigned int num) {
                              unsigned int bitcount;
                              for (bitcount = 0; num; bitcount++) {
                                num &= num - 1;
                              }
                              return bitcount;
                            }(flags & releaseFlags) > 1) ||
                            ((flags & hipEventInterprocess) && !(flags & hipEventDisableTiming));
  if (!illegalFlags) {
    hip::Event* e = nullptr;
    if (flags & hipEventInterprocess) {
      e = new hip::IPCEvent();
    } else {
      if (AMD_DIRECT_DISPATCH) {
        e = new hip::EventDD(flags);
      } else {
        e = new hip::Event(flags);
      }
    }
    // App might have used combination of flags i.e. hipEventInterprocess|hipEventDisableTiming
    // However based on hipEventInterprocess flag, IPCEvent creates even with
    // JUST hipEventInterprocess and hence, Actual hipEventInterprocess|hipEventDisableTiming
    // flag is getting supressed with hipEventInterprocess
    e->flags_ = flags;
    if (e == nullptr) {
      return hipErrorOutOfMemory;
    }
    *event = reinterpret_cast<hipEvent_t>(e);
    std::unique_lock lock(hip::eventSetLock);
    hip::eventSet.insert(*event);
  } else {
    return hipErrorInvalidValue;
  }
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

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  // There is a possibility that stream destroy be called first
  hipStream_t s = e->GetCaptureStream();
  if (hip::isValid(s)) {
    if (s != nullptr && s != hipStreamLegacy) {
      reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->EraseCaptureEvent(event);
    }
  }
  delete e;
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (start == nullptr || stop == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* eStart = reinterpret_cast<hip::Event*>(start);
  hip::Event* eStop = reinterpret_cast<hip::Event*>(stop);

  if (eStart->deviceId() != eStop->deviceId()) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  HIP_RETURN(eStart->elapsedTime(*eStop, *ms), "Elapsed Time = ", *ms);
}

// ================================================================================================
hipError_t hipEventRecord_common(hipEvent_t event, hipStream_t stream, unsigned int flags) {
  if (!(flags == hipEventRecordDefault || flags == hipEventRecordExternal)) {
    return hipErrorInvalidValue;
  }
  hipError_t status = hipSuccess;
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }
  getStreamPerThread(stream);
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::Stream* hip_stream = hip::getStream(stream);
  e->SetCaptureStream(stream);
  if ((stream != nullptr && stream != hipStreamLegacy) &&
      (s->GetCaptureStatus() == hipStreamCaptureStatusActive)) {
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] Current capture node EventRecord on stream : %p, Event %p", stream, event);
    s->SetCaptureEvent(event);
    std::vector<hip::GraphNode*> lastCapturedNodes = s->GetLastCapturedNodes();
    e->SetNodesPrevToRecorded(lastCapturedNodes);
    if (flags == hipEventRecordExternal) {
      hip::GraphNode* node = new hip::GraphEventRecordNode(reinterpret_cast<hipEvent_t>(e));
      hipError_t status = hip::ihipGraphAddNode(
          node, reinterpret_cast<hip::Graph*>(s->GetCaptureGraph()),
          reinterpret_cast<hip::GraphNode* const*>(s->GetLastCapturedNodes().data()),
          s->GetLastCapturedNodes().size(), false);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "hipEventRecord add external event node failed");
        return status;
      }
      s->SetLastCapturedNode(node);
    }
  } else {
    if (e->deviceId() != hip_stream->DeviceId()) {
      return hipErrorInvalidResourceHandle;
    }
    status = e->addMarker(hip_stream, nullptr, !hip::Event::kBatchFlush);
  }
  return status;
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
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned int flags) {
  HIP_INIT_API(hipEventRecordWithFlags, event, stream, flags);
  HIP_RETURN(hipEventRecord_common(event, stream, flags));
}

// ================================================================================================
hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(hipEventSynchronize, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  auto hip_stream = e->GetCaptureStream();
  hip::Stream* s = reinterpret_cast<hip::Stream*>(hip_stream);
  if ((hip_stream != nullptr && hip_stream != hipStreamLegacy) &&
      (s->GetCaptureStatus() == hipStreamCaptureStatusActive)) {
    s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
    HIP_RETURN(hipErrorCapturedEvent);
  }
  if (hip::Stream::StreamCaptureOngoing(hip_stream) == true) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  hipError_t status = e->synchronize();
  // Release freed memory for all memory pools on the device
  g_devices[e->deviceId()]->ReleaseFreedMemory();
  HIP_RETURN(status);
}

// ================================================================================================
hipError_t ihipEventQuery(hipEvent_t event) {
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  auto hip_stream = e->GetCaptureStream();
  hip::Stream* s = reinterpret_cast<hip::Stream*>(hip_stream);
  if ((hip_stream != nullptr && hip_stream != hipStreamLegacy) &&
      (s->GetCaptureStatus() == hipStreamCaptureStatusActive)) {
    s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
    HIP_RETURN(hipErrorCapturedEvent);
  }
  if (hip::Stream::StreamCaptureOngoing(e->GetCaptureStream())) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }
  return e->query();
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(hipEventQuery, event);
  HIP_RETURN(ihipEventQuery(event));
}
}  // namespace hip
