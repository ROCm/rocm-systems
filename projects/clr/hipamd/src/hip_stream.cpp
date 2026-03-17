/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_graph_internal.hpp"
#include "hip_event.hpp"
#include "thread/monitor.hpp"
#include "hip_prof_api.h"
#include <atomic>

namespace hip {

// ================================================================================================
Stream::Stream(hip::Device* dev, Priority p, unsigned int f, bool null_stream,
               const std::vector<uint32_t>& cuMask, hipStreamCaptureStatus captureStatus)
    : amd::HostQueue(*dev->asContext(), *dev->devices()[0], 0, amd::CommandQueue::RealTimeDisabled,
                     convertToQueuePriority(p), cuMask, null_stream),
      device_(dev),
      priority_(p),
      flags_(f),
      null_(null_stream),
      cuMask_(cuMask),
      stream_id_(GenerateStreamId()),
      captureStatus_(captureStatus) {
  device_->AddStream(this);
}

// ================================================================================================
hipError_t Stream::EndCapture() {
  // Detach all captured events from this stream.
  {
    std::scoped_lock lock(lock_);
    for (auto event : captureEvents_) {
      reinterpret_cast<hip::Event*>(event)->SetCaptureStream(nullptr);
    }
    captureEvents_.clear();
  }
  // Recursively end capture on all parallel (forked) streams.
  for (auto stream : parallelCaptureStreams_) {
    [[maybe_unused]] const auto err = reinterpret_cast<hip::Stream*>(stream)->EndCapture();
    assert(err == hipSuccess);
  }

  // Reset all capture state to defaults.
  captureStatus_ = hipStreamCaptureStatusNone;
  pCaptureGraph_ = nullptr;
  originStream_ = false;
  parentStream_ = nullptr;
  lastCapturedNodes_.clear();
  parallelCaptureStreams_.clear();

  return hipSuccess;
}

// ================================================================================================
bool Stream::Create() { return create(); }

// ================================================================================================
void Stream::Destroy(hip::Stream* stream, bool forceDestroy) {
  stream->device().removeFromActiveQueues(stream);
  stream->GetDevice()->RemoveStream(stream);
  stream->SetForceDestroy(forceDestroy);
  stream->release();
}

// ================================================================================================
bool Stream::terminate() {
  HostQueue::terminate();
  return true;
}

// ================================================================================================
bool isValid(hipStream_t& stream) {
  if (stream == nullptr || stream == hipStreamLegacy) {
    return true;
  }

  if (hipStreamPerThread == stream) {
    getStreamPerThread(stream);
  }

  const auto* s = reinterpret_cast<hip::Stream*>(stream);
  return std::any_of(g_devices.begin(), g_devices.end(),
                     [s](const auto& device) { return device->StreamExists(s); });
}

// ================================================================================================
void Stream::ReleaseCaptureGraph() {
  delete pCaptureGraph_;
  pCaptureGraph_ = nullptr;
}

// ================================================================================================
void Stream::AddCrossCapturedNode(const std::vector<hip::GraphNode*>& graphNodes, bool replace) {
  // Replace dependencies as per flag hipStreamSetCaptureDependencies.
  if (replace) {
    removedDependencies_.insert(removedDependencies_.end(),
                                lastCapturedNodes_.begin(), lastCapturedNodes_.end());
    lastCapturedNodes_.clear();
  }
  for (auto* node : graphNodes) {
    if (std::find(lastCapturedNodes_.begin(), lastCapturedNodes_.end(), node) ==
        lastCapturedNodes_.end()) {
      lastCapturedNodes_.push_back(node);
    }
  }
}

// ================================================================================================
int Stream::DeviceId() const { return device_->deviceId(); }

// ================================================================================================
int Stream::DeviceId(hipStream_t hStream) {
  assert(hip::isValid(hStream) && "Stream must be valid to get deviceId");

  // Legacy or null stream case
  if (hStream == nullptr || hStream == hipStreamLegacy) {
    return ihipGetDevice();
  }
  const int deviceId = reinterpret_cast<hip::Stream*>(hStream)->DeviceId();
  assert(deviceId >= 0 && deviceId < static_cast<int>(g_devices.size()) && "Invalid deviceId has been returned");
  return deviceId;
}

// ================================================================================================
bool Stream::StreamCaptureBlocking() {
  return std::any_of(g_devices.begin(), g_devices.end(),
                     [](const auto& device) { return device->StreamCaptureBlocking(); });
}

// ================================================================================================
bool Stream::StreamCaptureOngoing(hipStream_t hStream) {
  if (hStream == nullptr || hStream == hipStreamLegacy) {
    return false;
  }

  auto* s = reinterpret_cast<hip::Stream*>(hStream);
  const auto captureStatus = s->GetCaptureStatus();

  if (captureStatus == hipStreamCaptureStatusActive) {
    s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
    return true;
  }
  if (captureStatus == hipStreamCaptureStatusInvalidated) {
    return true;
  }
  if (captureStatus != hipStreamCaptureStatusNone) {
    // Defensive: unknown future enum value — treat as not ongoing.
    return false;
  }

  // Relaxed mode — no cross-stream interference.
  if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeRelaxed) {
    return false;
  }
  // Global mode — invalidate all capturing streams in any thread.
  if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeGlobal) {
    amd::ScopedLock lock(g_captureStreamsLock);
    if (!g_captureStreams.empty()) {
      for (auto stream : hip::g_captureStreams) {
        stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
      }
      return true;
    }
  }
  // ThreadLocal mode — invalidate all capturing streams in current thread.
  if (!hip::tls.capture_streams_.empty()) {
    for (auto stream : hip::tls.capture_streams_) {
      stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
    }
    return true;
  }
  return false;
}

// ================================================================================================
void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data) {
  auto* cbo = reinterpret_cast<StreamCallback*>(user_data);
  cbo->callback();
  delete cbo;
}

// ================================================================================================
static hipError_t ihipStreamCreate(hipStream_t* stream, unsigned int flags,
                                   hip::Stream::Priority priority,
                                   const std::vector<uint32_t>& cuMask = {}) {
  if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
    return hipErrorInvalidValue;
  }
  auto* hStream = new hip::Stream(hip::getCurrentDevice(), priority, flags, false, cuMask);
  if (!hStream->Create()) {
    hip::Stream::Destroy(hStream);
    return hipErrorOutOfMemory;
  }

  *stream = reinterpret_cast<hipStream_t>(hStream);
  return hipSuccess;
}

// ================================================================================================
StreamPerThread::StreamPerThread() : streams_(g_devices.size()) {}

// ================================================================================================
StreamPerThread::~StreamPerThread() {
  for (auto& stream : streams_) {
    if (stream != nullptr && hip::isValid(stream)) {
      // @note: Global variables in hip runtime will be destroyed after ROCR's global variables.
      // Any calls to rocr may cause invalid object access. Hence, avoid the stream destruction.
      if (IS_LINUX || (GPU_ENABLE_PAL != 0)) {
        hip::Stream::Destroy(reinterpret_cast<hip::Stream*>(stream));
      }
      stream = nullptr;
    }
  }
}

// ================================================================================================
hipStream_t StreamPerThread::Get() {
  const int currDev = hip::getCurrentDevice()->deviceId();
  // Defensive re-init: streams_ may have been cleared by hipDeviceReset.
  if (streams_.empty()) {
    streams_.resize(g_devices.size());
  }
  // hipDeviceReset may also destroy individual entries, so revalidate.
  if (streams_[currDev] == nullptr || !hip::isValid(streams_[currDev])) {
    hipError_t status =
        ihipStreamCreate(&streams_[currDev], hipStreamDefault, hip::Stream::Priority::Normal);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE, "Stream creation failed");
    }
  }
  return streams_[currDev];
}

// ================================================================================================
void StreamPerThread::Clear() {
  if (!streams_.empty()) {
    streams_[getCurrentDevice()->deviceId()] = nullptr;
  }
}

// ================================================================================================
void getStreamPerThread(hipStream_t& stream) {
  if (stream == hipStreamPerThread) {
    stream = hip::tls.stream_per_thread_obj_.Get();
  }
}

// ================================================================================================
hipStream_t getPerThreadDefaultStream() {
  // Function to get per thread default stream
  // More about the usecases yet to come
  hipStream_t stream = hipStreamPerThread;
  getStreamPerThread(stream);
  return stream;
}

// ================================================================================================
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, flags, hip::Stream::Priority::Normal), *stream);
}

// ================================================================================================
hipError_t hipStreamCreate(hipStream_t* stream) {
  HIP_INIT_API(hipStreamCreate, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, hip::Stream::Priority::Normal), *stream);
}

// ================================================================================================
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  HIP_INIT_API(hipStreamCreateWithPriority, stream, flags, priority);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto streamPriority = static_cast<hip::Stream::Priority>(
      std::clamp(priority, static_cast<int>(hip::Stream::Priority::High),
                           static_cast<int>(hip::Stream::Priority::Low)));

  HIP_RETURN(ihipStreamCreate(stream, flags, streamPriority), *stream);
}

// ================================================================================================
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  HIP_INIT_API(hipDeviceGetStreamPriorityRange, leastPriority, greatestPriority);

  if (leastPriority != nullptr) {
    *leastPriority = hip::Stream::Priority::Low;
  }
  if (greatestPriority != nullptr) {
    *greatestPriority = hip::Stream::Priority::High;
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamGetFlags_common(hipStream_t stream, unsigned int* flags) {
  if (flags == nullptr || stream == nullptr) {
    return hipErrorInvalidValue;
  }
  getStreamPerThread(stream);
  *flags = reinterpret_cast<hip::Stream*>(stream)->Flags();
  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  HIP_INIT_API(hipStreamGetFlags, stream, flags);
  HIP_RETURN(hipStreamGetFlags_common(stream, flags));
}

// ================================================================================================
hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags) {
  HIP_INIT_API(hipStreamGetFlags, stream, flags);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamGetFlags_common(stream, flags));
}

// ================================================================================================
hipError_t hipStreamGetId_common(hipStream_t stream, unsigned long long* streamId) {
  if (streamId == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  getStreamPerThread(stream);
  constexpr bool wait = false;
  *streamId = hip::getStream(stream, wait)->GetStreamId();
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* streamId) {
  HIP_INIT_API(hipStreamGetId, stream, streamId);
  HIP_RETURN(hipStreamGetId_common(stream, streamId));
}

// ================================================================================================
hipError_t hipStreamSynchronize_common(hipStream_t stream) {
  getStreamPerThread(stream);

  if (stream == nullptr) {
    // Sync blocking streams only on the null stream
    constexpr bool kBlockingOnly = true;
    getCurrentDevice()->SyncAllStreams(false, kBlockingOnly);
    return hipSuccess;
  }

  if (stream != hipStreamLegacy && hip::Stream::StreamCaptureOngoing(stream)) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  constexpr bool wait = false;
  auto hip_stream = hip::getStream(stream, wait);
  hip_stream->finish();
  hip_stream->GetDevice()->ReleaseFreedMemory();
  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

// ================================================================================================
hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

// ================================================================================================
hipError_t hipStreamDestroy(hipStream_t stream) {
  HIP_INIT_API(hipStreamDestroy, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  if (stream == hipStreamPerThread || stream == hipStreamLegacy) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  if (s->GetCaptureStatus() != hipStreamCaptureStatusNone) {
    if (s->GetParentStream() != nullptr) {
      reinterpret_cast<hip::Stream*>(s->GetParentStream())->EraseParallelCaptureStream(stream);
    }
    s->EndCapture();
  }
  s->GetDevice()->RemoveStreamFromPools(s);

  // Remove the stream from all capture-tracking lists.
  auto erase_from = [s](auto& container) {
    auto it = std::find(container.begin(), container.end(), s);
    if (it != container.end()) container.erase(it);
  };
  {
    amd::ScopedLock lock(g_captureStreamsLock);
    erase_from(g_captureStreams);
  }
  {
    amd::ScopedLock lock(g_streamSetLock);
    erase_from(g_allCapturingStreams);
  }
  erase_from(hip::tls.capture_streams_);

  hip::Stream::Destroy(s);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamWaitEvent_common(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  if (flags != hipEventWaitDefault && flags != hipEventWaitExternal) {
    return hipErrorInvalidValue;
  }
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }
  getStreamPerThread(stream);
  hip::Stream* waitStream = hip::getStream(stream);
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  auto eventStreamHandle = reinterpret_cast<hipStream_t>(e->GetCaptureStream());
  // The stream associated with the device might have been destroyed.
  // If so, the event has already completed and we can resume the current stream.
  if (!hip::isValid(eventStreamHandle)) {
    return hipSuccess;
  }

  hip::Stream* eventStream = reinterpret_cast<hip::Stream*>(eventStreamHandle);

  // External-event wait: add a graph node and return immediately.
  if (flags == hipEventWaitExternal) {
    // Ensure the wait stream is actively capturing and has a capture graph.
    if (waitStream == nullptr || waitStream->GetCaptureGraph() == nullptr) {
      return hipErrorInvalidHandle;
    }
    auto lastCapturedNodes = waitStream->GetLastCapturedNodes();
    hip::GraphNode* pGraphNode = waitStream->GetCaptureGraph()->AddExternalEventWaitNode(
                                      reinterpret_cast<hip::GraphNode*>(lastCapturedNodes.data()),
                                      lastCapturedNodes.size(),
                                      event);
    waitStream->SetLastCapturedNode(pGraphNode);
    return hipSuccess;
  }

  // Event is being captured on eventStream: wire up the capture graph.
  if (eventStream != nullptr && eventStream->IsEventCaptured(event)) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
            "[hipGraph] Current capture node StreamWaitEvent on stream : %p, Event %p", stream,
            event);
    if (waitStream == nullptr) {
      return hipErrorInvalidHandle;
    }
    // Don't set when forked stream joins back to the parent.
    if (!waitStream->IsOriginStream() &&
        waitStream != reinterpret_cast<hip::Stream*>(eventStream->GetParentStream())) {
      waitStream->SetCaptureGraph(eventStream->GetCaptureGraph());
      waitStream->SetCaptureID(eventStream->GetCaptureID());
      waitStream->SetCaptureMode(eventStream->GetCaptureMode());
      waitStream->SetParentStream(reinterpret_cast<hipStream_t>(eventStream));
      eventStream->SetParallelCaptureStream(stream);
    }
    waitStream->AddCrossCapturedNode(e->GetNodesPrevToRecorded());
    return hipSuccess;
  }

  // Non-capture path: validate isolation, register safe-stream, and wait.
  if (eventStream != nullptr) {
    if (eventStream->GetCaptureStatus() == hipStreamCaptureStatusActive) {
      // Stream is capturing but event is not recorded on event's stream.
      return hipErrorStreamCaptureIsolation;
    }
    if (waitStream != nullptr && stream != hipStreamLegacy &&
        eventStream->DeviceId() == waitStream->DeviceId()) {
      eventStream->GetDevice()->AddSafeStream(eventStream, waitStream);
    }
  }
  return e->streamWait(waitStream, flags);
}

// ================================================================================================
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);
  HIP_RETURN(hipStreamWaitEvent_common(stream, event, flags));
}

// ================================================================================================
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamWaitEvent_common(stream, event, flags));
}

// ================================================================================================
hipError_t hipStreamQuery_common(hipStream_t stream) {
  getStreamPerThread(stream);

  // If still capturing, return error.
  if (stream != nullptr && hip::Stream::StreamCaptureOngoing(stream)) {
    return hipErrorStreamCaptureUnsupported;
  }

  const bool wait = (stream == nullptr);
  hip::Stream* hip_stream = hip::getStream(stream, wait);

  if (hip_stream->vdev()->isFenceDirty()) {
    amd::Command* command = new amd::Marker(*hip_stream, kMarkerDisableFlush);
    command->enqueue();
    command->release();
  }

  amd::Command* command = hip_stream->getLastQueuedCommand(true);
  if (command == nullptr) {
    // Nothing was submitted to the queue.
    return hipSuccess;
  }

  amd::Event& event = command->event();
  if (command->type() != 0) {
    event.notifyCmdQueue();
  }

  // Check HW status of the ROCclr event. Note: not all ROCclr modes support HW status.
  bool ready = command->queue()->device().IsHwEventReady(event);
  if (!ready) {
    ready = (command->status() == CL_COMPLETE);
  }
  command->release();

  if (!ready) {
    return hipErrorNotReady;
  }

  // Stream is complete — opportunistically release its HW queue if idle.
  hip_stream->vdev()->ReleaseHwQueue();
  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamQuery(hipStream_t stream) {
  HIP_INIT_API(hipStreamQuery, stream);
  HIP_RETURN(hipStreamQuery_common(stream));
}

// ================================================================================================
hipError_t hipStreamQuery_spt(hipStream_t stream) {
  HIP_INIT_API(hipStreamQuery, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamQuery_common(stream));
}

// ================================================================================================
hipError_t streamCallback_common(hipStream_t stream, StreamCallback* cbo, void* userData) {
  if (cbo == nullptr) {
    return hipErrorInvalidHandle;
  }

  getStreamPerThread(stream);
  hip::Stream* hip_stream = hip::getStream(stream);
  amd::Command* last_command = hip_stream->getLastQueuedCommand(true);

  amd::Command::EventWaitList eventWaitList;
  if (last_command != nullptr) {
    eventWaitList.push_back(last_command);
  }

  // Callback marker — released after the HIP callback fires, not here.
  amd::Command* command = new amd::Marker(*hip_stream, !kMarkerDisableFlush, eventWaitList);
  if (!command->setCallback(CL_COMPLETE, ihipStreamCallback, cbo)) {
    command->release();
    if (last_command != nullptr) {
      last_command->release();
    }
    return hipErrorInvalidHandle;
  }
  command->enqueue();
  if (last_command != nullptr) {
    last_command->release();
  }

  // Blocking marker — stalls the stream until the callback completes.
  // Required because HW event checks may occur before the callback finishes.
  eventWaitList = {command};
  amd::Command* block_command = new amd::Marker(*hip_stream, !kMarkerDisableFlush, eventWaitList);
  block_command->enqueue();

  command->release();
  block_command->notifyCmdQueue();
  block_command->release();

  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamAddCallback_common(hipStream_t stream, hipStreamCallback_t callback,
                                       void* userData, unsigned int flags) {
  // flags is reserved for future use and must be 0.
  if (callback == nullptr || flags != 0) {
    return hipErrorInvalidValue;
  }

  if (stream != nullptr && stream != hipStreamLegacy && hip::isValid(stream)) {
    hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
    if (s->GetCaptureStatus() != hipStreamCaptureStatusNone) {
      s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
      return hipErrorStreamCaptureUnsupported;
    }
  } else if (Stream::StreamCaptureBlocking()) {
    // A blocking stream is capturing — return error and invalidate all capturing streams.
    CHECK_STREAM_CAPTURING();
  }

  auto* cbo = new StreamAddCallback(stream, callback, userData);
  return streamCallback_common(stream, cbo, userData);
}

// ================================================================================================
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
  HIP_INIT_API(hipStreamAddCallback, stream, callback, userData, flags);
  HIP_RETURN(hipStreamAddCallback_common(stream, callback, userData, flags));
}

// ================================================================================================
hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback,
                                    void* userData, unsigned int flags) {
  HIP_INIT_API(hipStreamAddCallback, stream, callback, userData, flags);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamAddCallback_common(stream, callback, userData, flags));
}

// ================================================================================================
hipError_t hipLaunchHostFunc_common(hipStream_t stream, hipHostFn_t fn, void* userData) {
  STREAM_CAPTURE(hipLaunchHostFunc, stream, fn, userData);
  if (fn == nullptr) {
    return hipErrorInvalidValue;
  }
  auto* cbo = new LaunchHostFuncCallback(fn, userData);
  return streamCallback_common(stream, cbo, userData);
}

// ================================================================================================
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData) {
  HIP_INIT_API(hipLaunchHostFunc, stream, fn, userData);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipLaunchHostFunc_common(stream, fn, userData));
}

// ================================================================================================
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
  HIP_INIT_API(hipLaunchHostFunc, stream, fn, userData);
  if (stream == nullptr && hip::Stream::StreamCaptureOngoing(stream)) {
    HIP_RETURN(hipErrorStreamCaptureImplicit);
  }
  HIP_RETURN(hipLaunchHostFunc_common(stream, fn, userData));
}

// ================================================================================================
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask) {
  HIP_INIT_API(hipExtStreamCreateWithCUMask, stream, cuMaskSize, cuMask);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  if (cuMaskSize == 0 || cuMask == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const std::vector<uint32_t> cuMaskv(cuMask, cuMask + cuMaskSize);

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, hip::Stream::Priority::Normal, cuMaskv),
             *stream);
}

// ================================================================================================
hipError_t hipStreamGetPriority_common(hipStream_t stream, int* priority) {
  if (priority == nullptr) {
    return hipErrorInvalidValue;
  }

  if (stream == nullptr) {
    *priority = 0;
    return hipSuccess;
  }

  getStreamPerThread(stream);
  *priority = static_cast<int>(reinterpret_cast<hip::Stream*>(stream)->GetPriority());
  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  HIP_INIT_API(hipStreamGetPriority, stream, priority);
  HIP_RETURN(hipStreamGetPriority_common(stream, priority));
}

// ================================================================================================
hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority) {
  HIP_INIT_API(hipStreamGetPriority, stream, priority);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamGetPriority_common(stream, priority));
}

// ================================================================================================
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
  HIP_INIT_API(hipExtStreamGetCUMask, stream, cuMaskSize, cuMask);

  if (cuMask == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const int deviceId = hip::getCurrentDevice()->deviceId();
  auto* deviceHandle = g_devices[deviceId]->devices()[0];
  const auto& info = deviceHandle->info();

  // Minimum number of uint32_t words needed to represent all CU bits.
  const uint32_t cuMaskSizeRequired = (info.maxComputeUnits_ + 31) / 32;
  if (cuMaskSize < cuMaskSizeRequired) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Build a default CU mask with all CUs active (one bit per CU).
  const uint32_t fullWords = info.maxComputeUnits_ / 32;
  const uint32_t remainingBits = info.maxComputeUnits_ % 32;
  std::vector<uint32_t> defaultCUMask(fullWords, 0xFFFFFFFF);
  if (remainingBits > 0) {
    defaultCUMask.push_back((1u << remainingBits) - 1);
  }

  // Null/per-thread stream: return globalCUMask if defined, otherwise the default.
  if (stream == nullptr || stream == hipStreamPerThread) {
    const auto& src = !info.globalCUMask_.empty() ? info.globalCUMask_ : defaultCUMask;
    std::copy(src.begin(), src.end(), cuMask);
    HIP_RETURN(hipSuccess);
  }

  // Non-null stream: AND the stream's CU mask with the base mask (global or default).
  const auto& baseMask = !info.globalCUMask_.empty() ? info.globalCUMask_ : defaultCUMask;
  auto streamCUMask = reinterpret_cast<hip::Stream*>(stream)->GetCUMask();

  std::vector<uint32_t> mask;
  mask.reserve(std::min(streamCUMask.size(), baseMask.size()));
  for (uint32_t i = 0; i < std::min(streamCUMask.size(), baseMask.size()); ++i) {
    mask.push_back(streamCUMask[i] & baseMask[i]);
  }

  // If the AND result is all zeros, fall back to the base mask.
  const bool allZero = std::all_of(mask.begin(), mask.end(),
                                    [](uint32_t m) { return m == 0; });
  if (allZero) {
    mask = baseMask;
  }
  std::copy(mask.begin(), mask.end(), cuMask);

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  HIP_INIT_API(hipStreamGetDevice, stream, device);

  if (device == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  if (stream == nullptr) {
    *device = hip::getCurrentDevice()->deviceId();
    HIP_RETURN(hipSuccess);
  }

  getStreamPerThread(stream);
  *device = reinterpret_cast<hip::Stream*>(stream)->DeviceId();
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 const hipStreamAttrValue* value) {
  HIP_INIT_API(hipStreamSetAttribute, stream, attr, value);

  if (value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  getStreamPerThread(stream);

  // If stream is capturing, don't allow changing stream attributes.
  if (hip::Stream::StreamCaptureOngoing(stream)) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  constexpr bool wait = false;
  hip::Stream* s = hip::getStream(stream, wait);

  switch (attr) {
    case hipStreamAttributeSynchronizationPolicy: {
      const auto syncPolicy = value->syncPolicy;
      if (syncPolicy < hipSyncPolicyAuto || syncPolicy > hipSyncPolicyBlockingSync) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      s->SetSyncPolicy(static_cast<amd::SyncPolicy>(syncPolicy));
      break;
    }
    default: {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamGetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 hipStreamAttrValue* value_out) {
  HIP_INIT_API(hipStreamGetAttribute, stream, attr, value_out);

  if (value_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  getStreamPerThread(stream);

  constexpr bool wait = false;
  const auto* s = hip::getStream(stream, wait);

  switch (attr) {
    case hipStreamAttributeSynchronizationPolicy: {
      value_out->syncPolicy = static_cast<hipSynchronizationPolicy>(s->GetSyncPolicy());
      break;
    }
    case hipStreamAttributePriority: {
      value_out->priority = s->GetPriority();
      break;
    }
    default: {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src) {
  HIP_INIT_API(hipStreamCopyAttributes, dst, src);

  if (!hip::isValid(src) || !hip::isValid(dst)) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  getStreamPerThread(src);
  getStreamPerThread(dst);

  constexpr bool wait = false;
  auto* src_stream = hip::getStream(src, wait);
  auto* dst_stream = hip::getStream(dst, wait);
  // Currently, SyncPolicy is the only stream attribute we can set during runtime.
  dst_stream->SetSyncPolicy(src_stream->GetSyncPolicy());
  HIP_RETURN(hipSuccess);
}
}  // namespace hip
