/* Copyright (c) 2024 Advanced Micro Devices, Inc.

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

#include "device/pal/palubercapturemgr.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palkernel.hpp"
#include "device/pal/palblit.hpp"

#include "palPlatform.h"
#include "palTraceSession.h"
#include "palRenderOpTraceController.h"
#include "palCodeObjectTraceSource.h"
#include "palQueueTimingsTraceSource.h"

#include "devDriverServer.h"
#include "protocols/driverControlServer.h"
#include "util/ddStructuredReader.h"

namespace amd::pal {

// ================================================================================================
// Returns true if the given device properties/settings support tracing.
static inline bool GpuSupportsTracing(const Pal::DeviceProperties& props,
                                      const Settings& settings) {
  return props.gfxipProperties.flags.supportRgpTraces && !settings.rgpSqttForceDisable_;
}

// ================================================================================================
// Creates the GPU Open Developer Mode manager class.
UberTraceCaptureMgr* UberTraceCaptureMgr::Create(Pal::IPlatform* platform, const Device& device) {
  UberTraceCaptureMgr* mgr = new UberTraceCaptureMgr(platform, device);

  if (mgr != nullptr && !mgr->Init(platform)) {
    delete mgr;
    mgr = nullptr;
  }

  return mgr;
}

// ================================================================================================
UberTraceCaptureMgr::UberTraceCaptureMgr(Pal::IPlatform* platform, const Device& device)
    : device_(device),
      dev_driver_server_(platform->GetDevDriverServer()),
      global_disp_count_(1),  // Must start from 1 according to RGP spec
      user_event_(nullptr),
      current_event_id_(0),
      trace_session_(platform->GetTraceSession()),
      trace_controller_(nullptr),
      code_object_trace_source_(nullptr),
      queue_timings_trace_source_(nullptr) {}

// ================================================================================================
UberTraceCaptureMgr::~UberTraceCaptureMgr() { DestroyUberTraceResources(); }

// ================================================================================================
bool UberTraceCaptureMgr::CreateUberTraceResources(Pal::IPlatform* platform) {
  bool success = false;

  do {
    // Create the user event RGP marker
    user_event_ = new RgpSqttMarkerUserEventWithString;
    if (user_event_ == nullptr) {
      break;
    }

    // Initialize the renderop trace controller
    trace_controller_ = new GpuUtil::RenderOpTraceController(platform, device_.iDev());
    if (trace_controller_ == nullptr) {
      break;
    }

    Pal::Result result = trace_session_->RegisterController(trace_controller_);
    if (result != Pal::Result::Success) {
      break;
    }

    // Initialize the code object trace source
    code_object_trace_source_ = new GpuUtil::CodeObjectTraceSource(platform);
    if (code_object_trace_source_ == nullptr) {
      break;
    }

    result = trace_session_->RegisterSource(code_object_trace_source_);
    if (result != Pal::Result::Success) {
      break;
    }

    // Initialize the queue timings trace source
    queue_timings_trace_source_ = new GpuUtil::QueueTimingsTraceSource(platform);
    if (queue_timings_trace_source_ == nullptr) {
      break;
    }

    result = trace_session_->RegisterSource(queue_timings_trace_source_);
    if (result != Pal::Result::Success) {
      break;
    }

    success = true;
  } while (false);

  return success;
}

// ================================================================================================
void UberTraceCaptureMgr::DestroyUberTraceResources() {
  // RGP user event marker
  if (user_event_ != nullptr) {
    delete user_event_;
    user_event_ = nullptr;
  }

  // RenderOp TraceController
  if (trace_controller_ != nullptr) {
    trace_session_->UnregisterController(trace_controller_);
    delete trace_controller_;
    trace_controller_ = nullptr;
  }

  // CodeObjects TraceSource
  if (code_object_trace_source_ != nullptr) {
    trace_session_->UnregisterSource(code_object_trace_source_);
    delete code_object_trace_source_;
    code_object_trace_source_ = nullptr;
  }

  // QueueTimings TraceSource
  if (queue_timings_trace_source_ != nullptr) {
    trace_session_->UnregisterSource(queue_timings_trace_source_);
    delete queue_timings_trace_source_;
    queue_timings_trace_source_ = nullptr;
  }
}

// ================================================================================================
bool UberTraceCaptureMgr::Init(Pal::IPlatform* platform) {
  // Finalize the devmode manager
  if (dev_driver_server_ == nullptr) {
    return false;
  }
  dev_driver_server_->Finalize();

  // Initialize the trace sources & controllers owned by the compute driver
  const bool success = CreateUberTraceResources(platform);

  if (!success) {
    DestroyUberTraceResources();
    return false;
  }

  return true;
}

// ================================================================================================
void UberTraceCaptureMgr::PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel, size_t x,
                                      size_t y, size_t z) {
  // Wait for the driver to be resumed in case it's been paused.
  WaitForDriverResume();

  // Increment dispatch count in RenderOp trace controller
  Pal::IQueue* pQueue = gpu->queue(MainEngine).iQueue_;
  GpuUtil::RenderOpCounts opCounts = {
      .dispatchCount = 1u,
  };
  trace_controller_->RecordRenderOps(pQueue, opCounts);

  if (trace_session_->GetTraceSessionState() == GpuUtil::TraceSessionState::Running) {
    RgpSqttMarkerEventType apiEvent = RgpSqttMarkerEventType::CmdNDRangeKernel;

    if (kernel.prog().isInternal()) {
      constexpr RgpSqttMarkerEventType ApiEvents[KernelBlitManager::BlitTotal] = {
          RgpSqttMarkerEventType::CmdCopyImage,
          RgpSqttMarkerEventType::CmdCopyImage,
          RgpSqttMarkerEventType::CmdCopyImageToBuffer,
          RgpSqttMarkerEventType::CmdCopyBufferToImage,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdFillBuffer,
          RgpSqttMarkerEventType::CmdFillImage,
          RgpSqttMarkerEventType::CmdScheduler};

      for (uint i = 0; i < KernelBlitManager::BlitTotal; ++i) {
        if (kernel.name().compare(BlitName[i]) == 0) {
          apiEvent = ApiEvents[i];
          break;
        }
      }
    }

    // Write the hash value
    WriteComputeBindMarker(gpu, kernel.prog().ApiHash());

    // Write dispatch marker
    WriteEventWithDimsMarker(gpu, apiEvent, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                             static_cast<uint32_t>(z));
  }

  // Increment the global dispatch counter
  global_disp_count_++;
}

// ================================================================================================
void UberTraceCaptureMgr::PostDispatch(VirtualGPU* gpu) {}

// ================================================================================================
// Waits for the driver to be resumed if it's currently paused.
void UberTraceCaptureMgr::WaitForDriverResume() {
  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();

  assert(pDriverControlServer != nullptr);
  pDriverControlServer->DriverTick();
}

// ================================================================================================
void UberTraceCaptureMgr::PreDeviceDestroy() {
  if (trace_session_->GetTraceSessionState() == GpuUtil::TraceSessionState::Ready) {
    DestroyUberTraceResources();
  }
}

// ================================================================================================
void UberTraceCaptureMgr::FinishRGPTrace(VirtualGPU* gpu, bool aborted) {
  // Nothing to be done
}

// ================================================================================================
bool UberTraceCaptureMgr::IsQueueTimingActive() const {
  return ((queue_timings_trace_source_ != nullptr) &&
          (queue_timings_trace_source_->IsTimingInProgress()));
}

// ================================================================================================
bool UberTraceCaptureMgr::RegisterTimedQueue(uint32_t queue_id, Pal::IQueue* iQueue,
                                             bool* debug_vmid) const {
  // Get the OS context handle for this queue (this is a thing that RGP needs on DX clients;
  // it may be optional for Vulkan, but we provide it anyway if available).
  Pal::KernelContextInfo kernelContextInfo = {};
  Pal::Result result = iQueue->QueryKernelContextInfo(&kernelContextInfo);

  // QueryKernelContextInfo may fail.
  // If so, just use a context identifier of 0.
  uint64_t queueContext =
      (result == Pal::Result::Success) ? kernelContextInfo.contextIdentifier : 0;

  // Register the queue with the GPA session class for timed queue operation support.
  result = queue_timings_trace_source_->RegisterTimedQueue(iQueue, queue_id, queueContext);

  return (result == Pal::Result::Success);
}

// ================================================================================================
Pal::Result UberTraceCaptureMgr::TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                                                  const Pal::SubmitInfo& submitInfo) const {
  // Fill in extra meta-data information to associate the API command buffer data with
  // the generated timing information.
  GpuUtil::TimedSubmitInfo timedSubmitInfo = {};
  Pal::uint64 apiCmdBufIds = cmdId;
  Pal::uint32 sqttCmdBufIds = 0;

  timedSubmitInfo.pApiCmdBufIds = &apiCmdBufIds;
  timedSubmitInfo.pSqttCmdBufIds = &sqttCmdBufIds;
  timedSubmitInfo.frameIndex = 0;

  // Do a timed submit of all the command buffers
  Pal::Result result = queue_timings_trace_source_->TimedSubmit(queue, submitInfo, timedSubmitInfo);

  // Punt to non-timed submit if a timed submit fails (or is not supported)
  if (result != Pal::Result::Success) {
    result = queue->Submit(submitInfo);
  }

  return result;
}

// ================================================================================================
bool UberTraceCaptureMgr::Update(Pal::IPlatform* platform) {
  Pal::Result result = queue_timings_trace_source_->Init(device_.iDev());
  return (result == Pal::Result::Success);
}

// ================================================================================================
uint64_t UberTraceCaptureMgr::AddElfBinary(const void* exe_binary, size_t exe_binary_size,
                                           const void* elf_binary, size_t elf_binary_size,
                                           Pal::IGpuMemory* pGpuMemory, size_t offset) {
  GpuUtil::ElfBinaryInfo elfBinaryInfo = {};
  elfBinaryInfo.pBinary = exe_binary;
  elfBinaryInfo.binarySize = exe_binary_size;  ///< FAT Elf binary size.
  elfBinaryInfo.pGpuMemory = pGpuMemory;       ///< GPU Memory where the compiled ISA resides.
  elfBinaryInfo.offset = static_cast<Pal::gpusize>(offset);

  elfBinaryInfo.originalHash = DevDriver::MetroHash::MetroHash64(
      reinterpret_cast<const DevDriver::uint8*>(elf_binary), elf_binary_size);

  elfBinaryInfo.compiledHash = DevDriver::MetroHash::MetroHash64(
      reinterpret_cast<const DevDriver::uint8*>(exe_binary), exe_binary_size);

  assert(code_object_trace_source_ != nullptr);

  code_object_trace_source_->RegisterElfBinary(elfBinaryInfo);

  return elfBinaryInfo.originalHash;
}

// ================================================================================================
void UberTraceCaptureMgr::WriteMarker(const VirtualGPU* gpu, const void* data,
                                      size_t data_size) const {
  assert((data_size % sizeof(uint32_t)) == 0);
  assert((data_size / sizeof(uint32_t)) > 0);

  Pal::RgpMarkerSubQueueFlags subQueueFlags = {};
  subQueueFlags.includeMainSubQueue = 1;

  gpu->queue(MainEngine)
      .iCmd()
      ->CmdInsertRgpTraceMarker(subQueueFlags, static_cast<uint32_t>(data_size / sizeof(uint32_t)),
                                data);
}

// ================================================================================================
// Inserts a compute bind marker
void UberTraceCaptureMgr::WriteComputeBindMarker(const VirtualGPU* gpu, uint64_t api_hash) const {
  RgpSqttMarkerPipelineBind marker = {};
  marker.identifier = RgpSqttMarkerIdentifierBindPipeline;
  marker.cbID = gpu->queue(MainEngine).cmdBufId();
  marker.bindPoint = 1;

  memcpy(marker.apiPsoHash, &api_hash, sizeof(api_hash));
  WriteMarker(gpu, &marker, sizeof(marker));
}

// ================================================================================================
// Inserts an RGP pre-dispatch marker
void UberTraceCaptureMgr::WriteEventWithDimsMarker(const VirtualGPU* gpu,
                                                   RgpSqttMarkerEventType apiType, uint32_t x,
                                                   uint32_t y, uint32_t z) const {
  assert(apiType != RgpSqttMarkerEventType::Invalid);

  RgpSqttMarkerEvent event = {};
  event.identifier = RgpSqttMarkerIdentifierEvent;
  event.apiType = static_cast<uint32_t>(apiType);
  event.cmdID = current_event_id_++;
  event.cbID = gpu->queue(MainEngine).cmdBufId();

  RgpSqttMarkerEventWithDims eventWithDims = {};
  eventWithDims.event = event;
  eventWithDims.event.hasThreadDims = 1;
  eventWithDims.threadX = x;
  eventWithDims.threadY = y;
  eventWithDims.threadZ = z;

  WriteMarker(gpu, &eventWithDims, sizeof(eventWithDims));
}

// ================================================================================================
void UberTraceCaptureMgr::WriteBarrierStartMarker(const VirtualGPU* gpu,
                                                  const Pal::Developer::BarrierData& data) const {
  if (trace_session_->GetTraceSessionState() == GpuUtil::TraceSessionState::Running) {
    amd::ScopedLock traceLock(&trace_mutex_);

    RgpSqttMarkerBarrierStart marker = {};
    marker.cbId = gpu->queue(MainEngine).cmdBufId();
    marker.identifier = RgpSqttMarkerIdentifierBarrierStart;
    marker.internal = true;
    marker.dword02 = data.reason;

    WriteMarker(gpu, &marker, sizeof(marker));
  }
}

// ================================================================================================
void UberTraceCaptureMgr::WriteBarrierEndMarker(const VirtualGPU* gpu,
                                                const Pal::Developer::BarrierData& data) const {
  if (trace_session_->GetTraceSessionState() == GpuUtil::TraceSessionState::Running) {
    amd::ScopedLock traceLock(&trace_mutex_);

    // Copy the operations part and include the same data from previous markers
    // within the same barrier sequence to create a full picture of all cache
    // syncs and pipeline stalls.
    Pal::Developer::BarrierOperations operations = data.operations;
    operations.pipelineStalls.u16All |= 0;
    operations.caches.u16All |= 0;

    RgpSqttMarkerBarrierEnd marker = {};
    marker.identifier = RgpSqttMarkerIdentifierBarrierEnd;
    marker.cbId = gpu->queue(MainEngine).cmdBufId();
    marker.numLayoutTransitions = 0;
    marker.waitOnEopTs = operations.pipelineStalls.eopTsBottomOfPipe;
    marker.vsPartialFlush = operations.pipelineStalls.vsPartialFlush;
    marker.psPartialFlush = operations.pipelineStalls.psPartialFlush;
    marker.csPartialFlush = operations.pipelineStalls.csPartialFlush;
    marker.pfpSyncMe = operations.pipelineStalls.pfpSyncMe;
    marker.syncCpDma = operations.pipelineStalls.syncCpDma;
    marker.invalTcp = operations.caches.invalTcp;
    marker.invalSqI = operations.caches.invalSqI$;
    marker.invalSqK = operations.caches.invalSqK$;
    marker.flushTcc = operations.caches.flushTcc;
    marker.invalTcc = operations.caches.invalTcc;
    marker.flushCb = operations.caches.flushCb;
    marker.invalCb = operations.caches.invalCb;
    marker.flushDb = operations.caches.flushDb;
    marker.invalDb = operations.caches.invalDb;

    WriteMarker(gpu, &marker, sizeof(marker));
  }
}

}  // namespace amd::pal
