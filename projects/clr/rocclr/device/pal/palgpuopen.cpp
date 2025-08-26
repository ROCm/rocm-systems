/* Copyright (c) 2016 - 2021 Advanced Micro Devices, Inc.

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

#include "device/pal/palgpuopen.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palkernel.hpp"
#include "device/pal/palblit.hpp"

// PAL headers
#include "palCmdAllocator.h"
#include "palFence.h"
#include "palQueueSemaphore.h"

#ifdef PAL_GPUOPEN_OCL
// gpuutil headers
#include "gpuUtil/palGpaSession.h"

// gpuopen headers
#include "devDriverServer.h"
#include "msgChannel.h"
#include "msgTransport.h"
#include "protocols/rgpServer.h"
#include "protocols/driverControlServer.h"

namespace amd::pal {
// ================================================================================================
RgpCaptureMgr::RgpCaptureMgr(Pal::IPlatform* platform, const Device& device)
    : device_(device),
      dev_driver_server_(platform->GetDevDriverServer()),
      user_event_(nullptr),
      num_prep_disp_(0),
      max_sqtt_disp_(device_.settings().rgpSqttDispCount_),
      trace_gpu_mem_limit_(0),
      global_disp_count_(1),  // Must start from 1 according to RGP spec
      se_mask_(0),
      perf_counter_mem_limit_(0),
      perf_counter_frequency_(0),
      value_(0) {
  memset(&trace_, 0, sizeof(trace_));
}

// ================================================================================================
RgpCaptureMgr::~RgpCaptureMgr() { DestroyRGPTracing(); }

// ================================================================================================
// Creates the GPU Open Developer Mode manager class.
RgpCaptureMgr* RgpCaptureMgr::Create(Pal::IPlatform* platform, const Device& device) {
  RgpCaptureMgr* mgr = new RgpCaptureMgr(platform, device);

  if (mgr != nullptr && !mgr->Init(platform)) {
    delete mgr;
    mgr = nullptr;
  }

  return mgr;
}

// ================================================================================================
uint64_t RgpCaptureMgr::AddElfBinary(const void* exe_binary, size_t exe_binary_size,
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

  assert(trace_.gpa_session_ != nullptr);

  trace_.gpa_session_->RegisterElfBinary(elfBinaryInfo);
  return elfBinaryInfo.originalHash;
}

// ================================================================================================
bool RgpCaptureMgr::Init(Pal::IPlatform* platform) {
  if (dev_driver_server_ == nullptr) {
    return false;
  }
  // Tell RGP that the server (i.e. the driver) supports tracing if requested.
  rgp_server_ = dev_driver_server_->GetRGPServer();
  if (rgp_server_ == nullptr) {
    return false;
  }

  // Finalize RGP settings
  Finalize();

  return true;
}

// ================================================================================================
bool RgpCaptureMgr::Update(Pal::IPlatform* platform) {
  bool result = true;

  const Settings& settings = device_.settings();
  // Fail initialization of trace resources if SQTT tracing has been force-disabled from
  // the panel (this will consequently fail the trace), or if the chosen device's gfxip
  // does not support SQTT.
  //
  // It's necessary to check this during RGP tracing init in addition to devmode init because
  // during the earlier devmode init we may be in a situation where some enumerated physical
  // devices support tracing and others do not.
  if (GpuSupportsTracing(device_.properties(), settings) == false) {
    result = false;
  }

  // Create a GPA session object for this trace session
  if (result) {
    assert(trace_.gpa_session_ == nullptr);

    const uint32_t api_version = settings.oclVersion_;

    trace_.gpa_session_ = new GpuUtil::GpaSession(
        platform, device_.iDev(),
        api_version >> 4,   // OCL API version major
        api_version & 0xf,  // OCL API version minor
        (amd::IS_HIP) ? GpuUtil::ApiType::Hip : GpuUtil::ApiType::OpenCl,
        RgpSqttInstrumentationSpecVersion, RgpSqttInstrumentationApiVersion);

    if (trace_.gpa_session_ == nullptr) {
      result = false;
    }
  }

  // Initialize the GPA session
  if (result && (trace_.gpa_session_->Init() != Pal::Result::Success)) {
    result = false;
  }

  if (result) {
    user_event_ = new RgpSqttMarkerUserEventWithString;
    if (nullptr == user_event_) {
      result = false;
    }
  }

  if (!result) {
    // If we've failed to initialize tracing, permanently disable traces
    if (rgp_server_ != nullptr) {
      rgp_server_->DisableTraces();

      trace_enabled_ = false;
    }

    // Clean up if we failed
    DestroyRGPTracing();
  } else {
    PostDeviceCreate();
  }

  static_vm_id_ = device_.properties().gfxipProperties.flags.supportStaticVmid;

  return result;
}

// ================================================================================================
// This function finds out all the queues in the device that we have to synchronize for RGP-traced
// frames and initializes resources for them.
bool RgpCaptureMgr::RegisterTimedQueue(uint32_t queue_id, Pal::IQueue* iQueue,
                                       bool* debug_vmid) const {
  bool result = true;

  // Get the OS context handle for this queue (this is a thing that RGP needs on DX clients;
  // it may be optional for Vulkan, but we provide it anyway if available).
  Pal::KernelContextInfo kernelContextInfo = {};
  Pal::Result palResult = iQueue->QueryKernelContextInfo(&kernelContextInfo);

  // Ensure we've acquired the debug VMID (note that some platforms do not
  // implement this function, so don't fail the whole trace if so)
  *debug_vmid = kernelContextInfo.flags.hasDebugVmid;
  assert((static_vm_id_ || *debug_vmid) && "Can't capture multiple queues!");

  // Register the queue with the GPA session class for timed queue operation support.
  if (trace_.gpa_session_->RegisterTimedQueue(
          iQueue, queue_id, kernelContextInfo.contextIdentifier) != Pal::Result::Success) {
    result = false;
  }

  return result;
}

// ================================================================================================
Pal::Result RgpCaptureMgr::TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
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
  Pal::Result result = trace_.gpa_session_->TimedSubmit(queue, submitInfo, timedSubmitInfo);

  // Punt to non-timed submit if a timed submit fails (or is not supported)
  if (result != Pal::Result::Success) {
    result = queue->Submit(submitInfo);
  }

  return result;
}

// ================================================================================================
// Called during initial device enumeration prior to calling Pal::IDevice::CommitSettingsAndInit().
//
// This finalizes the developer driver manager.
void RgpCaptureMgr::Finalize() {
  // Figure out if the gfxip supports tracing.  We decide tracing if there is at least one
  // enumerated GPU that can support tracing.  Since we don't yet know if that GPU will be
  // picked as the target of an eventual VkDevice, this check is imperfect.
  // In mixed-GPU situations where an unsupported GPU is picked for tracing,
  // trace capture will fail with an error.
  bool hw_support_tracing = GpuSupportsTracing(device_.properties(), device_.settings());

  if (hw_support_tracing == false) {
    rgp_server_->DisableTraces();
  }

  // Finalize the devmode manager
  dev_driver_server_->Finalize();

  // Figure out if tracing support should be enabled or not
  trace_enabled_ = (rgp_server_ != nullptr) && rgp_server_->TracesEnabled();
}


// ================================================================================================
// Waits for the driver to be resumed if it's currently paused.
void RgpCaptureMgr::WaitForDriverResume() {
  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();

  assert(pDriverControlServer != nullptr);

  pDriverControlServer->DriverTick();
}

// ================================================================================================
// Called before a swap chain presents.  This signals a frame-end boundary and
// is used to coordinate RGP trace start/stop.
void RgpCaptureMgr::PostDispatch(VirtualGPU* gpu) {
  if (rgp_server_->TracesEnabled()) {
    // If there's currently a trace running, submit the trace-end command buffer
    if (trace_.status_ == TraceStatus::Running) {
      amd::ScopedLock traceLock(&trace_mutex_);
      trace_.sqtt_disp_count_++;
      if (trace_.sqtt_disp_count_ >= max_sqtt_disp_) {
        Pal::Result res = EndRGPHardwareTrace(gpu);
        if (Pal::Result::ErrorIncompatibleQueue == res) {
          // continue until we find the right queue...
        } else if (Pal::Result::Success == res) {
          trace_.sqtt_disp_count_ = 0;
          // Stop the trace and save the result. Currently runtime can't delay upload in HIP,
          // because default stream doesn't have explicit destruction and
          // OS kills all threads on exit without any notification. That includes PAL RGP threads.
          {
            if (trace_.status_ == TraceStatus::WaitingForSqtt) {
              auto result = EndRGPTrace(gpu);
            }
            // Check if runtime is waiting for the final trace results
            if (trace_.status_ == TraceStatus::WaitingForResults) {
              // If results are ready, then finish the trace
              if (CheckForTraceResults() == Pal::Result::Success) {
                FinishRGPTrace(gpu, false);
              }
            }
          }
        } else {
          FinishRGPTrace(gpu, true);
        }
      }
    }

    if (IsQueueTimingActive()) {
      // Call TimedQueuePresent() to insert commands that collect GPU timestamp.
      Pal::IQueue* pPalQueue = gpu->queue(MainEngine).iQueue_;

      // Currently nothing in the PresentInfo struct is used for inserting a timed present marker.
      GpuUtil::TimedQueuePresentInfo timedPresentInfo = {};
      // Pal::Result result = trace_.gpa_session_->TimedQueuePresent(pPalQueue, timedPresentInfo);
      // assert(result == Pal::Result::Success);
    }
  }
}

// ================================================================================================
Pal::Result RgpCaptureMgr::CheckForTraceResults() {
  assert(trace_.status_ == TraceStatus::WaitingForResults);

  Pal::Result result = Pal::Result::NotReady;

  // Check if trace results are ready
  if (trace_.gpa_session_->IsReady() &&                   // GPA session is ready
      (trace_.begin_queue_->isDone(&trace_.end_event_)))  // "Trace end" cmdbuf has retired
  {
    bool success = false;

    // Fetch required trace data size from GPA session
    size_t traceDataSize = 0;
    void* pTraceData = nullptr;

    trace_.gpa_session_->GetResults(trace_.gpa_sample_id_, &traceDataSize, nullptr);

    // Allocate memory for trace data
    if (traceDataSize > 0) {
      pTraceData = amd::AlignedMemory::allocate(traceDataSize, 256);
    }

    if (pTraceData != nullptr) {
      // Get trace data from GPA session
      if (trace_.gpa_session_->GetResults(trace_.gpa_sample_id_, &traceDataSize, pTraceData) ==
          Pal::Result::Success) {
        // Transmit trace data to anyone who's listening
        auto devResult =
            rgp_server_->WriteTraceData(static_cast<Pal::uint8*>(pTraceData), traceDataSize);

        success = (devResult == DevDriver::Result::Success);
      }

      amd::AlignedMemory::deallocate(pTraceData);
    }

    if (success) {
      result = Pal::Result::Success;
    }
  }

  return result;
}

// ================================================================================================
// Called after a swap chain presents.  This signals a (next) frame-begin boundary and is
// used to coordinate RGP trace start/stop.
void RgpCaptureMgr::PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel, size_t x, size_t y,
                                size_t z) {
  // Wait for the driver to be resumed in case it's been paused.
  WaitForDriverResume();

  if (rgp_server_->TracesEnabled()) {
    amd::ScopedLock traceLock(&trace_mutex_);

    // Check if there's an RGP trace request pending and we're idle
    if ((trace_.status_ == TraceStatus::Idle) && rgp_server_->IsTracePending()) {
      // Attempt to start preparing for a trace
      if (PrepareRGPTrace(gpu) == Pal::Result::Success) {
        // Attempt to start the trace immediately if we do not need to prepare
        if (num_prep_disp_ == 0) {
          if (BeginRGPTrace(gpu) != Pal::Result::Success) {
            FinishRGPTrace(gpu, true);
          }
        }
      }
    } else if (trace_.status_ == TraceStatus::Preparing) {
      // Wait some number of "preparation frames" before starting the trace in order to get enough
      // timer samples to sync CPU/GPU clock domains.
      trace_.prepared_disp_count_++;

      // Take a calibration timing measurement sample for this frame.
      trace_.gpa_session_->SampleTimingClocks();

      // Start the SQTT trace if we've waited a sufficient number of preparation frames
      if (trace_.prepared_disp_count_ >= num_prep_disp_) {
        Pal::Result result = BeginRGPTrace(gpu);

        if (Pal::Result::ErrorIncompatibleQueue == result) {
          // Let's wait until the app will reach the same queue
        } else if (result != Pal::Result::Success) {
          FinishRGPTrace(gpu, true);
        }
      }
    }
    // Check if we're ending a trace waiting for SQTT to turn off.
    // If SQTT has turned off, end the trace
    else if (trace_.status_ == TraceStatus::WaitingForSqtt) {
      Pal::Result result = Pal::Result::Success;

      if (trace_.begin_queue_->isDone(&trace_.end_sqtt_event_)) {
        result = EndRGPTrace(gpu);
      } else {
        // todo: There is a wait inside the trace end for now
        result = EndRGPTrace(gpu);
      }

      if (result != Pal::Result::Success) {
        FinishRGPTrace(gpu, true);
      }
    }
    // Check if we're waiting for final trace results.
    else if (trace_.status_ == TraceStatus::WaitingForResults) {
      Pal::Result result = CheckForTraceResults();

      // Results ready: finish trace
      if (result == Pal::Result::Success) {
        FinishRGPTrace(gpu, false);
      }
      // Error while computing results: abort trace
      else if (result != Pal::Result::NotReady) {
        FinishRGPTrace(gpu, true);
      }
    }

    if (trace_.status_ == TraceStatus::Running) {
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

      WriteUserEventMarker(gpu, RgpSqttMarkerUserEventObjectName, kernel.name());

      // Write dispatch marker
      WriteEventWithDimsMarker(gpu, apiEvent, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                               static_cast<uint32_t>(z));
    }
  }

  global_disp_count_++;
}

// ================================================================================================
// This function starts preparing for an RGP trace.  Preparation involves some N frames of
// lead-up time during which timing samples are accumulated to synchronize CPU and GPU clock
// domains.
//
// This function transitions from the Idle state to the Preparing state.
Pal::Result RgpCaptureMgr::PrepareRGPTrace(VirtualGPU* gpu) {
  assert(trace_.status_ == TraceStatus::Idle);

  // We can only trace using a single device at a time currently, so recreate RGP trace
  // resources against this new one if the device is changing.
  Pal::Result result = Pal::Result::Success;

  const auto traceParameters = rgp_server_->QueryTraceParameters();

  num_prep_disp_ = traceParameters.captureStartIndex;
  uint32_t capture_disp = traceParameters.captureStopIndex - traceParameters.captureStartIndex;
  // Validate if the captured dispatches are in the range
  if ((capture_disp > 0) && (capture_disp < max_sqtt_disp_)) {
    max_sqtt_disp_ = capture_disp;
  }

  trace_gpu_mem_limit_ = traceParameters.gpuMemoryLimitInMb * 1024 * 1024;
  inst_tracing_enabled_ = traceParameters.flags.enableInstructionTokens;
  se_mask_ = traceParameters.seMask;

  // Setup streamed performance counters
  perf_counters_enabled_ = (traceParameters.flags.enableSpm != 0);

  DevDriver::RGPProtocol::ServerSpmConfig counter_config = {};
  DevDriver::Vector<DevDriver::RGPProtocol::ServerSpmCounterId> counters(
      dev_driver_server_->GetMessageChannel()->GetAllocCb());
  rgp_server_->QuerySpmConfig(&counter_config, &counters);

  Pal::PerfExperimentProperties perf_properties = {};

  result = gpu->dev().iDev()->GetPerfExperimentProperties(&perf_properties);

  // Querying performance properties should never fail
  assert(result == Pal::Result::Success);

  perf_counter_frequency_ = counter_config.sampleFrequency;
  perf_counter_mem_limit_ = counter_config.memoryLimitInMb * 1024 * 1024;

  perf_counter_ids_.clear();

  for (size_t idx = 0; idx < counters.Size(); ++idx) {
    const DevDriver::RGPProtocol::ServerSpmCounterId server_counter = counters[idx];
    const Pal::GpuBlockPerfProperties& block_perf_prop =
        perf_properties.blocks[server_counter.blockId];

    if (server_counter.instanceId == DevDriver::RGPProtocol::kSpmAllInstancesId) {
      for (uint32_t instance = 0; instance < block_perf_prop.instanceCount; ++instance) {
        GpuUtil::PerfCounterId counter_id = {};
        counter_id.block = static_cast<Pal::GpuBlock>(server_counter.blockId);
        counter_id.instance = instance;
        counter_id.eventId = server_counter.eventId;

        perf_counter_ids_.push_back(counter_id);
      }
    } else {
      GpuUtil::PerfCounterId counter_id = {};
      counter_id.block = static_cast<Pal::GpuBlock>(server_counter.blockId);
      counter_id.instance = server_counter.instanceId;
      counter_id.eventId = server_counter.eventId;

      perf_counter_ids_.push_back(counter_id);
    }
  }

  if (static_vm_id_) {
    result = device_.iDev()->SetStaticVmidMode(true);
    assert(result == Pal::Result::Success && "Static VM ID setup failed!");
  }

  if (result == Pal::Result::Success) {
    // Notify the RGP server that we are starting a trace
    if (rgp_server_->BeginTrace() != DevDriver::Result::Success) {
      result = Pal::Result::ErrorUnknown;
    }
  }
  // Tell the GPA session class we're starting a trace
  if (result == Pal::Result::Success) {
    GpuUtil::GpaSessionBeginInfo info = {};

    info.flags.enableQueueTiming = true;  // trace_.queueTimingEnabled;

    result = trace_.gpa_session_->Begin(info);
  }

  trace_.prepared_disp_count_ = 0;
  trace_.sqtt_disp_count_ = 0;

  // Sample the timing clocks prior to starting a trace.
  if (result == Pal::Result::Success) {
    trace_.gpa_session_->SampleTimingClocks();
  }

  if (result == Pal::Result::Success) {
    trace_.begin_queue_ = nullptr;
    trace_.status_ = TraceStatus::Preparing;
  } else {
    // We failed to prepare for the trace so abort it.
    if (rgp_server_ != nullptr) {
      const DevDriver::Result devDriverResult = rgp_server_->AbortTrace();

      // AbortTrace should always succeed unless we've used the api incorrectly.
      assert(devDriverResult == DevDriver::Result::Success);
    }
  }

  return result;
}

// ================================================================================================
// This function begins an RGP trace by initializing all dependent resources and submitting
// the "begin trace" information command buffer.
//
// This function transitions from the Preparing state to the Running state.
Pal::Result RgpCaptureMgr::BeginRGPTrace(VirtualGPU* gpu) {
  assert(trace_.status_ == TraceStatus::Preparing);
  assert(trace_enabled_);

  // We can only trace using a single device at a time currently, so recreate RGP trace
  // resources against this new one if the device is changing.
  Pal::Result result = Pal::Result::Success;


  // Start a GPA tracing sample with SQTT enabled
  if (result == Pal::Result::Success) {
    GpuUtil::GpaSampleConfig sampleConfig = {};

    sampleConfig.type = GpuUtil::GpaSampleType::Trace;
    // Configure SQTT
    sampleConfig.sqtt.gpuMemoryLimit = trace_gpu_mem_limit_;
    sampleConfig.sqtt.seDetailedMask = se_mask_;

    sampleConfig.sqtt.flags.enable = true;
    sampleConfig.sqtt.flags.supressInstructionTokens = (inst_tracing_enabled_ == false);

    // Configure SPM
    if (perf_counters_enabled_ && !perf_counter_ids_.empty()) {
      sampleConfig.perfCounters.gpuMemoryLimit = perf_counter_mem_limit_;
      sampleConfig.perfCounters.spmTraceSampleInterval = perf_counter_frequency_;
      sampleConfig.perfCounters.numCounters = perf_counter_ids_.size();
      sampleConfig.perfCounters.pIds = perf_counter_ids_.data();
    }

    // Fill GPU commands
    gpu->eventBegin(MainEngine);
    result = trace_.gpa_session_->BeginSample(gpu->queue(MainEngine).iCmd(), sampleConfig,
                                              &trace_.gpa_sample_id_);
    gpu->eventEnd(MainEngine, trace_.begin_sqtt_event_);
  }

  if (result == Pal::Result::Success) {
    GpuUtil::SampleTraceApiInfo sample_trace_api_info = {};
    sample_trace_api_info.instructionTraceMode = (inst_tracing_enabled_)
                                                     ? GpuUtil::InstructionTraceMode::FullFrame
                                                     : GpuUtil::InstructionTraceMode::Disabled;
    trace_.gpa_session_->SetSampleTraceApiInfo(sample_trace_api_info, trace_.gpa_sample_id_);
  }

  // Submit the trace-begin command buffer
  if (result == Pal::Result::Success) {
    static constexpr bool NeedFlush = true;
    // Update the global GPU event
    gpu->setGpuEvent(trace_.begin_sqtt_event_, NeedFlush);
  }

  // Make the trace active and remember which queue started it
  if (result == Pal::Result::Success) {
    trace_.status_ = TraceStatus::Running;
    trace_.begin_queue_ = gpu;
  }

  return result;
}

// ================================================================================================
// This function submits the command buffer to stop SQTT tracing.  Full tracing still continues.
//
// This function transitions from the Running state to the WaitingForSqtt state.
Pal::Result RgpCaptureMgr::EndRGPHardwareTrace(VirtualGPU* gpu) {
  assert(trace_.status_ == TraceStatus::Running);

  Pal::Result result = Pal::Result::Success;

  // Only allow SQTT trace to start and end on the same queue because it's critical that these are
  // in the same order
  if (gpu != trace_.begin_queue_) {
    result = Pal::Result::ErrorIncompatibleQueue;
  }

  // Tell the GPA session to insert any necessary commands to end the tracing sample and
  // end the session itself
  if (result == Pal::Result::Success) {
    assert(trace_.gpa_session_ != nullptr);

    // Write CB commands to finish the SQTT
    gpu->eventBegin(MainEngine);
    trace_.gpa_session_->EndSample(gpu->queue(MainEngine).iCmd(), trace_.gpa_sample_id_);
    gpu->eventEnd(MainEngine, trace_.end_sqtt_event_);

    static constexpr bool NeedFlush = true;
    // Update the global GPU event
    gpu->setGpuEvent(trace_.end_sqtt_event_, NeedFlush);

    trace_.status_ = TraceStatus::WaitingForSqtt;

    // Execute a device wait idle
    if (device_.settings().rgpSqttWaitIdle_) {
      // Make sure the trace is done. Note: required for SDMA data write back
      gpu->waitForEvent(&trace_.end_sqtt_event_);
    }
  }

  return result;
}

// ================================================================================================
// This function ends a running RGP trace.
//
// This function transitions from the WaitingForSqtt state to WaitingForResults state.
Pal::Result RgpCaptureMgr::EndRGPTrace(VirtualGPU* gpu) {
  assert(trace_.status_ == TraceStatus::WaitingForSqtt);

  Pal::Result result = Pal::Result::Success;

  // Tell the GPA session to insert any necessary commands to end the tracing sample and
  // end the session itself
  if (result == Pal::Result::Success) {
    assert(trace_.gpa_session_ != nullptr);
    EngineType engine = (gpu->dev().settings().disableSdma_) ? MainEngine : SdmaEngine;
    // Initiate SDMA copy
    gpu->eventBegin(engine);
    result = trace_.gpa_session_->End(gpu->queue(engine).iCmd());
    gpu->eventEnd(engine, trace_.end_event_);
  }

  // Submit the trace-end command buffer
  if (result == Pal::Result::Success) {
    static constexpr bool NeedFlush = true;
    // Update the global GPU event
    gpu->setGpuEvent(trace_.end_event_, NeedFlush);

    trace_.status_ = TraceStatus::WaitingForResults;

    if (device_.settings().rgpSqttWaitIdle_) {
      // Make sure the transfer is done
      gpu->waitForEvent(&trace_.end_event_);
    }
  }

  return result;
}

// ================================================================================================
// This function resets and possibly cancels a currently active (between begin/end) RGP trace.
// It frees any dependent resources.
void RgpCaptureMgr::FinishRGPTrace(VirtualGPU* gpu, bool aborted) {
  // Make sure current queue matches the capture queue
  if ((trace_.begin_queue_ == nullptr) || (trace_.begin_queue_ != gpu)) {
    return;
  }

  auto disp_count = trace_.sqtt_disp_count_;
  // Finish the trace if the queue was destroyed before OCL reached
  // the number of captured dispatches
  if (trace_.sqtt_disp_count_ != 0) {
    if (EndRGPHardwareTrace(gpu) != Pal::Result::Success) {
      aborted = true;
    }
  }
  // If the trace was aborted, then make sure the current results are sent to RGP server
  if (aborted) {
    if (trace_.status_ == TraceStatus::WaitingForSqtt) {
      auto result = EndRGPTrace(gpu);
      // The logic always checks for the trace status below and error can be ignored, since
      // runtime aborts the trace
    }
    // Check if runtime is waiting for the final trace results
    if (trace_.status_ == TraceStatus::WaitingForResults) {
      // If results are ready, then finish the trace
      if (CheckForTraceResults() == Pal::Result::Success) {
        rgp_server_->EndTrace();
      }
    }
  }

  // Inform RGP protocol that we're done with the trace, either by aborting it or finishing normally
  if (aborted) {
    rgp_server_->AbortTrace();
  } else {
    rgp_server_->EndTrace();
  }

  if (static_vm_id_) {
    auto result = device_.iDev()->SetStaticVmidMode(false);
    assert(result == Pal::Result::Success && "Static VM ID setup failed!");
  }

  if (trace_.gpa_session_ != nullptr) {
    trace_.gpa_session_->Reset();
  }
  // If applicaiton exits, then Windows kills all threads and
  // RGP can't finish data write into a file.
  amd::Os::sleep(10 * disp_count + 500);
  // Reset tracing state to idle
  trace_.prepared_disp_count_ = 0;
  trace_.sqtt_disp_count_ = 0;
  trace_.gpa_sample_id_ = 0;
  trace_.status_ = TraceStatus::Idle;
  trace_.begin_queue_ = nullptr;
}

// ================================================================================================
// Destroys device-persistent RGP resources
void RgpCaptureMgr::DestroyRGPTracing() {
  if (trace_.status_ != TraceStatus::Idle) {
    FinishRGPTrace(nullptr, true);
  }

  delete user_event_;

  // Destroy the GPA session
  // Util::Destructor(trace_.gpa_session_);
  delete trace_.gpa_session_;
  trace_.gpa_session_ = nullptr;

  memset(&trace_, 0, sizeof(trace_));
}

// ================================================================================================
// Returns true if the given device properties/settings support tracing.
bool RgpCaptureMgr::GpuSupportsTracing(const Pal::DeviceProperties& props,
                                       const Settings& settings) {
  return props.gfxipProperties.flags.supportRgpTraces && !settings.rgpSqttForceDisable_;
}

// ================================================================================================
// Called when a new device is created.  This will preallocate reusable RGP trace resources
// for that device.
void RgpCaptureMgr::PostDeviceCreate() {
  amd::ScopedLock traceLock(&trace_mutex_);

  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();

  assert(pDriverControlServer != nullptr);
}

// ================================================================================================
// Called prior to a device's being destroyed.  This will free persistent RGP trace resources for
// that device.
void RgpCaptureMgr::PreDeviceDestroy() {
  amd::ScopedLock traceLock(&trace_mutex_);
  // If we are idle, we can re-initialize trace resources based on the new device.
  if (trace_.status_ == TraceStatus::Idle) {
    DestroyRGPTracing();
  }
}

// ================================================================================================
// Sets up an Event marker's basic data.
RgpSqttMarkerEvent RgpCaptureMgr::BuildEventMarker(const VirtualGPU* gpu,
                                                   RgpSqttMarkerEventType api_type) const {
  RgpSqttMarkerEvent marker = {};

  marker.identifier = RgpSqttMarkerIdentifierEvent;
  marker.apiType = static_cast<uint32_t>(api_type);
  marker.cmdID = trace_.current_event_id_++;
  marker.cbID = gpu->queue(MainEngine).cmdBufId();

  return marker;
}

// ================================================================================================
void RgpCaptureMgr::WriteMarker(const VirtualGPU* gpu, const void* data, size_t data_size) const {
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
// Inserts an RGP pre-dispatch marker
void RgpCaptureMgr::WriteEventWithDimsMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType apiType,
                                             uint32_t x, uint32_t y, uint32_t z) const {
  assert(apiType != RgpSqttMarkerEventType::Invalid);

  RgpSqttMarkerEventWithDims eventWithDims = {};

  eventWithDims.event = BuildEventMarker(gpu, apiType);
  eventWithDims.event.hasThreadDims = 1;
  eventWithDims.threadX = x;
  eventWithDims.threadY = y;
  eventWithDims.threadZ = z;

  WriteMarker(gpu, &eventWithDims, sizeof(eventWithDims));
}

// ================================================================================================
void RgpCaptureMgr::WriteBarrierStartMarker(const VirtualGPU* gpu,
                                            const Pal::Developer::BarrierData& data) const {
  if (rgp_server_->TracesEnabled() && (trace_.status_ == TraceStatus::Running)) {
    amd::ScopedLock traceLock(&trace_mutex_);
    RgpSqttMarkerBarrierStart marker = {};

    marker.identifier = RgpSqttMarkerIdentifierBarrierStart;
    if (trace_.begin_queue_ != nullptr) {
      marker.cbId = trace_.begin_queue_->queue(MainEngine).cmdBufId();
    }
    marker.dword02 = data.reason;
    marker.internal = true;

    WriteMarker(gpu, &marker, sizeof(marker));
  }
}

// ================================================================================================
void RgpCaptureMgr::WriteBarrierEndMarker(const VirtualGPU* gpu,
                                          const Pal::Developer::BarrierData& data) const {
  if (rgp_server_->TracesEnabled() && (trace_.status_ == TraceStatus::Running)) {
    amd::ScopedLock traceLock(&trace_mutex_);
    // Copy the operations part and include the same data from previous markers
    // within the same barrier sequence to create a full picture of all cache
    // syncs and pipeline stalls.
    auto operations = data.operations;

    operations.pipelineStalls.u16All |= 0;
    operations.caches.u16All |= 0;

    RgpSqttMarkerBarrierEnd marker = {};

    marker.identifier = RgpSqttMarkerIdentifierBarrierEnd;
    if (trace_.begin_queue_ != nullptr) {
      marker.cbId = trace_.begin_queue_->queue(MainEngine).cmdBufId();
    }
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

    marker.numLayoutTransitions = 0;

    WriteMarker(gpu, &marker, sizeof(marker));
  }
}

// ================================================================================================
// Inserts a user event string marker
void RgpCaptureMgr::WriteUserEventMarker(const VirtualGPU* gpu,
                                         RgpSqttMarkerUserEventType eventType,
                                         const std::string& name) const {
  memset(user_event_, 0, sizeof(RgpSqttMarkerUserEventWithString));

  user_event_->header.identifier = RgpSqttMarkerIdentifierUserEvent;
  user_event_->header.dataType = eventType;

  size_t markerSize = sizeof(user_event_->header);

  if ((eventType != RgpSqttMarkerUserEventPop)) {
    size_t strLength =
        std::min(name.size(), RgpSqttMaxUserEventStringLengthInDwords * sizeof(uint32_t));
    for (uint32_t charIdx = 0; charIdx < strLength; ++charIdx) {
      uint32_t c = static_cast<uint32_t>(name[charIdx]);
      user_event_->stringData[charIdx / 4] |= (c << (8 * (charIdx % 4)));
      user_event_->stringLength = static_cast<uint32_t>(strLength);
    }

    // Every data type other than Pop includes a string length
    markerSize += sizeof(uint32_t);

    // Include string length (padded up to the nearest dword)
    markerSize += sizeof(uint32_t) * ((strLength + sizeof(uint32_t) - 1) / sizeof(uint32_t));
  }

  WriteMarker(gpu, user_event_, markerSize);
}

// ================================================================================================
// Inserts a compute bind marker
void RgpCaptureMgr::WriteComputeBindMarker(const VirtualGPU* gpu, uint64_t api_hash) const {
  RgpSqttMarkerPipelineBind marker = {};

  marker.identifier = RgpSqttMarkerIdentifierBindPipeline;
  marker.cbID = gpu->queue(MainEngine).cmdBufId();
  ;
  marker.bindPoint = 1;

  memcpy(marker.apiPsoHash, &api_hash, sizeof(api_hash));
  WriteMarker(gpu, &marker, sizeof(marker));
}

}  // namespace amd::pal

#endif  // PAL_GPUOPEN_OCL
