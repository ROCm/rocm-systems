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

#pragma once

#include <queue>
#include "device/pal/palcapturemgr.hpp"
#include "platform/commandqueue.hpp"
#include "device/blit.hpp"

// PAL headers
#include "palUtil.h"
#include "palPlatform.h"
#include "palCmdBuffer.h"
#include "palCmdAllocator.h"
#include "palQueue.h"
#include "palFence.h"
#include "palLinearAllocator.h"
#include "palHashMap.h"
#include "palQueue.h"
#include "palUtil.h"

namespace amd::pal {
class Settings;
class Device;
class VirtualGPU;
class HSAILKernel;

// ================================================================================================
enum class RgpSqqtBarrierReason : uint32_t {
  Invalid = 0,
  MemDependency = 0xC0000000,
  ProfilingControl = 0xC0000001,
  SignalSubmit = 0xC0000002,
  PostDeviceEnqueue = 0xC0000003,
  Unknown = 0xffffffff
};

}  // namespace amd::pal

#ifdef PAL_GPUOPEN_OCL
#include "protocols/rgpServer.h"
// gpuopen headers
#include "gpuopen.h"
// gpuutil headers
#include "gpuUtil/palGpaSession.h"

// PAL forward declarations
namespace Pal {
class ICmdBuffer;
class IFence;
class IQueueSemaphore;
struct PalPublicSettings;
class IGPuMemory;
}  // namespace Pal

// GPUOpen forward declarations
namespace DevDriver {
class DevDriverServer;
class IMsgChannel;
struct MessageBuffer;

namespace DriverControlProtocol {
enum struct DeviceClockMode : uint32_t;
class HandlerServer;
}  // namespace DriverControlProtocol

namespace SettingsProtocol {
class HandlerServer;
}

}  // namespace DevDriver

namespace amd::pal {

// ================================================================================================
// This class provides functionality to interact with the GPU Open Developer Mode message passing
// service and the rest of the driver.
class RgpCaptureMgr final : public ICaptureMgr {
 public:
  ~RgpCaptureMgr();

  static RgpCaptureMgr* Create(Pal::IPlatform* platform, const Device& device);

  void PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel, size_t x, size_t y,
                   size_t z) override;

  void PostDispatch(VirtualGPU* gpu) override;

  void FinishRGPTrace(VirtualGPU* gpu, bool aborted) override;

  void WriteBarrierStartMarker(const VirtualGPU* gpu,
                               const Pal::Developer::BarrierData& data) const override;

  void WriteBarrierEndMarker(const VirtualGPU* gpu,
                             const Pal::Developer::BarrierData& data) const override;

  bool RegisterTimedQueue(uint32_t queue_id, Pal::IQueue* iQueue, bool* debug_vmid) const override;

  Pal::Result TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                               const Pal::SubmitInfo& submitInfo) const override;

  bool Update(Pal::IPlatform* platform) override;

  uint64_t AddElfBinary(const void* exe_binary, size_t exe_binary_size, const void* elf_binary,
                        size_t elf_binary_size, Pal::IGpuMemory* pGpuMemory,
                        size_t offset) override;

 private:
  // Steps that an RGP trace goes through
  enum class TraceStatus {
    Idle = 0,   // No active trace and none requested
    Preparing,  // A trace has been requested but is not active yet because we are
                // currently sampling timing information over some number of lead frames.
    Running,    // SQTT and queue timing is currently active for all command buffer submits.
    WaitingForSqtt,
    WaitingForResults  // Tracing is no longer active, but all results are not yet ready.
  };

  // All per-device state to support RGP tracing
  struct TraceState {
    TraceStatus status_;  // Current trace status (idle, running, etc.)

    GpuEvent begin_sqtt_event_;  // Event that is signaled when a trace-end cmdbuf retires
    GpuEvent end_sqtt_event_;    // Event that is signaled when a trace-end cmdbuf retires
    GpuEvent end_event_;         // Event that is signaled when a trace-end cmdbuf retires

    VirtualGPU* begin_queue_;  // The queue that triggered starting SQTT

    GpuUtil::GpaSession* gpa_session_;  // GPA session helper object for building RGP data
    uint32_t gpa_sample_id_;            // Sample ID associated with the current trace
    bool queue_timing_;                 // Queue timing is enabled

    uint32_t prepared_disp_count_;  // Number of dispatches counted while preparing for a trace
    uint32_t sqtt_disp_count_;      // Number of dispatches counted while SQTT tracing is active
    mutable uint32_t current_event_id_;  // Current event ID
  };

  RgpCaptureMgr(Pal::IPlatform* platform, const Device& device);

  bool Init(Pal::IPlatform* platform);
  void Finalize();

  Pal::Result PrepareRGPTrace(VirtualGPU* pQueue);
  Pal::Result BeginRGPTrace(VirtualGPU* pQueue);
  Pal::Result EndRGPHardwareTrace(VirtualGPU* pQueue);
  Pal::Result EndRGPTrace(VirtualGPU* pQueue);
  void DestroyRGPTracing();
  Pal::Result CheckForTraceResults();
  static bool GpuSupportsTracing(const Pal::DeviceProperties& props, const Settings& settings);

  RgpSqttMarkerEvent BuildEventMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType api_type) const;
  void WriteMarker(const VirtualGPU* gpu, const void* data, size_t data_size) const;
  void WriteEventWithDimsMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType apiType, uint32_t x,
                                uint32_t y, uint32_t z) const;
  void WriteUserEventMarker(const VirtualGPU* gpu, RgpSqttMarkerUserEventType eventType,
                            const std::string& name) const;
  void WriteComputeBindMarker(const VirtualGPU* gpu, uint64_t api_hash) const;

  void WaitForDriverResume();

  void PostDeviceCreate();
  void PreDeviceDestroy();

  bool IsQueueTimingActive() const;

  const Device& device_;
  DevDriver::DevDriverServer* dev_driver_server_;
  DevDriver::RGPProtocol::RGPServer* rgp_server_;
  mutable amd::Monitor trace_mutex_;
  TraceState trace_;
  RgpSqttMarkerUserEventWithString* user_event_;

  uint32_t num_prep_disp_;
  uint32_t max_sqtt_disp_;  // Maximum number of the dispatches allowed in the trace
  uint32_t trace_gpu_mem_limit_;
  uint32_t global_disp_count_;

  uint32_t se_mask_;                 // Shader engine mask
  uint64_t perf_counter_mem_limit_;  // Memory limit for perf counters
  uint32_t perf_counter_frequency_;  // Counter sample frequency

  std::vector<GpuUtil::PerfCounterId> perf_counter_ids_;  // List of perf counter ids

  union {
    struct {
      uint32_t trace_enabled_ : 1;          // True if tracing is currently enabled (master flag)
      uint32_t inst_tracing_enabled_ : 1;   // Enable instruction-level SQTT tokens
      uint32_t perf_counters_enabled_ : 1;  // True if perf counters are enabled
      uint32_t static_vm_id_ : 1;           // Static VM ID can be used for capture
    };
    uint32_t value_;
  };

  PAL_DISALLOW_DEFAULT_CTOR(RgpCaptureMgr);
  PAL_DISALLOW_COPY_AND_ASSIGN(RgpCaptureMgr);
};

// ================================================================================================
// Returns true if queue operations are currently being timed by RGP traces.
inline bool RgpCaptureMgr::IsQueueTimingActive() const {
  return (trace_.queue_timing_ &&
          (trace_.status_ == TraceStatus::Running || trace_.status_ == TraceStatus::Preparing ||
           trace_.status_ == TraceStatus::WaitingForSqtt));
}
}  // namespace amd::pal
#else   // PAL_GPUOPEN_OCL
namespace amd::pal {
class RgpCaptureMgr {
 public:
  static RgpCaptureMgr* Create(Pal::IPlatform* platform, const Device& device) { return nullptr; }
  Pal::Result TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                               Pal::SubmitInfo& submitInfo) const {
    return Pal::Result::Success;
  }
  void PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel, size_t x, size_t y, size_t z) {}
  void PostDispatch(VirtualGPU* gpu) {}
  void FinishRGPTrace(VirtualGPU* gpu, bool aborted) {}
  bool RegisterTimedQueue(uint32_t queue_id, Pal::IQueue* iQueue, bool* debug_vmid) const {
    return true;
  }
  bool Update(Pal::IPlatform* platform) const { return true; }
  bool AddElfBinary(const void* exe_binary, size_t exe_binary_size, const void* elf_binary,
                    size_t elf_binary_size, Pal::IGpuMemory* pGpuMemory, size_t offset) {
    return true;
  }
};
}  // namespace amd::pal
#endif  // PAL_GPUOPEN_OCL
