/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/pal/palcapturemgr.hpp"
#include "thread/monitor.hpp"

namespace DevDriver {
class DevDriverServer;
}

namespace GpuUtil {
class TraceSession;
class RenderOpTraceController;
class CodeObjectTraceSource;
class QueueTimingsTraceSource;
}  // namespace GpuUtil

namespace amd::pal {

// ================================================================================================
class UberTraceCaptureMgr final : public ICaptureMgr {
 public:
  static UberTraceCaptureMgr* Create(Pal::IPlatform* platform, const Device& device);

  ~UberTraceCaptureMgr();

  bool Update(Pal::IPlatform* platform) override;

  void PreDispatch(VirtualGPU* gpu, const pal::Kernel& kernel, size_t x, size_t y,
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

  uint64_t AddElfBinary(const void* exe_binary, size_t exe_binary_size, const void* elf_binary,
                        size_t elf_binary_size, Pal::IGpuMemory* pGpuMemory,
                        size_t offset) override;

  VirtualGPU* GetCurrentGPU() {
      return current_gpu_;
  }

 private:
  UberTraceCaptureMgr(Pal::IPlatform* platform, const Device& device);
  bool Init(Pal::IPlatform* platform);
  void WaitForDriverResume();

  void PreDeviceDestroy();

  bool IsQueueTimingActive() const;

  bool CreateUberTraceResources(Pal::IPlatform* platform);
  void DestroyUberTraceResources();

  void WriteMarker(const VirtualGPU* gpu, const void* data, size_t data_size) const;
  void WriteComputeBindMarker(const VirtualGPU* gpu, uint64_t api_hash) const;
  void WriteEventWithDimsMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType apiType, uint32_t x,
                                uint32_t y, uint32_t z) const;

  const Device& device_;
  DevDriver::DevDriverServer* dev_driver_server_;
  uint64_t global_disp_count_;
  RgpSqttMarkerUserEventWithString* user_event_;
  mutable uint32_t current_event_id_;

  mutable amd::Monitor trace_mutex_;
  GpuUtil::TraceSession* trace_session_;
  GpuUtil::RenderOpTraceController* trace_controller_;
  GpuUtil::CodeObjectTraceSource* code_object_trace_source_;
  GpuUtil::QueueTimingsTraceSource* queue_timings_trace_source_;

  VirtualGPU*                       current_gpu_;
  bool                              registered_trace_state_callback_;

  PAL_DISALLOW_DEFAULT_CTOR(UberTraceCaptureMgr);
  PAL_DISALLOW_COPY_AND_ASSIGN(UberTraceCaptureMgr);
};

}  // namespace amd::pal
