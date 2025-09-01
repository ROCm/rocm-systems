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

  uint64_t AddElfBinary(const void* exe_binary, size_t exe_binary_size, const void* elf_binary,
                        size_t elf_binary_size, Pal::IGpuMemory* pGpuMemory,
                        size_t offset) override;

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

  PAL_DISALLOW_DEFAULT_CTOR(UberTraceCaptureMgr);
  PAL_DISALLOW_COPY_AND_ASSIGN(UberTraceCaptureMgr);
};

}  // namespace amd::pal
