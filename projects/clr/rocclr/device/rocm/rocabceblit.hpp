/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// rocblit.hpp is not standalone: it needs the HSA types plus complete Device,
// VirtualGPU, and ProfilingSignal definitions. rocdevice.hpp is the umbrella
// header that provides all of them, so include it first to keep this standalone.
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocabce.hpp"
#include "device/rocm/rocblit.hpp"

#include <vector>

namespace amd::roc {

//! Blit manager that selects ABCE at the same public buffer-copy entry points
//! where KernelBlitManager selects between shader and ROCr paths.
class AbceBlitManager : public KernelBlitManager {
 public:
  AbceBlitManager(VirtualGPU& gpu, Setup setup = Setup()) : KernelBlitManager(gpu, setup) {}

  ~AbceBlitManager() override = default;

  bool copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                  const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                  const amd::Coord3D& size, bool entire = false,
                  amd::CopyMetadata copyMetadata = amd::CopyMetadata()) const override;

  bool readBuffer(device::Memory& srcMemory, void* dstHost, const amd::Coord3D& origin,
                  const amd::Coord3D& size, bool entire = false,
                  amd::CopyMetadata copyMetadata = amd::CopyMetadata()) const override;

  bool writeBuffer(const void* srcHost, device::Memory& dstMemory, const amd::Coord3D& origin,
                   const amd::Coord3D& size, bool entire = false,
                   amd::CopyMetadata copyMetadata = amd::CopyMetadata()) const override;

  //! Maps and submits the complete SDMA batch as one ABCE plan.
  bool submitCopyBatch(const std::vector<amd::BatchCopyOp>& copyOps,
                       const std::vector<hsa_signal_t>* externalWaitEvents = nullptr,
                       std::vector<ProfilingSignal*>* outBatchSignals = nullptr) const override;

 private:
  enum class CopyResult { kNotSubmitted, kSubmitted, kFailed };

  CopyResult submitCopy(address dst, hsa_agent_t dstAgent, uint32_t dstDriverNodeId,
                        const_address src, hsa_agent_t srcAgent, uint32_t srcDriverNodeId,
                        size_t size) const;

  CopyResult copyHostBuffer(const_address hostSrc, address hostDst, size_t size,
                            bool hostToDevice) const;

  //! Makes the calling device the policy source for device-to-device peer copies.
  void normalizePeerEndpoints(hsa_agent_t& srcAgent, uint32_t& srcDriverNodeId,
                              hsa_agent_t& dstAgent, uint32_t& dstDriverNodeId) const;

  //! Converts one CLR batch descriptor into its ABCE equivalent.
  bool describeCopy(const amd::BatchCopyOp& op, abce_copy_op_t& copy) const;

  //! Decodes and hex-dumps every SDMA packet in one plan frame at DETAIL_DEBUG.
  void logFramePackets(const abce_plan_frame_t& frame, uint32_t frame_idx, const char* label,
                       amd::LogMask mask) const;

  AbceBlitManager(const AbceBlitManager&) = delete;
  AbceBlitManager& operator=(const AbceBlitManager&) = delete;
};

}  // namespace amd::roc
