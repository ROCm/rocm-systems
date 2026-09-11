// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <memory>
#include <hsa/hsa.h>

namespace amd { class Kernel; }
namespace amd::roc {
class VirtualGPU;
namespace aql_resident {
class ResidentImage;

// One instance per VirtualGPU, used under that queue's execution lock. GPU
// input/target lifetime and atomic reservation of fixup+root are the caller's
// responsibility. Preparation does not publish the fixup packet.
class BindingKernel final {
 public:
  static std::unique_ptr<BindingKernel> create(VirtualGPU& gpu);
  ~BindingKernel();
  BindingKernel(const BindingKernel&) = delete;
  BindingKernel& operator=(const BindingKernel&) = delete;
  bool prepare(VirtualGPU& gpu, uint64_t bindings, uint64_t entries,
               const ResidentImage& target, uint32_t count,
               hsa_kernel_dispatch_packet_t& packet);
 private:
  BindingKernel(amd::Kernel* kernel, VirtualGPU& gpu) : kernel_(kernel), queue_(&gpu) {}
  amd::Kernel* kernel_;
  VirtualGPU* queue_;
};
}  // namespace aql_resident
}  // namespace amd::roc
