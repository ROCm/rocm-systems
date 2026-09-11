// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_fixup.hpp"
#include "aql_resident_kernels.hpp"
#include "aql_resident_storage.hpp"
#include "rocdevice.hpp"
#include "rocvirtual.hpp"
#include "platform/kernel.hpp"
#include <cstring>

namespace amd::roc::aql_resident {

BindingKernel::~BindingKernel() { kernel_->release(); }

std::unique_ptr<BindingKernel> BindingKernel::create(VirtualGPU& gpu) {
  auto& device = const_cast<Device&>(gpu.dev());
  if (device.settings().ext_dispatch_packet_) return {};
  // Queue initialization owns creation of the shared internal program.
  if (!device.blitProgram()) return {};
  auto* program = device.blitProgram()->program_;
  const auto* symbol = program->findSymbol(kBindingKernelName);
  if (!symbol) return {};
  auto* kernel = new amd::Kernel(*program, *symbol, kBindingKernelName);
  std::unique_ptr<BindingKernel> result;
  try {
    result.reset(new BindingKernel(kernel, gpu));
  } catch (...) {
    kernel->release();
    throw;
  }
  if (!device.validateKernel(*kernel, &gpu)) return {};
  const auto* info = kernel->getDeviceKernel(device)->workGroupInfo();
  if (info->privateMemSize_ || info->scratchRegs_ || (info->usedStackSize_ & 1)) return {};
  // The internal source contract is three GPU pointers and one uint count.
  for (size_t i = 0; i < 4; ++i) {
    if (kernel->signature().at(i).size_ != (i < 3 ? 8 : 4)) return {};
  }
  return result;
}

bool BindingKernel::prepare(VirtualGPU& gpu, uint64_t bindings, uint64_t entries,
                            const ResidentImage& target, uint32_t count,
                            hsa_kernel_dispatch_packet_t& packet) {
  if (&gpu != queue_ || &gpu.dev() != &target.device() || !count ||
      !bindings || !entries || (bindings & 7) || (entries & 7)) return false;
  const uint64_t addresses[] = {bindings, entries, reinterpret_cast<uintptr_t>(target.base())};
  auto* parameters = kernel_->parameters().values();
  for (size_t i = 0; i < 3; ++i) {
    const auto& argument = kernel_->signature().at(i);
    std::memcpy(parameters + argument.offset_, &addresses[i], sizeof(uint64_t));
    // Direct GPU VA arguments, as in KernelBlitManager::setArgument.
    reinterpret_cast<amd::Memory**>(parameters + kernel_->parameters().memoryObjOffset())
        [argument.info_.arrayIndex_] = nullptr;
  }
  std::memcpy(parameters + kernel_->signature().at(3).offset_, &count, sizeof(count));
  const size_t local[] = {64};
  const size_t global[] = {(size_t(count) + 63) & ~size_t(63)};
  amd::NDRangeContainer range(1, nullptr, global, local);
  if (!gpu.submitKernelInternal(range, *kernel_, parameters, nullptr, 0,
                                nullptr, &packet, false, true)) return false;
  // Same-queue earlier consumers must finish before this mutable image is
  // patched. A dirty-target JUMP is additionally required after this dispatch.
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH |
      (1u << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  return true;
}
}  // namespace amd::roc::aql_resident
