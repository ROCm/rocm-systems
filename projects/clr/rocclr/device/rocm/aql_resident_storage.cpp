// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_storage.hpp"
#include "rocdevice.hpp"
#include "rocrctx.hpp"
#include <limits>
#include <new>

namespace amd::roc::aql_resident {

ResidentImage::ResidentImage(Device& device) : owner_(device), device_(&device) {}

ResidentImage::~ResidentImage() {
  if (base_) device_->memFree(base_, size_);
}

hsa_status_t ResidentImage::publish(Device& device, const std::vector<uint8_t>& hostImage,
                                    std::shared_ptr<ResidentImage>& output) try {
  // One variant begins on a packet boundary. Limit relative addressing to
  // the fixup ABI; packet-count validation belongs to the program compiler.
  if (hostImage.empty() || hostImage.size() % 64 ||
      hostImage.size() > std::numeric_limits<uint32_t>::max()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto image = std::shared_ptr<ResidentImage>(new ResidentImage(device));
  image->size_ = hostImage.size();
  amd::Device::AllocationFlags flags{};
  flags.executable_ = true;
  // No atomics/pseudo-fine-grain flag: ROCclr selects the coarse GPU pool,
  // owned by this agent. Cross-device access is not needed.
  image->base_ = device.deviceLocalAlloc(image->size_, flags, false);
  if (!image->base_) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  if (reinterpret_cast<uintptr_t>(image->base_) % 64) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  const hsa_status_t status = Hsa::memory_copy(image->base_, hostImage.data(), image->size_);
  if (status != HSA_STATUS_SUCCESS) return status;
  // Synchronous publication completes before exposing the object. There is
  // deliberately no host patch method; subsequent mutation uses ordered GPU
  // fixup. Every failure above leaves the caller's previous object untouched.
  output = std::move(image);
  return HSA_STATUS_SUCCESS;
} catch (const std::bad_alloc&) {
  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

}  // namespace amd::roc::aql_resident
