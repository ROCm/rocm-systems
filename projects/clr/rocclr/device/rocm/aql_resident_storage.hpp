// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"
#include <hsa/hsa.h>
#include <memory>
#include <vector>

namespace amd::roc {
class Device;
namespace aql_resident {

// Published executable storage for one physical-queue variant. The program
// and all submissions referring to it share ownership. Its destructor may
// run only after the last completion reference is released.
class ResidentImage final {
 public:
  static hsa_status_t publish(Device& device, const std::vector<uint8_t>& hostImage,
                              std::shared_ptr<ResidentImage>& output);
  ~ResidentImage();
  ResidentImage(const ResidentImage&) = delete;
  ResidentImage& operator=(const ResidentImage&) = delete;
  void* base() const { return base_; }
  size_t size() const { return size_; }
  Device& device() const { return *device_; }

 private:
  explicit ResidentImage(Device& device);
  amd::SharedReference<amd::Device> owner_;
  Device* device_;
  void* base_ = nullptr;
  size_t size_ = 0;
};

}  // namespace aql_resident
}  // namespace amd::roc
