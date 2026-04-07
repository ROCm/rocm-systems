/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/rocm/rocdevice.hpp"
#include "device/devsignal.hpp"

namespace amd::roc {

class Signal : public device::Signal {
 private:
  hsa_signal_t signal_;

 public:
  ~Signal() override;

  bool Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) override;

  uint64_t Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) override;

  void Reset(uint64_t value) override;

  void* getHandle() override { return reinterpret_cast<void*>(signal_.handle); }
};

};  // namespace amd::roc
