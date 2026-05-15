/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/devsignal.hpp"

#include <amd_hsa_signal.h>

#include "palEvent.h"

namespace amd::pal {

class Device;

class Signal : public device::Signal {
 private:
  const Device* dev_;
  amd_signal_t* amdSignal_;
  Util::Event event_;

 public:
  ~Signal() override;

  bool Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) override;

  uint64_t Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) override;

  void Reset(uint64_t value) override;

  void* getHandle() override { return amdSignal_; }
};

};  // namespace amd::pal
