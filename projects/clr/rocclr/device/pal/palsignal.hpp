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
  const Device* dev_ = nullptr;
  amd_signal_t* amdSignal_ = nullptr;
  Util::Event event_;
  bool eventRegistered_ = false;

 public:
  ~Signal() override;

  bool Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) override;

  uint64_t Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) override;

  void Reset(uint64_t value) override;
  uint64_t Load() override { return amdSignal_->value; }

  void* getHandle() override { return amdSignal_; }
  Doorbell getDoorbell() override {
    return {reinterpret_cast<uint64_t*>(const_cast<int64_t*>(&amdSignal_->value)),
            reinterpret_cast<uint64_t*>(amdSignal_->event_mailbox_ptr), amdSignal_->event_id};
  }
};

};  // namespace amd::pal
