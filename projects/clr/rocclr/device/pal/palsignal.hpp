/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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
