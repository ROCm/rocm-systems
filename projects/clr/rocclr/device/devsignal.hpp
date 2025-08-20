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

#include "top.hpp"

namespace amd {
class Device;
};

namespace amd::device {

// Light abstraction over HSA/PAL signals
class Signal : public amd::HeapObject {
 public:
  enum class Condition : uint32_t {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Gte = 3,
  };

  enum class WaitState : uint32_t {
    Blocked = 0,
    Active = 1,
  };

 protected:
  WaitState ws_;

 public:
  virtual ~Signal() {}

  virtual bool Init(const amd::Device& dev, uint64_t init, WaitState ws) { return false; }

  // Blocks the current thread untill the condition c is satisfied
  // or amount of time specified by timeout passes
  virtual uint64_t Wait(uint64_t value, Condition c, uint64_t timeout) { return -1; }

  // Atomically sets the current value of the signal
  virtual void Reset(uint64_t value) {}

  // Return the handle to the underlying amd_signal_t object
  virtual void* getHandle() { return nullptr; }
};

};  // namespace amd::device