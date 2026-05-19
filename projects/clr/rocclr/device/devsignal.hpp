/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "top.hpp"

namespace amd {
class Device;
};

namespace amd::device {

// Light abstraction over HSA/PAL signals
class Signal {
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

  // Atomically loads the current value of the signal
  virtual uint64_t Load() { return 0; }

  // Exports the signal as an IPC handle into the provided buffer
  virtual bool IpcExport(void* handle, size_t handle_size) { return false; }

  // Initializes this signal from an IPC handle (alternative to Init for imported signals)
  virtual bool IpcImport(const void* handle, size_t handle_size,
                         const amd::Device* dev = nullptr) { return false; }

  // Return the handle to the underlying amd_signal_t object
  virtual void* getHandle() { return nullptr; }

  // Return a GPU-accessible handle (may differ from getHandle for IPC signals)
  virtual void* getGpuHandle() { return getHandle(); }
};

};  // namespace amd::device