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

//! IPC-capable signal using hsa_amd_ipc_signal_create/attach.
//! Used directly as completion_signal / dep_signal on AQL barrier packets.
class IpcSignal : public device::Signal {
 private:
  hsa_signal_t signal_;
  void* gpu_ptr_ = nullptr;  // GPU-accessible pointer from memory_lock (if needed)

 public:
  IpcSignal() { signal_.handle = 0; }
  ~IpcSignal() override;

  bool Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) override;

  uint64_t Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) override;
  void Reset(uint64_t value) override;
  uint64_t Load() override;
  bool IpcExport(void* handle, size_t handle_size) override;
  bool IpcImport(const void* handle, size_t handle_size,
                 const amd::Device* dev = nullptr) override;
  void* getHandle() override { return reinterpret_cast<void*>(signal_.handle); }
  void* getGpuHandle() override {
    return gpu_ptr_ ? gpu_ptr_ : reinterpret_cast<void*>(signal_.handle);
  }
};

};  // namespace amd::roc
